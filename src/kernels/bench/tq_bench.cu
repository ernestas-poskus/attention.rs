// TurboQuant k8v4 decode bench at MOSS-TTS shapes (Qwen3 8B: 32 q heads,
// 8 kv heads, head_dim 128, block_size 32, 36 layers). Every launch reads a
// different layer's cache, so the KV stream comes from DRAM as in a real
// decode step. Compares the shipped single-pass and split-K kernels with a
// GQA-grouped split kernel that reads each kv head once for its q heads.
//
// nvcc -O3 -arch=sm_86 -std=c++17 -I../src tq_bench.cu ../src/flash/flash_instantiate.cu ../src/flash/flash_decode.cu
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

extern "C" void call_flash_tq_decode_k8v4(const void*, const void*, const void*, const void*, void*,
    const int*, const int*, unsigned, unsigned, unsigned, unsigned, unsigned, float, unsigned, unsigned,
    float, const float*, unsigned, int, int64_t);
extern "C" void call_flash_tq_decode_k8v4_splitk(const void*, const void*, const void*, const void*, void*,
    const int*, const int*, unsigned, unsigned, unsigned, unsigned, unsigned, float, unsigned, unsigned,
    unsigned, float, const float*, unsigned, int, int64_t);
extern "C" void call_flash_decode_paged_reduce(const void*, void*, unsigned, unsigned, unsigned, unsigned, int, int64_t);

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// GQA-grouped split-K k8v4 decode. Grid (num_kv_heads, num_splits, num_seqs),
// NW warps. Each warp walks a contiguous slice of the split; per position a
// lane loads its 4 fp8 key bytes and 2 packed value bytes once and scores
// them against all G query heads of the kv head. K, V and the value scale
// for a batch of BC positions are loaded together before any math, so a
// batch costs one memory round trip, not two.
// ---------------------------------------------------------------------------
template <int G, int NW, int BC>
__global__ void __launch_bounds__(NW * 32) tq_k8v4_gqa_splitk(
    const __nv_bfloat16* __restrict__ Q, const unsigned char* __restrict__ K_cache,
    const float* __restrict__ V_absmax, const unsigned char* __restrict__ V_quant,
    float* __restrict__ workspace, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, unsigned max_blocks_per_seq, unsigned num_q_heads,
    unsigned num_kv_heads, unsigned block_size, float inv_sqrt_d, unsigned num_splits,
    unsigned q_stride, const float* __restrict__ k_scale_ptr) {
    constexpr int HD = 128, VEC = 4;
    const unsigned kv_head = blockIdx.x, split = blockIdx.y, seq = blockIdx.z;
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned seq_len = (unsigned)seq_lens[seq];
    const unsigned split_size = (seq_len + num_splits - 1) / num_splits;
    unsigned s0 = split * split_size, s1 = min(s0 + split_size, seq_len);
    if (s0 > s1) s0 = s1;
    const unsigned per_warp = (s1 - s0 + NW - 1) / NW;
    unsigned w0 = min(s0 + warp * per_warp, s1), w1 = min(w0 + per_warp, s1);
    const float k_scale = k_scale_ptr ? k_scale_ptr[kv_head] : 1.0f;

    float q[G][VEC];
#pragma unroll
    for (int g = 0; g < G; g++) {
        const __nv_bfloat16* qp = Q + (size_t)seq * q_stride + (size_t)(kv_head * G + g) * HD + lane * VEC;
        uint2 v = __ldg((const uint2*)qp);
        __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&v.x), b = *reinterpret_cast<__nv_bfloat162*>(&v.y);
        q[g][0] = __low2float(a) * inv_sqrt_d; q[g][1] = __high2float(a) * inv_sqrt_d;
        q[g][2] = __low2float(b) * inv_sqrt_d; q[g][3] = __high2float(b) * inv_sqrt_d;
    }
    float m[G], l[G], o[G][VEC];
#pragma unroll
    for (int g = 0; g < G; g++) { m[g] = -1e30f; l[g] = 0.f; for (int i = 0; i < VEC; i++) o[g][i] = 0.f; }

    const int* bt = block_tables + (size_t)seq * max_blocks_per_seq;
    const size_t k_row = (size_t)num_kv_heads * HD, v_row = (size_t)num_kv_heads * (HD / 2);
    for (unsigned p0 = w0; p0 < w1; p0 += BC) {
        unsigned kw[BC]; unsigned short vw[BC]; float va[BC];
#pragma unroll
        for (int b = 0; b < BC; b++) {
            const unsigned p = min(p0 + b, w1 - 1);
            const size_t slot = (size_t)bt[p / block_size] * block_size + p % block_size;
            kw[b] = __ldg((const unsigned*)(K_cache + slot * k_row + kv_head * HD + lane * VEC));
            vw[b] = __ldg((const unsigned short*)(V_quant + slot * v_row + kv_head * (HD / 2) + lane * 2));
            va[b] = __ldg(V_absmax + slot * num_kv_heads + kv_head);
        }
        float s[BC][G];
#pragma unroll
        for (int b = 0; b < BC; b++) {
            __half2_raw h0 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kw[b] & 0xFFFF), __NV_E4M3);
            __half2_raw h1 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kw[b] >> 16), __NV_E4M3);
            float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&h0));
            float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&h1));
#pragma unroll
            for (int g = 0; g < G; g++)
                s[b][g] = (q[g][0] * f0.x + q[g][1] * f0.y + q[g][2] * f1.x + q[g][3] * f1.y) * k_scale;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
#pragma unroll
            for (int b = 0; b < BC; b++)
#pragma unroll
                for (int g = 0; g < G; g++) s[b][g] += __shfl_xor_sync(0xffffffff, s[b][g], off);
        float vd[BC][VEC];
#pragma unroll
        for (int b = 0; b < BC; b++) {
            const float sc = va[b] * 0.13333333f;
            const unsigned lo = vw[b] & 0xFF, hi = vw[b] >> 8;
            vd[b][0] = ((float)(lo & 0xF) - 7.5f) * sc; vd[b][1] = ((float)(lo >> 4) - 7.5f) * sc;
            vd[b][2] = ((float)(hi & 0xF) - 7.5f) * sc; vd[b][3] = ((float)(hi >> 4) - 7.5f) * sc;
        }
#pragma unroll
        for (int g = 0; g < G; g++) {
            float mn = m[g];
#pragma unroll
            for (int b = 0; b < BC; b++) if (p0 + b < w1) mn = fmaxf(mn, s[b][g]);
            const float eo = __expf(m[g] - mn);
            l[g] *= eo;
#pragma unroll
            for (int i = 0; i < VEC; i++) o[g][i] *= eo;
#pragma unroll
            for (int b = 0; b < BC; b++) {
                const float e = (p0 + b < w1) ? __expf(s[b][g] - mn) : 0.f;
                l[g] += e;
#pragma unroll
                for (int i = 0; i < VEC; i++) o[g][i] += e * vd[b][i];
            }
            m[g] = mn;
        }
    }

    __shared__ float sm[NW][G], sl[NW][G], so[NW][G][HD];
#pragma unroll
    for (int g = 0; g < G; g++) {
        if (lane == 0) { sm[warp][g] = m[g]; sl[warp][g] = l[g]; }
#pragma unroll
        for (int i = 0; i < VEC; i++) so[warp][g][lane * VEC + i] = o[g][i];
    }
    __syncthreads();
    for (unsigned idx = threadIdx.x; idx < G * HD; idx += NW * 32) {
        const unsigned g = idx / HD, d = idx % HD;
        float M = -1e30f;
        for (int w = 0; w < NW; w++) if (sl[w][g] > 0.f) M = fmaxf(M, sm[w][g]);
        float L = 0.f, O = 0.f;
        for (int w = 0; w < NW; w++) if (sl[w][g] > 0.f) {
            const float e = __expf(sm[w][g] - M);
            L += sl[w][g] * e; O += so[w][g][d] * e;
        }
        float* ws = workspace + (((size_t)seq * num_q_heads + kv_head * G + g) * num_splits + split) * (HD + 2);
        ws[d] = O;
        if (d == 0) { ws[HD] = M; ws[HD + 1] = L; }
    }
}

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static float e4m3(unsigned char b) {
    const int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
    const float v = e == 0 ? ldexpf((float)m / 8.f, -6) : ldexpf(1.f + (float)m / 8.f, e - 7);
    return s ? -v : v;
}

int main() {
    const unsigned NQ = 32, NKV = 8, HD = 128, BS = 32, NCTX = 2304, NSEQ = 5, LAYERS = 36;
    const unsigned BPS = NCTX / BS, NBLK = BPS * NSEQ;
    const size_t k_layer = (size_t)NBLK * BS * NKV * HD, v_layer = k_layer / 2, a_layer = (size_t)NBLK * BS * NKV;
    std::mt19937 rng(7);
    std::vector<unsigned char> hk(k_layer * LAYERS), hv(v_layer * LAYERS);
    std::vector<float> ha(a_layer * LAYERS);
    for (auto& x : hk) x = (unsigned char)(0x20 + rng() % 0x30) | (rng() & 0x80);
    for (auto& x : hv) x = (unsigned char)rng();
    for (auto& x : ha) x = 0.5f + (rng() % 1000) / 1000.f;
    std::vector<uint16_t> hq(NSEQ * NQ * HD);
    for (auto& x : hq) { float f = ((int)(rng() % 2001) - 1000) / 1000.f; uint32_t u; memcpy(&u, &f, 4); x = u >> 16; }
    std::vector<int> bt(NBLK);
    for (unsigned i = 0; i < NBLK; i++) bt[i] = i;
    std::shuffle(bt.begin(), bt.end(), rng);

    unsigned char *dk, *dv; float *da, *dws; void *dq, *do1, *do2; int *dbt, *dsl;
    CK(cudaMalloc(&dk, hk.size())); CK(cudaMalloc(&dv, hv.size())); CK(cudaMalloc(&da, ha.size() * 4));
    CK(cudaMalloc(&dq, hq.size() * 2)); CK(cudaMalloc(&do1, hq.size() * 2)); CK(cudaMalloc(&do2, hq.size() * 2));
    CK(cudaMalloc(&dws, (size_t)NSEQ * NQ * 64 * (HD + 2) * 4));
    CK(cudaMalloc(&dbt, bt.size() * 4)); CK(cudaMalloc(&dsl, NSEQ * 4));
    CK(cudaMemcpy(dk, hk.data(), hk.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, hv.data(), hv.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice));
    const float isd = 1.f / sqrtf((float)HD);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int ITERS = 360;

    auto base = [&](int L, void* out) {
        call_flash_tq_decode_k8v4(dq, dk + L * k_layer, da + L * a_layer, dv + L * v_layer, out, dbt, dsl, BPS,
            NQ, NKV, HD, BS, isd, NSEQ, NQ * HD, 0.f, nullptr, 0, 1, 0);
    };
    auto split = [&](int L, void* out, unsigned S) {
        call_flash_tq_decode_k8v4_splitk(dq, dk + L * k_layer, da + L * a_layer, dv + L * v_layer, dws, dbt, dsl, BPS,
            NQ, NKV, HD, BS, isd, S, NSEQ, NQ * HD, 0.f, nullptr, 0, 1, 0);
        call_flash_decode_paged_reduce(dws, out, NQ, HD, S, NSEQ, 1, 0);
    };
    auto gqa = [&](int L, void* out, unsigned S, int nw) {
        dim3 grid(NKV, S, NSEQ);
        if (nw == 4)
            tq_k8v4_gqa_splitk<4, 4, 8><<<grid, 128>>>((const __nv_bfloat16*)dq, dk + L * k_layer, da + L * a_layer,
                dv + L * v_layer, dws, dbt, dsl, BPS, NQ, NKV, BS, isd, S, NQ * HD, nullptr);
        else
            tq_k8v4_gqa_splitk<4, 8, 8><<<grid, 256>>>((const __nv_bfloat16*)dq, dk + L * k_layer, da + L * a_layer,
                dv + L * v_layer, dws, dbt, dsl, BPS, NQ, NKV, BS, isd, S, NQ * HD, nullptr);
        call_flash_decode_paged_reduce(dws, out, NQ, HD, S, NSEQ, 1, 0);
    };
    auto maxdiff = [&](void* a, void* b) {
        std::vector<uint16_t> x(hq.size()), y(hq.size());
        CK(cudaMemcpy(x.data(), a, x.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(y.data(), b, y.size() * 2, cudaMemcpyDeviceToHost));
        float md = 0, mx = 0;
        for (size_t i = 0; i < x.size(); i++) { md = fmaxf(md, fabsf(bf2f(x[i]) - bf2f(y[i]))); mx = fmaxf(mx, fabsf(bf2f(x[i]))); }
        if (!(md == md)) md = INFINITY;
        return md / fmaxf(mx, 1e-6f);
    };
    auto cpu_err = [&](void* dev, const std::vector<int>& sl) {
        std::vector<uint16_t> x(hq.size());
        CK(cudaMemcpy(x.data(), dev, x.size() * 2, cudaMemcpyDeviceToHost));
        float md = 0, mx = 0;
        for (unsigned s = 0; s < NSEQ; s++) for (unsigned h = 0; h < NQ; h++) {
            const unsigned kvh = h / (NQ / NKV);
            std::vector<double> sc(sl[s]); double M = -1e300;
            for (int p = 0; p < sl[s]; p++) {
                size_t slot = (size_t)bt[s * BPS + p / BS] * BS + p % BS; double d = 0;
                for (unsigned i = 0; i < HD; i++) d += bf2f(hq[(s * NQ + h) * HD + i]) * e4m3(hk[(slot * NKV + kvh) * HD + i]);
                sc[p] = d * isd; M = std::max(M, sc[p]);
            }
            std::vector<double> o(HD, 0.0); double L = 0;
            for (int p = 0; p < sl[s]; p++) {
                size_t slot = (size_t)bt[s * BPS + p / BS] * BS + p % BS; double e = exp(sc[p] - M); L += e;
                const double va = ha[slot * NKV + kvh] * 0.13333333;
                for (unsigned i = 0; i < HD; i++) {
                    unsigned char by = hv[(slot * NKV + kvh) * (HD / 2) + i / 2];
                    o[i] += e * (((i & 1) ? (by >> 4) : (by & 0xF)) - 7.5) * va;
                }
            }
            for (unsigned i = 0; i < HD; i++) {
                float r = (float)(o[i] / L), g = bf2f(x[(s * NQ + h) * HD + i]);
                md = fmaxf(md, fabsf(r - g)); mx = fmaxf(mx, fabsf(r));
            }
        }
        return md / mx;
    };
    auto timeit = [&](auto&& f) {
        for (int i = 0; i < 36; i++) f(i % LAYERS);
        CK(cudaDeviceSynchronize());
        cudaEventRecord(e0);
        for (int i = 0; i < ITERS; i++) f(i % LAYERS);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        return ms * 1000.f / ITERS;
    };

    for (unsigned ctx : {400u, 1400u, 2300u}) {
        std::vector<int> sl(NSEQ);
        for (unsigned s = 0; s < NSEQ; s++) sl[s] = std::max(1, (int)ctx - (int)(s * ctx / 10));
        CK(cudaMemcpy(dsl, sl.data(), NSEQ * 4, cudaMemcpyHostToDevice));
        double bytes = 0;
        for (int v : sl) bytes += (double)v * NKV * (HD + HD / 2 + 4);
        base(0, do1); CK(cudaDeviceSynchronize());
        float tb = timeit([&](int L) { base(L, do1); });
        printf("ctx %4u  base          %7.1f us  %6.0f GB/s(unique KV)  cpu_relerr %.2e\n", ctx, tb, bytes / tb / 1e3, cpu_err(do1, sl));
        for (unsigned S : {4u, 8u, 16u}) {
            split(0, do2, S); CK(cudaDeviceSynchronize());
            float d = cpu_err(do2, sl);
            float t = timeit([&](int L) { split(L, do2, S); });
            printf("ctx %4u  splitk S=%-2u   %7.1f us  %6.0f GB/s  relerr %.2e\n", ctx, S, t, bytes / t / 1e3, d);
        }
        for (int nw : {4, 8})
            for (unsigned S : {4u, 8u, 12u}) {
                gqa(0, do2, S, nw); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                float d = cpu_err(do2, sl);
                float t = timeit([&](int L) { gqa(L, do2, S, nw); });
                printf("ctx %4u  gqa nw=%d S=%-2u %7.1f us  %6.0f GB/s  relerr %.2e\n", ctx, nw, S, t, bytes / t / 1e3, d);
            }
    }
    return 0;
}
