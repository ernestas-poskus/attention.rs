/**
 * @brief TurboQuant k8v4 split-K decode, grouped by kv head (head_dim 128).
 *
 * The per-q-head kernels in flash_turboquant.cuh give every query head its
 * own block, so the four q heads of a GQA group each stream the same kv head
 * from memory, and each 8-position batch waits on the keys and then again on
 * the values. This kernel launches one block per (kv head, split, sequence):
 * a lane loads its 4 fp8 key bytes, 2 packed value bytes and the value scale
 * for a whole batch at once, then scores the batch against all kGroup query heads.
 * Partials go to the split-K workspace in the layout
 * flash_decode_paged_reduce reads, (head_dim + 2) floats per
 * (seq, q head, split): unnormalised output, running max, running sum.
 * Empty splits write l = 0, which the reduce skips.
 *
 * At MOSS-TTS shapes (32 q / 8 kv heads, five sequences, RTX 3090) it runs
 * the decode at ~210 GB/s of unique KV against ~60 GB/s for the single pass.
 */
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define TQ_GQA_HDIM 128

template <typename HalfT>
__device__ __forceinline__ float2 tq_gqa_half2_to_float2(unsigned int v);
template <>
__device__ __forceinline__ float2 tq_gqa_half2_to_float2<__half>(unsigned int v) {
    return __half22float2(*reinterpret_cast<__half2*>(&v));
}
template <>
__device__ __forceinline__ float2 tq_gqa_half2_to_float2<__nv_bfloat16>(unsigned int v) {
    return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v));
}

template <typename HalfT, int kGroup, int kWarps, int kBatch>
__global__ void __launch_bounds__(kWarps * 32) flash_tq_decode_k8v4_gqa_splitk(
    const HalfT* __restrict__ Q,               // [num_seqs, q_stride]
    const unsigned char* __restrict__ K_cache, // [blocks, block_size, kv_heads, 128] fp8
    const float* __restrict__ V_absmax,        // [blocks, block_size, kv_heads]
    const unsigned char* __restrict__ V_quant, // [blocks, block_size, kv_heads, 64]
    float* __restrict__ workspace,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const unsigned int max_blocks_per_seq,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const float inv_sqrt_d,
    const unsigned int num_splits,
    const unsigned int q_stride,
    const float* __restrict__ k_scale_ptr,
    const unsigned int sliding_window) {
    constexpr int kHd = TQ_GQA_HDIM, kVec = 4;
    const unsigned int kv_head = blockIdx.x, split = blockIdx.y, seq = blockIdx.z;
    const unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned int seq_len = (unsigned int)seq_lens[seq];
    const unsigned int start =
        (sliding_window > 0 && seq_len > sliding_window) ? seq_len - sliding_window : 0u;
    const unsigned int split_size = (seq_len - start + num_splits - 1) / num_splits;
    const unsigned int s1 = min(start + (split + 1) * split_size, seq_len);
    const unsigned int s0 = min(start + split * split_size, s1);
    const unsigned int per_warp = (s1 - s0 + kWarps - 1) / kWarps;
    const unsigned int w0 = min(s0 + warp * per_warp, s1), w1 = min(w0 + per_warp, s1);
    const float k_scale = k_scale_ptr ? k_scale_ptr[kv_head] : 1.0f;

    // Query pre-scaled by 1/sqrt(d); lane owns dims [4*lane, 4*lane + 4).
    float q[kGroup][kVec];
#pragma unroll
    for (int g = 0; g < kGroup; g++) {
        const HalfT* qp = Q + (size_t)seq * q_stride + (size_t)(kv_head * kGroup + g) * kHd + lane * kVec;
        const uint2 v = __ldg(reinterpret_cast<const uint2*>(qp));
        const float2 a = tq_gqa_half2_to_float2<HalfT>(v.x), b = tq_gqa_half2_to_float2<HalfT>(v.y);
        q[g][0] = a.x * inv_sqrt_d; q[g][1] = a.y * inv_sqrt_d;
        q[g][2] = b.x * inv_sqrt_d; q[g][3] = b.y * inv_sqrt_d;
    }
    float m[kGroup], l[kGroup], o[kGroup][kVec];
#pragma unroll
    for (int g = 0; g < kGroup; g++) {
        m[g] = -1e30f; l[g] = 0.f;
#pragma unroll
        for (int i = 0; i < kVec; i++) o[g][i] = 0.f;
    }

    const int* bt = block_tables + (size_t)seq * max_blocks_per_seq;
    const size_t k_row = (size_t)num_kv_heads * kHd, v_row = (size_t)num_kv_heads * (kHd / 2);
    for (unsigned int p0 = w0; p0 < w1; p0 += kBatch) {
        unsigned int kw[kBatch];
        unsigned short vw[kBatch];
        float va[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            // Past the warp's end, reload the last position; its weight is 0.
            const unsigned int p = min(p0 + b, w1 - 1);
            const size_t slot = (size_t)bt[p / block_size] * block_size + p % block_size;
            kw[b] = __ldg(reinterpret_cast<const unsigned int*>(K_cache + slot * k_row + kv_head * kHd + lane * kVec));
            vw[b] = __ldg(reinterpret_cast<const unsigned short*>(V_quant + slot * v_row + kv_head * (kHd / 2) + lane * 2));
            va[b] = __ldg(V_absmax + slot * num_kv_heads + kv_head);
        }
        float s[kBatch][kGroup];
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            __half2_raw h0 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kw[b] & 0xFFFF), __NV_E4M3);
            __half2_raw h1 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(kw[b] >> 16), __NV_E4M3);
            const float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&h0));
            const float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&h1));
#pragma unroll
            for (int g = 0; g < kGroup; g++)
                s[b][g] = (q[g][0] * f0.x + q[g][1] * f0.y + q[g][2] * f1.x + q[g][3] * f1.y) * k_scale;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
#pragma unroll
            for (int b = 0; b < kBatch; b++)
#pragma unroll
                for (int g = 0; g < kGroup; g++) s[b][g] += __shfl_xor_sync(0xffffffff, s[b][g], off);
        float vd[kBatch][kVec];
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            const float sc = va[b] * 0.13333333f;
            const unsigned int lo = vw[b] & 0xFF, hi = vw[b] >> 8;
            vd[b][0] = ((float)(lo & 0xF) - 7.5f) * sc; vd[b][1] = ((float)(lo >> 4) - 7.5f) * sc;
            vd[b][2] = ((float)(hi & 0xF) - 7.5f) * sc; vd[b][3] = ((float)(hi >> 4) - 7.5f) * sc;
        }
#pragma unroll
        for (int g = 0; g < kGroup; g++) {
            float mn = m[g];
#pragma unroll
            for (int b = 0; b < kBatch; b++)
                if (p0 + b < w1) mn = fmaxf(mn, s[b][g]);
            const float eo = __expf(m[g] - mn);
            l[g] *= eo;
#pragma unroll
            for (int i = 0; i < kVec; i++) o[g][i] *= eo;
#pragma unroll
            for (int b = 0; b < kBatch; b++) {
                const float e = (p0 + b < w1) ? __expf(s[b][g] - mn) : 0.f;
                l[g] += e;
#pragma unroll
                for (int i = 0; i < kVec; i++) o[g][i] += e * vd[b][i];
            }
            m[g] = mn;
        }
    }

    __shared__ float sm[kWarps][kGroup], sl[kWarps][kGroup], so[kWarps][kGroup][kHd];
#pragma unroll
    for (int g = 0; g < kGroup; g++) {
        if (lane == 0) { sm[warp][g] = m[g]; sl[warp][g] = l[g]; }
#pragma unroll
        for (int i = 0; i < kVec; i++) so[warp][g][lane * kVec + i] = o[g][i];
    }
    __syncthreads();
    for (unsigned int idx = threadIdx.x; idx < kGroup * kHd; idx += kWarps * 32) {
        const unsigned int g = idx / kHd, d = idx % kHd;
        float M = -1e30f;
#pragma unroll
        for (int w = 0; w < kWarps; w++)
            if (sl[w][g] > 0.f) M = fmaxf(M, sm[w][g]);
        float L = 0.f, O = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; w++)
            if (sl[w][g] > 0.f) {
                const float e = __expf(sm[w][g] - M);
                L += sl[w][g] * e;
                O += so[w][g][d] * e;
            }
        float* ws = workspace + (((size_t)seq * num_q_heads + kv_head * kGroup + g) * num_splits + split) * (kHd + 2);
        ws[d] = O;
        if (d == 0) { ws[kHd] = M; ws[kHd + 1] = L; }
    }
}

// Launches the grouped kernel when it applies (head_dim 128, four q heads per
// kv head, no softcap) and returns true; returns false otherwise so the
// caller falls back to the per-q-head split kernel.
template <typename HalfT>
inline bool launch_flash_tq_decode_k8v4_gqa_splitk(
    const void* Q, const void* K_cache, const void* V_absmax, const void* V_quant, void* workspace,
    const int* block_tables, const int* seq_lens, unsigned int max_blocks_per_seq,
    unsigned int num_q_heads, unsigned int num_kv_heads, unsigned int head_dim, unsigned int block_size,
    float inv_sqrt_d, unsigned int num_splits, unsigned int num_seqs, unsigned int q_stride, float softcap,
    const float* k_scale_ptr, unsigned int sliding_window, cudaStream_t s) {
    if (head_dim != TQ_GQA_HDIM || num_kv_heads == 0 || num_q_heads != 4 * num_kv_heads || softcap > 0.f)
        return false;
    dim3 grid(num_kv_heads, num_splits, num_seqs);
    flash_tq_decode_k8v4_gqa_splitk<HalfT, 4, 4, 8><<<grid, 4 * 32, 0, s>>>(
        (const HalfT*)Q, (const unsigned char*)K_cache, (const float*)V_absmax,
        (const unsigned char*)V_quant, (float*)workspace, block_tables, seq_lens, max_blocks_per_seq,
        num_q_heads, num_kv_heads, block_size, inv_sqrt_d, num_splits, q_stride, k_scale_ptr, sliding_window);
    return true;
}
