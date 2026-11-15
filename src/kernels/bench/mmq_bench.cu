// Prefill matmul bench at Qwen3 8B shapes: llama.cpp's vendored MMQ
// (moss_llamacpp_mmq) against the dequantize + cuBLAS F32 SGEMM path candle
// takes for M >= 64. The SGEMM time excludes dequantization, so it is a
// lower bound on what prefill pays today. Errors are against a float64 CPU
// reference on sampled outputs, relative to the largest reference value.
//
// See pod_run.sh for the build line.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int moss_llamacpp_mmq(const void * x, int type_x, const float * y, float * dst, int64_t nrows_x, int64_t ncols_x,
                                 int64_t ncols_y, void * stream);

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static const int GGML_Q4_K = 12, GGML_Q6_K = 14;

static float h2f(uint16_t h) { return __half2float(*reinterpret_cast<__half *>(&h)); }
static uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t *>(&h); }

static void scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// One 256-value block into y.
static void deq_q4k(const uint8_t * b, float * y) {
    const float d = h2f(*(const uint16_t *) b), dmin = h2f(*(const uint16_t *) (b + 2));
    const uint8_t * sc = b + 4, * q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is, sc, &s, &m); const float d1 = d * s, m1 = dmin * m;
        scale_min_k4(is + 1, sc, &s, &m); const float d2 = d * s, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}

static void deq_q6k(const uint8_t * b, float * y) {
    const uint8_t * ql = b, * qh = b + 128;
    const int8_t * sc = (const int8_t *) (b + 192);
    const float d = h2f(*(const uint16_t *) (b + 208));
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

int main() {
    struct Shape { const char * name; int n, k; } shapes[] = {
        {"q/o 4096x4096", 4096, 4096}, {"k/v 1024x4096", 1024, 4096},
        {"gate/up 12288x4096", 12288, 4096}, {"down 4096x12288", 4096, 12288},
    };
    std::mt19937 rng(11);
    cublasHandle_t blas; cublasCreate(&blas);
    cudaStream_t stream; CK(cudaStreamCreate(&stream)); cublasSetStream(blas, stream);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int type : {GGML_Q4_K, GGML_Q6_K}) {
        const int bsz = type == GGML_Q4_K ? 144 : 210;
        for (auto sh : shapes) {
            const int n = sh.n, k = sh.k, nb = k / 256;
            std::vector<uint8_t> w((size_t) n * nb * bsz);
            for (auto & x : w) x = (uint8_t) rng();
            for (size_t b = 0; b < (size_t) n * nb; ++b) {
                uint8_t * p = w.data() + b * bsz;
                const float d = 0.002f + (rng() % 1000) * 2e-6f;
                if (type == GGML_Q4_K) { *(uint16_t *) p = f2h(d); *(uint16_t *) (p + 2) = f2h(d * 0.5f); }
                else *(uint16_t *) (p + 208) = f2h(d);
            }
            std::vector<float> wf((size_t) n * k);
            for (int r = 0; r < n; ++r)
                for (int b = 0; b < nb; ++b)
                    (type == GGML_Q4_K ? deq_q4k : deq_q6k)(w.data() + ((size_t) r * nb + b) * bsz, wf.data() + (size_t) r * k + b * 256);
            void * dw; float * dwf, * dy, * d1, * d2;
            CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dwf, wf.size() * 4));
            CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dwf, wf.data(), wf.size() * 4, cudaMemcpyHostToDevice));
            for (int m : {5, 8, 64, 405}) {
                std::vector<float> y((size_t) m * k);
                // Activations with a few large channels, as real prompts have.
                for (auto & v : y) v = ((int) (rng() % 2001) - 1000) / 500.f;
                for (int i = 0; i < 8; ++i) for (int r = 0; r < m; ++r) y[(size_t) r * k + (rng() % k)] *= 40.f;
                CK(cudaMalloc(&dy, y.size() * 4)); CK(cudaMalloc(&d1, (size_t) m * n * 4)); CK(cudaMalloc(&d2, (size_t) m * n * 4));
                CK(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
                // dst[m][n] = y[m][k] . w[n][k]^T, column-major view: dst^T (n x m) = w (n x k) * y^T.
                const float one = 1.f, zero = 0.f;
                auto sgemm = [&] { cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, dwf, k, dy, k, &zero, d1, n); };
                auto mmq = [&] { return moss_llamacpp_mmq(dw, type, dy, d2, n, k, m, stream); };
                sgemm();
                const int rc = mmq();
                CK(cudaStreamSynchronize(stream));
                if (rc != 0) { printf("mmq rc %d\n", rc); return 1; }
                std::vector<float> o1((size_t) m * n), o2((size_t) m * n);
                CK(cudaMemcpy(o1.data(), d1, o1.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(o2.data(), d2, o2.size() * 4, cudaMemcpyDeviceToHost));
                double e1max = 0, e2max = 0, rmax = 0;
                for (int s = 0; s < 4000; ++s) {
                    const int r = rng() % m, c = rng() % n;
                    double ref = 0;
                    for (int i = 0; i < k; ++i) ref += (double) y[(size_t) r * k + i] * wf[(size_t) c * k + i];
                    rmax = fmax(rmax, fabs(ref));
                    e1max = fmax(e1max, fabs(ref - o1[(size_t) r * n + c]));
                    e2max = fmax(e2max, fabs(ref - o2[(size_t) r * n + c]));
                }
                bool finite = true;
                for (float v : o2) finite &= std::isfinite(v);
                auto timeit = [&](auto && f) {
                    for (int i = 0; i < 3; ++i) f();
                    cudaEventRecord(e0, stream);
                    for (int i = 0; i < 20; ++i) f();
                    cudaEventRecord(e1, stream); CK(cudaEventSynchronize(e1));
                    float ms; cudaEventElapsedTime(&ms, e0, e1); return ms * 1000.f / 20;
                };
                const float ts = timeit(sgemm), tm = timeit(mmq);
                const double tops = 2.0 * m * n * k / 1e6;
                printf("%s %-20s M=%-5d sgemm %7.1f us (%5.1f TF)  mmq %7.1f us (%5.1f TOPS)  x%.2f  relerr sgemm %.1e mmq %.1e finite %d\n",
                       type == GGML_Q4_K ? "Q4_K" : "Q6_K", sh.name, m, ts, tops / ts, tm, tops / tm, ts / tm, e1max / rmax, e2max / rmax, finite);
                cudaFree(dy); cudaFree(d1); cudaFree(d2);
            }
            cudaFree(dw); cudaFree(dwf);
        }
    }
    return 0;
}
