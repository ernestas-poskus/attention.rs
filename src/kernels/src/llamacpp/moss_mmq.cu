// Dense quantized matmul through llama.cpp's current MMQ kernels (int8
// tensor-core mma on Ampere and newer), vendored from ggml-org/llama.cpp
// 61dbf49 under ggml/ (MIT, see LICENSE). Only the dense, single-channel
// case candle needs is exposed: dst[m][n] = y[m][k] . x[n][k]^T, with x in
// its GGUF block format and y in F32. The activations are quantized to
// Q8_1 like every other GGUF matmul, and the output stays F32, so prompts
// that overflow F16 are safe.
//
// The rest of the ggml CUDA backend is not linked; this file provides the
// few symbols the kernels' host code touches (device table, error and log
// hooks, the context's buffer pool).
#include "ggml/src/ggml-cuda/common.cuh"
#include "ggml/src/ggml-cuda/mmq.cuh"
#include "ggml/src/ggml-cuda/quantize.cuh"

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <vector>

void ggml_log_internal(enum ggml_log_level level, const char * format, ...) {
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "llama.cpp mmq: %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1;
    (void) cudaGetDevice(&id);
    ggml_abort(file, line, "CUDA error %s on device %d in %s: %s", msg, id, func, stmt);
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info info = {};
        CUDA_CHECK(cudaGetDeviceCount(&info.device_count));
        info.device_count = std::min(info.device_count, GGML_CUDA_MAX_DEVICES);
        info.physical_device_count = info.device_count;
        for (int id = 0; id < info.device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & d = info.devices[id];
            d.cc = 100 * prop.major + 10 * prop.minor;
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.smpbo = prop.sharedMemPerBlockOptin;
            d.warp_size = prop.warpSize;
            d.total_vram = prop.totalGlobalMem;
            d.physical_device = id;
            d.physical_share_count = 1;
        }
        return info;
    }();
    return info;
}

bool ggml_cuda_pool_debug_enabled() {
    return false;
}

// Grow-only device buffer. Freeing an outgrown buffer with cudaFree waits
// for the device, so kernels still reading it finish first.
struct moss_grow_buffer {
    void * ptr = nullptr;
    size_t size = 0;
    void * get(size_t need) {
        if (need > size) {
            if (ptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
            size = need + need / 4;
            CUDA_CHECK(cudaMalloc(&ptr, size));
        }
        return ptr;
    }
};

// The stream-k fixup scratch: one allocation is live at a time per call.
struct moss_mmq_pool : ggml_cuda_pool {
    moss_grow_buffer buf;
    void * alloc(size_t size, size_t * actual_size) override {
        *actual_size = size;
        return buf.get(size);
    }
    void free(void *, size_t) override {}
};

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context &, int, int) {
    return std::unique_ptr<ggml_cuda_pool>(new moss_mmq_pool());
}

ggml_backend_cuda_context::ggml_backend_cuda_context(int device) : device(device), name(GGML_CUDA_NAME + std::to_string(device)) {}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

// Same search as ggml_cuda_mmq_get_J_best in llama.cpp's mmq.cu.
static int moss_mmq_J_best(ggml_type type, int64_t ncols, bool fallback, int cc, size_t smpbo, int * nthreads_best) {
    int J_best = 0;
    int ntiles_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, GGML_PREC_Q8);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }
        const int ntiles = (ncols + config.J - 1) / config.J;
        if (ntiles < ntiles_best) {
            J_best = J;
            *nthreads_best = config.nthreads;
            ntiles_best = ntiles;
        }
    }
    return J_best;
}

struct moss_mmq_device_state {
    std::unique_ptr<ggml_backend_cuda_context> ctx;
    moss_grow_buffer y_q8_1;
};

static std::mutex moss_mmq_mutex;
static moss_mmq_device_state moss_mmq_state[GGML_CUDA_MAX_DEVICES];

// Returns 0 on success, 1 when the type is not built here, 2 while the
// stream is capturing a CUDA graph (the scratch may need to grow), 3 when
// no tile fits in shared memory. The caller falls back on any nonzero.
extern "C" int moss_llamacpp_mmq(const void * x, int type_x, const float * y, float * dst, int64_t nrows_x, int64_t ncols_x,
                                 int64_t ncols_y, void * stream_ptr) {
    const ggml_type type = (ggml_type) type_x;
    if (type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q6_K) {
        return 1;
    }
    cudaStream_t stream = (cudaStream_t) stream_ptr;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return 2;
    }

    std::lock_guard<std::mutex> lock(moss_mmq_mutex);
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    moss_mmq_device_state & state = moss_mmq_state[id];
    if (!state.ctx) {
        state.ctx.reset(new ggml_backend_cuda_context(id));
    }

    const int64_t k = ncols_x, n = nrows_x, m = ncols_y;
    const int64_t k_padded = GGML_PAD(k, MATRIX_ROW_PADDING);
    const bool fallback = ggml_cuda_mmq_needs_fallback(n);
    int nthreads_best = 0;
    const int J_best = moss_mmq_J_best(type, m, fallback, cc, smpbo, &nthreads_best);
    if (J_best == 0) {
        return 3;
    }
    const size_t chunk = nthreads_best * sizeof(int);
    const size_t padding = ((J_best * sizeof(block_q8_1_mmq) + chunk - 1) / chunk) * chunk;
    const size_t nbytes_y = m * k_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ + padding;
    void * y_q8_1 = state.y_q8_1.get(nbytes_y);

    quantize_mmq_q8_1_cuda(y, nullptr, y_q8_1, type, k, /*s01=*/k, /*s02=*/m * k, /*s03=*/m * k, k_padded, m, 1, 1, stream);
    CUDA_CHECK(cudaGetLastError());

    const int64_t s01 = k / QK_K; // Q4_K and Q6_K blocks hold QK_K values
    const int64_t s12 = m * k_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const mmq_args args = {
        (const char *) x, type, (const int *) y_q8_1, nullptr, nullptr, dst, nullptr,
        k, n, m, s01, m, /*s1=*/n,
        1, 1, n * s01, s12, n * m,
        1, 1, n * s01, s12, n * m,
        (m + J_best - 1) / J_best, J_best, {}};
    if (type == GGML_TYPE_Q4_K) {
        mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(*state.ctx, args, stream);
    } else {
        mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(*state.ctx, args, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
