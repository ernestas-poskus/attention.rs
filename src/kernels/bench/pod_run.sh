# Run on a CUDA pod from the repository root: the llama.cpp MMQ prefill bench.
set -u
nvidia-smi --query-gpu=name,power.limit --format=csv,noheader
cd src/kernels/bench
L=../src/llamacpp
nvcc -O3 -arch=sm_86 -std=c++17 -use_fast_math -extended-lambda --expt-relaxed-constexpr \
  -I$L -I$L/ggml/include -I$L/ggml/src -I$L/ggml/src/ggml-cuda \
  mmq_bench.cu $L/moss_mmq.cu $L/ggml/src/ggml-cuda/quantize.cu \
  $L/ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu $L/ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu \
  -lcublas -o /root/mmqb 2>&1 | grep -E "error|undefined" | head -40
/root/mmqb 2>&1
