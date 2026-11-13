# Run on a CUDA pod from the repository root: the TurboQuant decode bench.
set -u
nvidia-smi --query-gpu=name,power.limit --format=csv,noheader
cd src/kernels/bench
nvcc -O3 -arch=sm_86 -std=c++17 --expt-relaxed-constexpr -I../src -I../src/flash tq_bench.cu ../src/flash/flash_instantiate.cu ../src/flash/flash_decode.cu -o /root/tqb 2>&1 | grep -E "error|warning.*tq_bench" | head -30
/root/tqb 2>&1; echo "=== ATTN_TQ_GQA=0"; ATTN_TQ_GQA=0 /root/tqb 2>&1 | grep splitk
