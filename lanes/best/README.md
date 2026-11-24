# prism-pascal-best — PrismML llama.cpp fork (branch prism, base d12ca13bb) tuned for P104-100 / sm_61

Built from /home/<user>/CORE/llama/prism-pascal-arch (kernel architect tree) with the I=128 PTQ1_0 MMQ tile rows taken from optB.
`best.patch` = full diff against d12ca13bb (ggml/ only). `src/` = patched source, `bin/` = binaries + shared libs.

## Build (exact)
    export PATH=/usr/local/cuda-12.6/bin:$PATH
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61" -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
    cmake --build build -j 4 --target llama-bench llama-server test-backend-ops llama-perplexity llama-cli
(-j 4: gcc 13 segfaulted/ICE'd at -j 8/14 on this host.)

## Run
The binaries carry RUNPATH=/home/<user>/CORE/llama/prism-pascal-arch/build/bin; to use the copies here set
`LD_LIBRARY_PATH=/home/<user>/CORE/llama/prism-pascal-best/bin`.

## What changed (all gated to cc 6.1; other archs untouched)
1. `ggml/src/ggml-cuda/mmvq-ptq1_0-p61.cuh` (new) + hook in `mmvq.cu`: one-column PTQ1_0 mat-vec for sm_61.
   8-lane octet per 2 rows, coalesced 16-B weight loads staged in warp shared memory (pipelined one K-step ahead),
   "H-trick" base-3 decode (digit_t = H_{t+1} - 3 H_t, H_s = high byte of b*3^s: no masks), x*3 as IADD3, gate+up+GLU
   fusion and bias supported. Requires K % 1024 == 0 and 16-B aligned rows, else falls back to the generic kernel.
2. `mmq.cu` + `mmq-config-pascal-dp4a.cuh`: PTQ1_0 prefill on sm_61 uses the existing dp4a MMQ tile kernel
   (I=128 tiles) instead of dequantize + cuBLAS. `mmq-load-tiles.cuh`: sm_61-only IADD3 multiply and carry-free trit map.
3. `ggml-cuda.cu`: CUDA graphs enabled for cc >= 6.1 (was >= 7.0); FWHT+q8_1 fusion enabled for cc >= 6.1 (was >= 7.5).

## Kill switches (env)
- `GGML_CUDA_PTQ1_P61=0`          -> reference generic one-column PTQ1_0 mat-vec
- `GGML_CUDA_DISABLE_GRAPHS=1`    -> no CUDA graphs
- `GGML_CUDA_FWHT_FUSION=0`       -> no FWHT+q8_1 fusion
- `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0` -> PTQ1_0 prefill back to dequantize + cuBLAS (reference path)
With GGML_CUDA_PTQ1_P61=0 and GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0 the generated text was byte-identical to the reference (graphs on/off and the FWHT fusion do not change the text).

## Measured (P104-100, llama-bench -ngl 99 -fa 1 -r 3, interleaved with the reference build /home/<user>/CORE/llama/prism-pascal/bin)
| model / test          | reference | best   |
|-----------------------|-----------|--------|
| PTQ1_0 pp512          | 92.6      | 183.1  |
| PTQ1_0 tg128          | 17.84     | 29.44  |
| PTQ1_0 pp512 -ub 128  | 68.3      | 164.6  |
| PTQ1_0 tg128 @ d4096  | 17.16     | 27.56  |
| PQ2_0 pp512           | 170.8     | 170.9  |
| PQ2_0 tg128           | 20.13     | 20.52  |
Correctness: test-backend-ops -p ptq1_0 (all ops) 415 OK / 0 FAIL; MUL_MAT pq2_0 262 OK / 0 FAIL.
KLD vs reference logits (1024 tokens): decode path (ub=1) mean KLD 1.38e-4, top-1 99.61 %, PPL ratio 1.0013;
MMQ prefill (ub=512) vs reference ub=1: KLD 1.41e-4, top-1 99.41 %.
