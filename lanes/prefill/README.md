# prism-pascal-prefill — ternary tile kernel for PTQ1_0 prefill on sm_61 (P104-100)

Base: /home/mbx/CORE/llama/prism-pascal-best/src. Diff: prefill.patch (mmq.cu hook, new mmq-ptq1-p61.cuh, test cases).
Build: same cmake line as prism-pascal-best (arch 61). Binaries: src/build/bin (RUNPATH to src/build/bin).
Standalone microbench (kernel vs naive reference, variants): bench/ (bench.cu + ptq1-mmq-p61.cuh with experiment macros).

Kernel (ggml/src/ggml-cuda/mmq-ptq1-p61.cuh), used for PTQ1_0 MUL_MAT on cc 6.1 when ncols >= 64, dense 2D, no ids/fusion:
- tile 16*RI rows x 128 cols, 256 threads, K step = 1 PTQ1_0 block; trits unpacked once per tile into int8 smem
  (H-trick, no masks), y block_q8_1_mmq copied verbatim; 144-B rows -> conflict-free LDS.128; thread = RI x 8 outputs.
- RI=4 default (64-row tile, 127 regs, 27 KB, 2 CTAs/SM). RI=8 (128-row, 1 CTA/SM) is slower.
- sub-block loop deliberately not unrolled (fully unrolled = instruction-cache bound, 4.3 TOPS).
- same numerics as generic MMQ (int dot per 32, d_x*d_y*float(sum) in fp32); only fp32 summation order differs.
Env: GGML_CUDA_PTQ1_P61_MMQ=0 (generic MMQ), GGML_CUDA_PTQ1_P61_RI=4|8, GGML_CUDA_PTQ1_P61_MMQ_MIN_COLS (default 64).

## Measured (P104-100, 128 W cap, same day, interleaved)
op (test-backend-ops perf), TOPS new / generic MMQ: 17408x512x5120 11.25 / 10.26; 5120x512x17408 11.05 / 10.00;
5120x512x6144 11.00 / 9.97; 6144x512x5120 11.09 / 10.09; 10240x512x5120 11.09 / 10.09; 12288x512x5120 11.22 / 10.20;
1024x512x5120 8.76 / 7.79; head 142744x512x5120 11.32 / 10.04; n=128: 17408 10.76 / 9.82, 5120x17408 10.24 / 9.35.
llama-bench pp512 (-fa 1): ub512 182.9 vs best 167.6 (+9.1 %), ub256 179.6 vs 164.3, ub128 167.4 vs 153.9;
kernel off = best (167.8/164.4/153.8). tg128 29.91 vs 29.92.
Correctness: MUL_MAT ptq1_0 215/215 (RI=4 and RI=8); KLD vs best @ub512 (4096 tok): mean 0.000000, max 5.3e-5, top-1 100 %;
greedy 96-token text identical (only timing lines differ).
