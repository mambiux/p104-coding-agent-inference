// memtest: fill most of free VRAM with address-hashed patterns, read back, count bit errors; also report BW.
// Use at stock as baseline and after each memory-offset step. Pure user-space; no clock/power changes.
#include <cstdio>
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned h(unsigned long long i, unsigned s){ unsigned long long x=i*0x9E3779B97F4A7C15ull ^ s; x^=x>>29; x*=0xBF58476D1CE4E5B9ull; x^=x>>32; return (unsigned)x; }
__global__ void wr(uint4* p, size_t n, unsigned s){ for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){ unsigned long long b=i*4; p[i]=make_uint4(h(b,s),h(b+1,s),h(b+2,s),h(b+3,s)); } }
__global__ void ck(const uint4* p, size_t n, unsigned s, unsigned long long* err){ unsigned long long e=0; for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){ unsigned long long b=i*4; uint4 v=p[i]; e+=__popc(v.x^h(b,s))+__popc(v.y^h(b+1,s))+__popc(v.z^h(b+2,s))+__popc(v.w^h(b+3,s)); } if(e) atomicAdd(err,e); }
int main(int argc,char**argv){ int passes=argc>1?atoi(argv[1]):8; size_t fr,tot; cudaMemGetInfo(&fr,&tot); size_t bytes=(fr-(256<<20))&~((size_t)(1<<20)-1);
  uint4* p; if(cudaMalloc(&p,bytes)){printf("alloc failed\n");return 1;} size_t n=bytes/16; unsigned long long *e; cudaMallocManaged(&e,8); *e=0;
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms, rdms=0;
  for(int s=0;s<passes;s++){ wr<<<15*16,256>>>(p,n,0x1234567u*(s+1)); for(int r=0;r<4;r++){ cudaEventRecord(a); ck<<<15*16,256>>>(p,n,0x1234567u*(s+1),e); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); rdms+=ms; } }
  cudaDeviceSynchronize(); printf("memtest: %.2f GB x %d patterns x 4 reads, bit errors = %llu, check-read BW %.1f GB/s\n", bytes/1e9, passes, *e, (double)bytes*passes*4/(rdms*1e6)); return *e?2:0; }
