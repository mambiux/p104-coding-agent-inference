// Read-only hardware probe for the P104: attributes, DRAM BW, L2 retention vs cache hints, PCIe + copy/compute overlap.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s: %s\n",#x,cudaGetErrorString(e));}}while(0)
__global__ void rd(const int4* __restrict__ p, size_t n, int* out){ int4 a{0,0,0,0}; for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){int4 v=p[i]; a.x^=v.x;a.y^=v.y;a.z^=v.z;a.w^=v.w;} if((a.x^a.y^a.z^a.w)==0x12345) *out=1; }
__global__ void rd_cs(const int4* p, size_t n, int* out){ int4 a{0,0,0,0}; for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){int4 v=__ldcs(p+i); a.x^=v.x;a.y^=v.y;a.z^=v.z;a.w^=v.w;} if((a.x^a.y^a.z^a.w)==0x12345) *out=1; }
__global__ void spin(long long cyc){ long long t=clock64(); while(clock64()-t<cyc); }
int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("name %s cc %d.%d SMs %d L2 %d B memclk %d kHz bus %d bit smem/SM %zu regs/SM %d maxthr/SM %d\n",p.name,p.major,p.minor,p.multiProcessorCount,p.l2CacheSize,p.memoryClockRate,p.memoryBusWidth,p.sharedMemPerMultiprocessor,p.regsPerMultiprocessor,p.maxThreadsPerMultiProcessor);
  printf("asyncEngineCount %d concurrentKernels %d persistingL2CacheMaxSize %d accessPolicyMaxWindowSize %d\n",p.asyncEngineCount,p.concurrentKernels,p.persistingL2CacheMaxSize,p.accessPolicyMaxWindowSize);
  int v; 
  #define A(x) CK(cudaDeviceGetAttribute(&v,x,0)); printf(#x " = %d\n",v);
  A(cudaDevAttrComputePreemptionSupported) A(cudaDevAttrConcurrentManagedAccess) A(cudaDevAttrPageableMemoryAccess) A(cudaDevAttrManagedMemory)
  A(cudaDevAttrGlobalL1CacheSupported) A(cudaDevAttrLocalL1CacheSupported) A(cudaDevAttrStreamPrioritiesSupported) A(cudaDevAttrCooperativeLaunch)
  A(cudaDevAttrSingleToDoublePrecisionPerfRatio) A(cudaDevAttrMaxSharedMemoryPerBlockOptin) A(cudaDevAttrCanUseHostPointerForRegisteredMem) A(cudaDevAttrHostNativeAtomicSupported)
  A(cudaDevAttrGPUDirectRDMASupported) A(cudaDevAttrMemoryPoolsSupported) A(cudaDevAttrClockRate)
  int ndev; cudaGetDeviceCount(&ndev); printf("visible devices %d (P2P between P104s not testable with one card)\n",ndev);
  cudaStreamAttrValue sv{}; sv.accessPolicyWindow.base_ptr=0; sv.accessPolicyWindow.num_bytes=0; 
  cudaError_t e=cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 1<<20); printf("cudaLimitPersistingL2CacheSize set -> %s\n",cudaGetErrorString(e)); cudaGetLastError();
  int *o; CK(cudaMalloc(&o,4));
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  // DRAM BW
  size_t big=(size_t)1<<30; int4* B; CK(cudaMalloc(&B,big)); cudaMemset(B,1,big);
  size_t nb=big/16; rd<<<15*16,256>>>(B,nb,o); cudaDeviceSynchronize();
  for(int g: {15*8,15*16,15*32}){ cudaEventRecord(a); for(int r=0;r<5;r++) rd<<<g,256>>>(B,nb,o); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); printf("DRAM read grid %d: %.1f GB/s\n",g,5.0*big/ms/1e6); }
  cudaEventRecord(a); for(int r=0;r<5;r++) rd_cs<<<15*16,256>>>(B,nb,o); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); printf("DRAM read ld.cs: %.1f GB/s\n",5.0*big/ms/1e6);
  // L2 hit bandwidth & retention
  for(size_t hs: {(size_t)512<<10,(size_t)1<<20,(size_t)1536<<10,(size_t)2<<20,(size_t)4<<20}){
    int4* H=B; size_t nh=hs/16;
    rd<<<15*16,256>>>(H,nh,o); cudaEventRecord(a); for(int r=0;r<200;r++) rd<<<15*16,256>>>(H,nh,o); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
    printf("hot set %5zu KB repeated: %.1f GB/s (L2 if >> DRAM)\n",hs>>10,200.0*hs/ms/1e6);
  }
  // retention: hot 1 MB, then stream 64 MB normal vs ld.cs, then re-read hot
  int4* H=B; size_t nh=(1<<20)/16; int4* S=B+(64<<20)/16; size_t ns=(64<<20)/16;
  for(int mode=0;mode<2;mode++){ float tot=0; for(int r=0;r<50;r++){ rd<<<15*16,256>>>(H,nh,o); if(mode==0) rd<<<15*16,256>>>(S,ns,o); else rd_cs<<<15*16,256>>>(S,ns,o); cudaEventRecord(a); rd<<<15*16,256>>>(H,nh,o); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); tot+=ms; }
    printf("hot 1MB re-read after 64MB stream (%s): %.2f us -> %.1f GB/s\n", mode?"ld.cs":"default", tot/50*1000, (1<<20)/(tot/50)/1e6); }
  // PCIe
  void* hp; CK(cudaMallocHost(&hp,256<<20));
  cudaEventRecord(a); cudaMemcpy(B,hp,256<<20,cudaMemcpyHostToDevice); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); printf("H2D pinned 256MB: %.3f GB/s\n",(256<<20)/ms/1e6);
  cudaEventRecord(a); cudaMemcpy(hp,B,256<<20,cudaMemcpyDeviceToHost); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); printf("D2H pinned 256MB: %.3f GB/s\n",(256<<20)/ms/1e6);
  // small-transfer latency
  cudaEventRecord(a); for(int i=0;i<1000;i++) cudaMemcpy(hp,B,64,cudaMemcpyDeviceToHost); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); printf("D2H 64B sync roundtrip: %.2f us\n",ms);
  // overlap: kernel stream + copy stream
  cudaStream_t s1,s2; cudaStreamCreateWithFlags(&s1,cudaStreamNonBlocking); cudaStreamCreateWithFlags(&s2,cudaStreamNonBlocking);
  cudaEventRecord(a); for(int r=0;r<3;r++) rd<<<15*16,256,0,s1>>>(B,nb,o); cudaMemcpyAsync(B+nb/2,hp,128<<20,cudaMemcpyHostToDevice,s2); cudaDeviceSynchronize(); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); printf("3x1GB read + 128MB H2D concurrent: %.1f ms (serial ~ %.1f+%.1f)\n",ms, 3.0*big/245e6, (128<<20)/0.4e6);
  printf("done\n"); return 0; }
