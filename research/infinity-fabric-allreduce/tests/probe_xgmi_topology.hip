// ---------------------------------------------------------------------------
// probe_xgmi_topology.hip — answer the #1 risk before building mesh AllReduce:
//   "Does the runtime let us drive all 7 xGMI links concurrently via P2P,
//    or does it serialize hipMemcpyPeerAsync?"
//
// Methodology (same as the DPP/MFMA probes): measure hardware behavior with a
// minimal kernel BEFORE building the abstraction. Cheap to run; gates the whole
// Infinity Fabric AllReduce design.
//
// What it does:
//   1. Enumerate GPUs and the P2P access matrix (who can reach whom).
//   2. Report per-link P2P bandwidth (hipDeviceGetP2PAttribute perf rank).
//   3. Measure aggregate bandwidth when GPU 0 sends to 1 peer vs N-1 peers
//      SIMULTANEOUSLY (one stream per peer). If aggregate scales with peer
//      count → links are concurrent → mesh AllReduce is viable. If it stays
//      flat → runtime serializes → mesh advantage evaporates (design must pivot).
//
// Run on the smallest multi-GPU node available (2× is enough to detect
// serialization; 8× confirms full-mesh scaling).
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 probe_xgmi_topology.hip -o probe_xgmi
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#define HC(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
    printf("HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); return 1; } } while(0)

int main() {
    int nGPU = 0;
    HC(hipGetDeviceCount(&nGPU));
    printf("=== xGMI / Infinity Fabric topology probe ===\n");
    printf("GPUs visible: %d\n\n", nGPU);
    if (nGPU < 2) {
        printf("Need >=2 GPUs to probe P2P. This node has %d.\n", nGPU);
        printf("(Single-GPU node — rerun on a multi-GPU droplet.)\n");
        return 0;
    }

    // --- 1. P2P access matrix ---
    printf("--- P2P access matrix (1 = can access) ---\n    ");
    for (int j=0;j<nGPU;++j) printf("G%d ", j);
    printf("\n");
    for (int i=0;i<nGPU;++i){
        printf("G%d: ", i);
        for (int j=0;j<nGPU;++j){
            if (i==j){ printf(" - "); continue; }
            int can=0; HC(hipDeviceCanAccessPeer(&can,i,j));
            printf(" %d ", can);
        }
        printf("\n");
    }

    // --- 2. P2P perf rank (lower = faster link) ---
    printf("\n--- P2P performance rank (hipDevP2PAttrPerformanceRank) ---\n    ");
    for (int j=0;j<nGPU;++j) printf("G%d ", j);
    printf("\n");
    for (int i=0;i<nGPU;++i){
        printf("G%d: ", i);
        for (int j=0;j<nGPU;++j){
            if (i==j){ printf(" - "); continue; }
            int rank=-1;
            hipError_t e = hipDeviceGetP2PAttribute(&rank, hipDevP2PAttrPerformanceRank, i, j);
            if (e==hipSuccess) printf(" %d ", rank); else printf(" ? ");
        }
        printf("\n");
    }

    // Enable all P2P access from GPU 0
    HC(hipSetDevice(0));
    for (int j=1;j<nGPU;++j){
        int can=0; hipDeviceCanAccessPeer(&can,0,j);
        if (can) hipDeviceEnablePeerAccess(j,0);
    }

    // --- 3. Concurrency test: GPU0 -> k peers simultaneously ---
    // Allocate a buffer on GPU0 and one per peer. Copy GPU0->peer on k streams
    // at once; measure aggregate bandwidth. Scaling with k => concurrent links.
    const size_t bytes = 256ull*1024*1024;   // 256 MB per transfer
    printf("\n--- Concurrency test: GPU0 broadcasts %zu MB to k peers at once ---\n", bytes/(1024*1024));
    printf("If aggregate BW scales with k, links are concurrent (mesh viable).\n");
    printf("If it stays flat, the runtime serializes (mesh advantage gone).\n\n");

    HC(hipSetDevice(0));
    void* src; HC(hipMalloc(&src, bytes));
    HC(hipMemset(src, 1, bytes));

    std::vector<void*> dst(nGPU, nullptr);
    std::vector<hipStream_t> st(nGPU);
    for (int j=1;j<nGPU;++j){
        HC(hipSetDevice(j)); HC(hipMalloc(&dst[j], bytes));
        HC(hipSetDevice(0)); HC(hipStreamCreate(&st[j]));
    }

    printf("%-8s  %12s  %14s  %12s\n","#peers","time(ms)","aggregate GB/s","per-link GB/s");
    for (int k=1;k<nGPU;++k){
        HC(hipSetDevice(0));
        // warmup
        for (int j=1;j<=k;++j) hipMemcpyPeerAsync(dst[j],j,src,0,bytes,st[j]);
        HC(hipDeviceSynchronize());

        hipEvent_t t0,t1; HC(hipEventCreate(&t0)); HC(hipEventCreate(&t1));
        const int reps=20;
        HC(hipEventRecord(t0,0));
        for (int r=0;r<reps;++r)
            for (int j=1;j<=k;++j)
                hipMemcpyPeerAsync(dst[j],j,src,0,bytes,st[j]);
        HC(hipEventRecord(t1,0)); HC(hipEventSynchronize(t1));
        float ms=0; HC(hipEventElapsedTime(&ms,t0,t1)); ms/=reps;
        double agg = (double)bytes*k / (ms/1e3) / 1e9;
        printf("%-8d  %12.3f  %14.1f  %12.1f\n", k, ms, agg, agg/k);
        hipEventDestroy(t0); hipEventDestroy(t1);
    }

    printf("\nInterpretation:\n");
    printf("  per-link GB/s roughly CONSTANT as #peers grows  -> links concurrent, MESH VIABLE\n");
    printf("  aggregate GB/s roughly CONSTANT (per-link drops) -> runtime serializes, PIVOT design\n");

    hipFree(src);
    for (int j=1;j<nGPU;++j){ hipSetDevice(j); hipFree(dst[j]); }
    return 0;
}
