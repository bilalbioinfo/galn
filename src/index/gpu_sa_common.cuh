/* gpu_sa_common.cuh: GPU kernels shared by the two BWT builders (gpu_sa.cu, gpu_sa_ranks.cu). */
#ifndef GALN_GPU_SA_COMMON_CUH
#define GALN_GPU_SA_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#define GCK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
	fprintf(stderr, "[galn index] CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

#define KSYM 32                     /* symbols per sort key (2 bits each) */
#define BSYM 11                     /* a bucket is the first 11 symbols */
#define BSHIFT (64 - 2 * BSYM)
#define NBUCKET (1ULL << (2 * BSYM))
#define L_UNSET 0xff
#define L_DOLLAR 4

typedef unsigned long long ull;
#define GRID_STRIDE(i, cnt) for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < (cnt); i += (uint64_t)gridDim.x * blockDim.x)

namespace {

struct MaxOp {
	__device__ __forceinline__ uint64_t operator()(const uint64_t &a, const uint64_t &b) const { return a > b ? a : b; }
};

/* 40-bit rank per text position: the row of the suffix, or of the first member of its group while tied */
/* T: 32 symbols per word, first symbol in the top two bits, zeros after the end (two spare words) */
__device__ __forceinline__ uint64_t key_at(const uint64_t *__restrict__ T, uint64_t p)
{
	uint64_t w = p >> 5;
	unsigned s = (unsigned)(p & 31) << 1;
	uint64_t a = T[w];
	return s ? (a << s) | (T[w + 1] >> (64 - s)) : a;
}
__device__ __forceinline__ uint8_t sym_at(const uint64_t *__restrict__ T, uint64_t p)
{
	return (uint8_t)(T[p >> 5] >> (62 - ((p & 31) << 1)) & 3);
}
/* the suffix at p ends within the symbols compared so far (lim = depth + 32) */
__device__ __forceinline__ bool runs_off(uint64_t p, uint64_t n, uint64_t lim) { return n - p < lim; }

static const unsigned TPB = 256;
static unsigned nblocks(uint64_t cnt) { uint64_t b = (cnt + TPB - 1) / TPB; return (unsigned)std::max<uint64_t>(1, std::min<uint64_t>(b, 1u << 20)); }

/* a slot in an output list for each calling thread (warp-aggregated atomic) */
__device__ __forceinline__ ull append_slot(ull *ctr)
{
	namespace cg = cooperative_groups;
	cg::coalesced_group g = cg::coalesced_threads();
	ull base = 0;
	if (g.thread_rank() == 0) base = atomicAdd(ctr, (ull)g.size());
	return g.shfl(base, 0) + g.thread_rank();
}

/* ---------------- text and buckets ---------------- */
__global__ void text_kernel(const uint8_t *__restrict__ pac, uint64_t l_pac, uint64_t n, uint64_t nw, uint64_t *__restrict__ T)
{
	GRID_STRIDE(w, nw) {
		uint64_t x = 0;
		for (int j = 0; j < 32; ++j) {
			uint64_t i = w * 32 + j, c = 0;
			if (i < l_pac) c = pac[i >> 2] >> ((~i & 3) << 1) & 3;                     /* forward strand (_get_pac) */
			else if (i < n) { uint64_t f = n - 1 - i; c = 3 - (pac[f >> 2] >> ((~f & 3) << 1) & 3); }   /* reverse complement */
			x |= c << (62 - 2 * j);
		}
		T[w] = x;
	}
}

__global__ void hist_kernel(const uint64_t *__restrict__ T, uint64_t n, ull *__restrict__ cnt)
{
	GRID_STRIDE(p, n) atomicAdd(&cnt[key_at(T, p) >> BSHIFT], 1ULL);
}

/* positions and keys of the suffixes in buckets [b0, b1), in any order (warp-aggregated append) */
__global__ void select_kernel(const uint64_t *__restrict__ T, uint64_t n, uint64_t b0, uint64_t b1, ull *ctr,
                              uint64_t *__restrict__ pos, uint64_t *__restrict__ key)
{
	const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	const unsigned lane = threadIdx.x & 31;
	for (uint64_t base = blockIdx.x * (uint64_t)blockDim.x; base < n; base += stride) {   /* uniform per warp */
		uint64_t p = base + threadIdx.x, k = 0;
		bool sel = false;
		if (p < n) { k = key_at(T, p); uint64_t b = k >> BSHIFT; sel = b >= b0 && b < b1; }
		unsigned m = __ballot_sync(0xffffffffu, sel);
		if (!m) continue;
		int leader = __ffs(m) - 1;
		ull at = 0;
		if ((int)lane == leader) at = atomicAdd(ctr, (ull)__popc(m));
		at = __shfl_sync(0xffffffffu, at, leader);
		if (sel) { ull o = at + __popc(m & ((1u << lane) - 1)); pos[o] = p; key[o] = k; }
	}
}
/* ---------------- groups (steps A and B) ---------------- */
__global__ void runoff_find_kernel(const uint64_t *__restrict__ P, uint64_t c, uint64_t n, uint64_t lim, ull *idx, ull *nidx)
{
	GRID_STRIDE(i, c) if (runs_off(P[i], n, lim)) idx[atomicAdd(nidx, 1ULL)] = i;
}

/* Runs of equal (group, key) holding suffixes that end within the key (at most 32 per depth): those come
 * first, shortest (largest p) first. One thread: such runs are rare. G may be NULL (one group). */
__global__ void runoff_fix_kernel(uint64_t *P, const uint64_t *G, const uint64_t *K, uint64_t c, uint64_t n, uint64_t lim,
                                  const ull *idx, const ull *nidx)
{
	uint64_t done[KSYM];
	int ndone = 0, ni = (int)*nidx;
	for (int t = 0; t < ni; ++t) {
		uint64_t i = idx[t], a = i, b = i + 1;
		while (a > 0 && K[a - 1] == K[i] && (!G || G[a - 1] == G[i])) --a;
		while (b < c && K[b] == K[i] && (!G || G[b] == G[i])) ++b;
		bool seen = false;
		for (int d = 0; d < ndone; ++d) seen |= done[d] == a;
		if (seen) continue;
		done[ndone++] = a;
		uint64_t rp[KSYM];
		int r = 0;
		for (uint64_t j = a; j < b; ++j) if (runs_off(P[j], n, lim)) rp[r++] = P[j];
		for (int x = 1; x < r; ++x) {                       /* descending p = ascending length */
			uint64_t v = rp[x];
			int y = x - 1;
			while (y >= 0 && rp[y] < v) { rp[y + 1] = rp[y]; --y; }
			rp[y + 1] = v;
		}
		uint64_t w = b;
		for (uint64_t j = b; j-- > a;) if (!runs_off(P[j], n, lim)) P[--w] = P[j];
		for (int x = 0; x < r; ++x) P[a + x] = rp[x];
	}
}

/* F[i]: i starts a new group (group or key changes, or a suffix ending within the key is on either side);
 * H0[i] = i if so (a max-scan then gives each element its group's first index). G may be NULL. */
__global__ void group_flag_kernel(const uint64_t *__restrict__ G, const uint64_t *__restrict__ K, const uint64_t *__restrict__ P,
                                  uint64_t c, uint64_t n, uint64_t lim, uint8_t *__restrict__ F, uint64_t *__restrict__ H0)
{
	GRID_STRIDE(i, c) {
		bool f = i == 0 || K[i] != K[i - 1] || (G && G[i] != G[i - 1]) || runs_off(P[i], n, lim) || runs_off(P[i - 1], n, lim);
		F[i] = f;
		H0[i] = f ? i : 0;
	}
}

__global__ void count_kernel(const uint8_t *__restrict__ x, uint64_t c, ull *cnt)
{
	ull s = 0;
	GRID_STRIDE(i, c) s += x[i];
	for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
	if ((threadIdx.x & 31) == 0 && s) atomicAdd(cnt, s);
}
static int nbits(uint64_t x) { int b = 0; while (x) { ++b; x >>= 1; } return b; }

struct DevMem {
	int dev = -1;                   /* the device to make current when freeing (-1: the current one) */
	std::vector<void*> blocks;
	void *get(size_t bytes) {
		void *p = NULL;
		if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess) { cudaGetLastError(); return NULL; }
		blocks.push_back(p);
		return p;
	}
	void put(void *p) {
		if (!p) return;
		auto it = std::find(blocks.begin(), blocks.end(), p);
		if (it != blocks.end()) { cudaFree(p); blocks.erase(it); }
	}
	void release() {
		if (!blocks.empty() && dev >= 0) cudaSetDevice(dev);
		for (void *p : blocks) cudaFree(p);
		blocks.clear();
	}
	~DevMem() { release(); }
};

}  // namespace

#endif
