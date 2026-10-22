/* gpu_sa.cu -- GPU BWT of T = F . revcomp(F) for long texts (libcubwt is limited to n < 2^31).
 *
 * Suffix sorting in two steps, all on the GPU:
 *   A. Suffixes are bucketed by their first 11 symbols. Chunks of consecutive buckets that fit in device
 *      memory are sorted one at a time: collect the chunk's positions, radix sort them by their first 32
 *      symbols and give each suffix its row. A suffix that differs from both neighbours is final; tied
 *      suffixes (equal first 32 symbols) get the row of their group's first member as rank. While a
 *      chunk has more ties than step B can hold, its tied groups are split further by the next 32
 *      symbols of the text.
 *   B. Prefix doubling (Larsson & Sadakane) over the suffixes still tied, with the ranks of all n suffixes
 *      in a 40-bit array: within its group a tied suffix at p is ordered by the rank of p + h, h doubling
 *      from the depth all tied groups have reached, until no ties remain.
 * Each suffix writes its BWT symbol T[p-1] at its row once it is final. The result (L with $ removed and
 * primary) is the same as libcubwt_bwt and libsais64 give.
 * T is followed by $, smaller than every symbol: a suffix that ends within a 32-symbol key has the key
 * padded with A (0), and among suffixes with equal keys such suffixes come first, shortest first.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "gpu_sa.h"
#include "idxcommon.h"

#define GCK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
	fprintf(stderr, "[galn-gpu-bwt] CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

#define KSYM 32                     /* symbols per sort key (2 bits each) */
#define BSYM 11                     /* a bucket is the first 11 symbols */
#define BSHIFT (64 - 2 * BSYM)
#define NBUCKET (1ULL << (2 * BSYM))
#define L_UNSET 0xff
#define L_DOLLAR 4
#define MAX_REFINE 8                /* text rounds per chunk in step A */

typedef unsigned long long ull;

struct MaxOp {
	__device__ __forceinline__ uint64_t operator()(const uint64_t &a, const uint64_t &b) const { return a > b ? a : b; }
};

/* 40-bit rank per text position: the row of the suffix, or of the first member of its group while tied */
struct Rank { uint32_t *lo; uint8_t *hi; };
__device__ __forceinline__ uint64_t rank_get(const Rank r, uint64_t p) { return (uint64_t)r.hi[p] << 32 | r.lo[p]; }
__device__ __forceinline__ void rank_set(const Rank r, uint64_t p, uint64_t v) { r.lo[p] = (uint32_t)v; r.hi[p] = (uint8_t)(v >> 32); }

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

#define GRID_STRIDE(i, cnt) for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < (cnt); i += (uint64_t)gridDim.x * blockDim.x)
static const unsigned TPB = 256;
static unsigned nblocks(uint64_t cnt) { uint64_t b = (cnt + TPB - 1) / TPB; return (unsigned)std::max<uint64_t>(1, std::min<uint64_t>(b, 1u << 20)); }

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

/* step A, after the first sort: ranks, final symbols; H is replaced by each element's index (its slot) */
__global__ void chunk_finish_kernel(const uint64_t *__restrict__ T, const uint64_t *__restrict__ P, uint64_t *H,
                                    const uint8_t *__restrict__ F, uint64_t c, uint64_t row0, Rank R, uint8_t *__restrict__ Lf,
                                    uint8_t *__restrict__ keep)
{
	GRID_STRIDE(i, c) {
		uint64_t p = P[i];
		rank_set(R, p, row0 + H[i]);
		bool single = F[i] && (i + 1 == c || F[i + 1]);
		if (single) Lf[row0 + i + 1] = p ? sym_at(T, p - 1) : L_DOLLAR;   /* row 0 of the BWT matrix is $ */
		keep[i] = !single;
		H[i] = i;
	}
}

/* key of the next 32 symbols at depth off (all A past the end) */
__global__ void text_key_kernel(const uint64_t *__restrict__ T, const uint64_t *__restrict__ P, uint64_t t, uint64_t off, uint64_t n,
                                uint64_t *__restrict__ K)
{
	GRID_STRIDE(i, t) { uint64_t q = P[i] + off; K[i] = q >= n ? 0 : key_at(T, q); }
}

/* step A refinement round: slot S[i] is the element's row - row0; the group's first slot gives the rank */
__global__ void refine_finish_kernel(const uint64_t *__restrict__ T, const uint64_t *__restrict__ P, const uint64_t *__restrict__ H,
                                     const uint64_t *__restrict__ S, const uint8_t *__restrict__ F, uint64_t t, uint64_t row0, Rank R,
                                     uint8_t *__restrict__ Lf, uint8_t *__restrict__ keep)
{
	GRID_STRIDE(i, t) {
		uint64_t p = P[i];
		rank_set(R, p, row0 + S[H[i]]);
		bool single = F[i] && (i + 1 == t || F[i + 1]);
		if (single) Lf[row0 + S[i] + 1] = p ? sym_at(T, p - 1) : L_DOLLAR;
		keep[i] = !single;
	}
}

/* ---------------- step B: prefix doubling on the tied suffixes ---------------- */
/* rank of the suffix h further on; p + h == n is the $ suffix, smaller than all */
__global__ void next_rank_kernel(const uint64_t *__restrict__ P, uint64_t u, uint64_t h, uint64_t n, Rank R, uint64_t *__restrict__ K)
{
	GRID_STRIDE(i, u) { uint64_t q = P[i] + h; K[i] = q >= n ? 0 : rank_get(R, q) + 1; }
}

__global__ void head_kernel(const uint64_t *__restrict__ P, uint64_t u, Rank R, uint64_t *__restrict__ K)
{
	GRID_STRIDE(i, u) K[i] = rank_get(R, P[i]);
}

__global__ void round_finish_kernel(const uint64_t *__restrict__ T, const uint64_t *__restrict__ P, const uint64_t *__restrict__ H,
                                    const uint64_t *__restrict__ Row, const uint8_t *__restrict__ F, uint64_t u, Rank R,
                                    uint8_t *__restrict__ Lf, uint8_t *__restrict__ keep)
{
	GRID_STRIDE(i, u) {
		uint64_t p = P[i];
		rank_set(R, p, Row[H[i]]);
		bool single = F[i] && (i + 1 == u || F[i + 1]);
		if (single) Lf[Row[i] + 1] = p ? sym_at(T, p - 1) : L_DOLLAR;
		keep[i] = !single;
	}
}

__global__ void last_symbol_kernel(const uint64_t *__restrict__ T, uint64_t n, uint8_t *Lf) { Lf[0] = sym_at(T, n - 1); }

__global__ void count_unset_kernel(const uint8_t *__restrict__ Lf, uint64_t m, ull *cnt)
{
	ull c = 0;
	GRID_STRIDE(i, m) c += Lf[i] == L_UNSET;
	if (c) atomicAdd(cnt, c);
}

/* ---------------- driver ---------------- */
static int nbits(uint64_t x) { int b = 0; while (x) { ++b; x >>= 1; } return b; }

struct DevMem {
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
	~DevMem() { for (void *p : blocks) cudaFree(p); }
};

extern "C" int64_t galn_gpu_bwt(const uint8_t *pac, int64_t l_pac, uint8_t *L)
{
	const uint64_t n = 2 * (uint64_t)l_pac, nw = (n + 31) / 32 + 2;
	double t0 = galn_realtime(), t = t0;
	if (n < 2 || n >= (1ULL << 40)) return -1;
	DevMem M;
	size_t fr, tot;
	GCK(cudaMemGetInfo(&fr, &tot));

	/* text, ranks and the BWT (indexed by matrix row, n + 1 rows) stay for the whole run */
	uint64_t *T = (uint64_t*)M.get(nw * 8);
	Rank R;
	R.lo = (uint32_t*)M.get(n * 4);
	R.hi = (uint8_t*)M.get(n);
	uint8_t *Lf = (uint8_t*)M.get(n + 1);
	ull *cnt = (ull*)M.get(NBUCKET * 8), *ctr = (ull*)M.get(64 * 8);
	if (!T || !R.lo || !R.hi || !Lf || !cnt || !ctr) {
		fprintf(stderr, "[galn-gpu-bwt] n=%llu: %.1f GB of device memory needed for text and ranks, %.1f GB free\n",
		        (ull)n, (nw * 8 + 6.0 * n) / 1e9, fr / 1e9);
		return -1;
	}
	{
		uint64_t lb = (l_pac + 3) / 4;
		uint8_t *dpac = (uint8_t*)M.get(lb);
		if (!dpac) return -1;
		GCK(cudaMemcpy(dpac, pac, lb, cudaMemcpyHostToDevice));
		text_kernel<<<nblocks(nw), TPB>>>(dpac, (uint64_t)l_pac, n, nw, T);
		GCK(cudaGetLastError());
		GCK(cudaDeviceSynchronize());
		M.put(dpac);
	}
	GCK(cudaMemset(Lf, L_UNSET, n + 1));
	GCK(cudaMemset(cnt, 0, NBUCKET * 8));
	hist_kernel<<<nblocks(n), TPB>>>(T, n, cnt);
	GCK(cudaGetLastError());
	std::vector<ull> bucket(NBUCKET);
	GCK(cudaMemcpy(bucket.data(), cnt, NBUCKET * 8, cudaMemcpyDeviceToHost));
	M.put(cnt);
	ull bmax = 0;
	for (ull b : bucket) bmax = std::max(bmax, b);
	double t_text = galn_realtime() - t;

	/* step B holds 5 x u64 + 2 flags per tied suffix (plus CUB scratch); step A refines ties until they fit */
	GCK(cudaMemGetInfo(&fr, &tot));
	const size_t margin = (size_t)1 << 30;
	uint64_t avail = fr > margin ? fr - margin : 0;
	double tied_target = 0.85 * (avail / 46.0) / n;   /* fraction of suffixes step B can take */
	const char *env_tied = getenv("GALN_GPU_BWT_TIED");   /* testing: e.g. 0 refines every chunk MAX_REFINE times */
	if (env_tied) tied_target = atof(env_tied);

	/* chunk size: 4 x u64 (keys and positions, double-buffered) + 2 flags per suffix, plus CUB scratch */
	uint64_t C = std::min<uint64_t>(avail / 36, (uint64_t)INT_MAX - 4096);
	C = std::min<uint64_t>(C, n);
	size_t tmp_bytes = 0;
	for (;;) {
		size_t a = 0, b = 0, c = 0;
		cub::DoubleBuffer<uint64_t> dk(NULL, NULL), dv(NULL, NULL);
		GCK(cub::DeviceRadixSort::SortPairs(NULL, a, dk, dv, (int)C, 0, 64));
		GCK(cub::DeviceScan::InclusiveScan(NULL, b, (uint64_t*)NULL, (uint64_t*)NULL, MaxOp(), (int)C));
		GCK(cub::DeviceSelect::Flagged(NULL, c, (uint64_t*)NULL, (uint8_t*)NULL, (uint64_t*)NULL, (int*)NULL, (int)C));
		tmp_bytes = std::max(a, std::max(b, c));
		if (34 * C + tmp_bytes <= avail || C < bmax) break;
		C = C / 20 * 19;
	}
	if (C < bmax || 34 * C + tmp_bytes > avail) {
		fprintf(stderr, "[galn-gpu-bwt] not enough device memory to sort (largest bucket %llu suffixes)\n", bmax);
		return -1;
	}
	uint64_t *pool = (uint64_t*)M.get(C * 32);
	uint8_t *F = (uint8_t*)M.get(C), *keep = (uint8_t*)M.get(C);
	void *tmp = M.get(tmp_bytes);
	int *dnum = (int*)M.get(sizeof(int));
	if (!pool || !F || !keep || !tmp || !dnum) { fprintf(stderr, "[galn-gpu-bwt] allocation failed\n"); return -1; }
	const uint64_t TC = C / 5 * 2;           /* tied suffixes a chunk can refine: 5 arrays of TC in pool[0, 2C) */
	const int kb = nbits(n) + 1;
	ull *nidx = ctr + 1, *cntd = ctr + 2, *idx = ctr + 8;   /* ctr[0]: select; ctr[8..]: up to 32 indices */

	/* ---- step A ---- */
	t = galn_realtime();
	std::vector<uint64_t> Upos, Urow;   /* tied suffixes (positions, rows), in row order */
	uint64_t row0 = 0, nchunk = 0, tied32 = 0;
	int min_depth = INT_MAX, max_rounds = 0;
	double t_sort = 0;
	for (uint64_t b0 = 0; b0 < NBUCKET;) {
		uint64_t b1 = b0, c = 0;
		while (b1 < NBUCKET && c + bucket[b1] <= C) c += bucket[b1++];
		if (c == 0) { b0 = b1 + (b1 == b0); continue; }
		++nchunk;
		GCK(cudaMemset(ctr, 0, 8 * 8));
		select_kernel<<<nblocks(n), TPB>>>(T, n, b0, b1, ctr, pool + 2 * C, pool);
		GCK(cudaGetLastError());
		ull got;
		GCK(cudaMemcpy(&got, ctr, 8, cudaMemcpyDeviceToHost));
		if (got != c) { fprintf(stderr, "[galn-gpu-bwt] internal error: chunk has %llu suffixes, expected %llu\n", got, (ull)c); exit(4); }
		double ts = galn_realtime();
		cub::DoubleBuffer<uint64_t> dk(pool, pool + C), dv(pool + 2 * C, pool + 3 * C);
		size_t tb = tmp_bytes;
		GCK(cub::DeviceRadixSort::SortPairs(tmp, tb, dk, dv, (int)c, 0, 64));
		GCK(cudaDeviceSynchronize());
		t_sort += galn_realtime() - ts;
		uint64_t *K = dk.Current(), *Ka = dk.Alternate(), *P = dv.Current(), *Pa = dv.Alternate();
		runoff_find_kernel<<<nblocks(c), TPB>>>(P, c, n, KSYM, idx, nidx);
		runoff_fix_kernel<<<1, 1>>>(P, NULL, K, c, n, KSYM, idx, nidx);
		group_flag_kernel<<<nblocks(c), TPB>>>(NULL, K, P, c, n, KSYM, F, Ka);
		tb = tmp_bytes; GCK(cub::DeviceScan::InclusiveScan(tmp, tb, Ka, Pa, MaxOp(), (int)c));   /* group starts -> Pa */
		chunk_finish_kernel<<<nblocks(c), TPB>>>(T, P, Pa, F, c, row0, R, Lf, keep);          /* Pa -> slots */
		count_kernel<<<nblocks(c), TPB>>>(keep, c, cntd);
		GCK(cudaGetLastError());
		ull tc;
		GCK(cudaMemcpy(&tc, cntd, 8, cudaMemcpyDeviceToHost));
		tied32 += tc;
		/* tied suffixes (positions, slots) to pool[0, 2C): K and Ka are no longer needed */
		uint64_t *Pt = pool, *St = pool + (tc <= TC ? 4 * TC : C);
		tb = tmp_bytes; GCK(cub::DeviceSelect::Flagged(tmp, tb, P, keep, Pt, dnum, (int)c));
		tb = tmp_bytes; GCK(cub::DeviceSelect::Flagged(tmp, tb, Pa, keep, St, dnum, (int)c));
		uint64_t t_n = tc;
		int d = 0;
		if (tc <= TC) {   /* split tied groups by the next 32 symbols while there are more ties than step B can take */
			uint64_t *fb[3] = {pool + TC, pool + 2 * TC, pool + 3 * TC};
			while (t_n && d < MAX_REFINE && t_n > tied_target * c) {
				++d;
				const uint64_t off = (uint64_t)KSYM * d, lim = off + KSYM;
				text_key_kernel<<<nblocks(t_n), TPB>>>(T, Pt, t_n, off, n, fb[1]);
				cub::DoubleBuffer<uint64_t> rk(fb[1], fb[2]), rv(Pt, fb[0]);
				tb = tmp_bytes; GCK(cub::DeviceRadixSort::SortPairs(tmp, tb, rk, rv, (int)t_n, 0, 64));
				head_kernel<<<nblocks(t_n), TPB>>>(rv.Current(), t_n, R, rk.Current());
				tb = tmp_bytes; GCK(cub::DeviceRadixSort::SortPairs(tmp, tb, rk, rv, (int)t_n, 0, kb));   /* stable: by group */
				uint64_t *G = rk.Current(), *Kx = rk.Alternate(), *Pc = rv.Current(), *Pf = rv.Alternate();
				text_key_kernel<<<nblocks(t_n), TPB>>>(T, Pc, t_n, off, n, Kx);
				GCK(cudaMemset(nidx, 0, 8));
				runoff_find_kernel<<<nblocks(t_n), TPB>>>(Pc, t_n, n, lim, idx, nidx);
				runoff_fix_kernel<<<1, 1>>>(Pc, G, Kx, t_n, n, lim, idx, nidx);
				group_flag_kernel<<<nblocks(t_n), TPB>>>(G, Kx, Pc, t_n, n, lim, F, Pf);
				tb = tmp_bytes; GCK(cub::DeviceScan::InclusiveScan(tmp, tb, Pf, G, MaxOp(), (int)t_n));   /* group starts -> G */
				refine_finish_kernel<<<nblocks(t_n), TPB>>>(T, Pc, G, St, F, t_n, row0, R, Lf, keep);
				GCK(cudaGetLastError());
				tb = tmp_bytes; GCK(cub::DeviceSelect::Flagged(tmp, tb, Pc, keep, Pf, dnum, (int)t_n));
				tb = tmp_bytes; GCK(cub::DeviceSelect::Flagged(tmp, tb, St, keep, Kx, dnum, (int)t_n));
				int nt;
				GCK(cudaMemcpy(&nt, dnum, sizeof(int), cudaMemcpyDeviceToHost));
				/* positions now in Pf, slots in Kx; the other three buffers are free */
				uint64_t *all[5] = {Pt, fb[0], fb[1], fb[2], St};
				int r = 0;
				for (int i = 0; i < 5; ++i) if (all[i] != Pf && all[i] != Kx) fb[r++] = all[i];
				Pt = Pf; St = Kx;
				t_n = (uint64_t)nt;
			}
		}
		if (t_n) {
			size_t old = Upos.size();
			Upos.resize(old + t_n); Urow.resize(old + t_n);
			GCK(cudaMemcpy(Upos.data() + old, Pt, t_n * 8, cudaMemcpyDeviceToHost));
			GCK(cudaMemcpy(Urow.data() + old, St, t_n * 8, cudaMemcpyDeviceToHost));
			for (size_t i = old; i < old + t_n; ++i) Urow[i] += row0;
			min_depth = std::min(min_depth, KSYM * (d + 1));
		}
		max_rounds = std::max(max_rounds, d);
		row0 += c;
		b0 = b1;
	}
	if (row0 != n) { fprintf(stderr, "[galn-gpu-bwt] internal error: %llu rows, expected %llu\n", (ull)row0, (ull)n); exit(4); }
	M.put(pool); M.put(F); M.put(keep); M.put(tmp);
	double t_a = galn_realtime() - t;
	fprintf(stderr, "[galn-gpu-bwt] n=%llu: %llu chunks of up to %llu suffixes (radix sort %.2f s), %.2f s; tied after 32 symbols %.2f%%, "
	        "after up to %d text rounds %.2f%% (%llu)\n", (ull)n, (ull)nchunk, (ull)C, t_sort, t_a, 100.0 * tied32 / n,
	        max_rounds, 100.0 * Upos.size() / n, (ull)Upos.size());

	/* ---- step B ---- */
	t = galn_realtime();
	uint64_t u = Upos.size(), h = min_depth == INT_MAX ? KSYM : (uint64_t)min_depth;
	int rounds = 0;
	if (u) {
		GCK(cudaMemGetInfo(&fr, &tot));
		avail = fr > margin ? fr - margin : 0;
		size_t a = 0, b = 0, c = 0;
		{
			cub::DoubleBuffer<uint64_t> dk(NULL, NULL), dv(NULL, NULL);
			GCK(cub::DeviceRadixSort::SortPairs(NULL, a, dk, dv, (int)std::min<uint64_t>(u, INT_MAX), 0, kb));
			GCK(cub::DeviceScan::InclusiveScan(NULL, b, (uint64_t*)NULL, (uint64_t*)NULL, MaxOp(), (int)std::min<uint64_t>(u, INT_MAX)));
			GCK(cub::DeviceSelect::Flagged(NULL, c, (uint64_t*)NULL, (uint8_t*)NULL, (uint64_t*)NULL, (int*)NULL, (int)std::min<uint64_t>(u, INT_MAX)));
		}
		tmp_bytes = std::max(a, std::max(b, c));
		if (u > (uint64_t)INT_MAX - 4096 || 42 * u + tmp_bytes > avail) {
			fprintf(stderr, "[galn-gpu-bwt] %llu tied suffixes do not fit in device memory\n", (ull)u);
			return -1;
		}
		uint64_t *buf[5];
		for (int i = 0; i < 5; ++i) buf[i] = (uint64_t*)M.get(u * 8);
		F = (uint8_t*)M.get(u); keep = (uint8_t*)M.get(u); tmp = M.get(tmp_bytes);
		for (int i = 0; i < 5; ++i) if (!buf[i]) return -1;
		if (!F || !keep || !tmp) return -1;
		uint64_t *P = buf[0], *P2 = buf[1], *K = buf[2], *K2 = buf[3], *Row = buf[4];
		GCK(cudaMemcpy(P, Upos.data(), u * 8, cudaMemcpyHostToDevice));
		GCK(cudaMemcpy(Row, Urow.data(), u * 8, cudaMemcpyHostToDevice));
		std::vector<uint64_t>().swap(Upos); std::vector<uint64_t>().swap(Urow);
		while (u) {
			size_t tb;
			/* order within groups by the rank h further on: sort by that rank, then (stable) by group */
			next_rank_kernel<<<nblocks(u), TPB>>>(P, u, h, n, R, K);
			cub::DoubleBuffer<uint64_t> dk(K, K2), dv(P, P2);
			tb = tmp_bytes; GCK(cub::DeviceRadixSort::SortPairs(tmp, tb, dk, dv, (int)u, 0, kb));
			head_kernel<<<nblocks(u), TPB>>>(dv.Current(), u, R, dk.Current());
			tb = tmp_bytes; GCK(cub::DeviceRadixSort::SortPairs(tmp, tb, dk, dv, (int)u, 0, kb));
			uint64_t *G = dk.Current(), *Kx = dk.Alternate(), *Pc = dv.Current(), *Pf = dv.Alternate();
			next_rank_kernel<<<nblocks(u), TPB>>>(Pc, u, h, n, R, Kx);
			group_flag_kernel<<<nblocks(u), TPB>>>(G, Kx, Pc, u, n, 0, F, Pf);             /* H0 -> Pf */
			tb = tmp_bytes; GCK(cub::DeviceScan::InclusiveScan(tmp, tb, Pf, G, MaxOp(), (int)u));   /* group starts -> G */
			round_finish_kernel<<<nblocks(u), TPB>>>(T, Pc, G, Row, F, u, R, Lf, keep);
			GCK(cudaGetLastError());
			tb = tmp_bytes; GCK(cub::DeviceSelect::Flagged(tmp, tb, Pc, keep, Pf, dnum, (int)u));
			tb = tmp_bytes; GCK(cub::DeviceSelect::Flagged(tmp, tb, Row, keep, Kx, dnum, (int)u));
			int nu;
			GCK(cudaMemcpy(&nu, dnum, sizeof(int), cudaMemcpyDeviceToHost));
			/* next round: positions in Pf, rows in Kx; the other three buffers are free */
			uint64_t *rest[3], *all[5] = {P, P2, K, K2, Row};
			int r = 0;
			for (int i = 0; i < 5; ++i) if (all[i] != Pf && all[i] != Kx) rest[r++] = all[i];
			P = Pf; Row = Kx; P2 = rest[0]; K = rest[1]; K2 = rest[2];
			u = (uint64_t)nu;
			h <<= 1;
			++rounds;
			if (h >= (1ULL << 41)) { fprintf(stderr, "[galn-gpu-bwt] internal error: prefix doubling did not finish\n"); exit(4); }
		}
	}
	double t_b = galn_realtime() - t;

	/* ---- L ($ removed) and primary ---- */
	last_symbol_kernel<<<1, 1>>>(T, n, Lf);
	GCK(cudaMemset(ctr, 0, 8));
	count_unset_kernel<<<nblocks(n + 1), TPB>>>(Lf, n + 1, ctr);
	ull unset;
	GCK(cudaMemcpy(&unset, ctr, 8, cudaMemcpyDeviceToHost));
	uint32_t lo; uint8_t hi;
	GCK(cudaMemcpy(&lo, R.lo, 4, cudaMemcpyDeviceToHost));
	GCK(cudaMemcpy(&hi, R.hi, 1, cudaMemcpyDeviceToHost));
	uint64_t primary = ((uint64_t)hi << 32 | lo) + 1;
	uint8_t at_primary;
	GCK(cudaMemcpy(&at_primary, Lf + primary, 1, cudaMemcpyDeviceToHost));
	if (unset || at_primary != L_DOLLAR) {
		fprintf(stderr, "[galn-gpu-bwt] internal error: %llu rows without a symbol, row %llu holds %d\n", unset, (ull)primary, at_primary);
		exit(4);
	}
	GCK(cudaMemcpy(L, Lf, primary, cudaMemcpyDeviceToHost));
	GCK(cudaMemcpy(L + primary, Lf + primary + 1, n - primary, cudaMemcpyDeviceToHost));
	fprintf(stderr, "[galn-gpu-bwt] text + buckets %.2f s, step A %.2f s, step B %.2f s (%d rounds, longest tie < %llu symbols), total %.2f s\n",
	        t_text, t_a, t_b, rounds, (ull)h, galn_realtime() - t0);
	return (int64_t)primary;
}
