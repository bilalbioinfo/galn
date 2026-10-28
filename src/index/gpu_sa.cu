/* gpu_sa.cu: GPU BWT and SA samples of T = F . revcomp(F) for texts too long for gpu_sa_ranks.cu.
 *
 * Device memory holds the 2-bit text (n/4 bytes), a bitmap of n bits and working buffers; the BWT and the SA
 * samples go to host memory in large blocks only (no random host access, so it runs on any CUDA GPU).
 *   Pass 1. Suffixes are bucketed by their first 11 symbols. Chunks of consecutive buckets are sorted one at a
 *      time by their first 32 symbols (radix sort); tied suffixes (equal first symbols) are split further by
 *      the next 32 symbols of the text, round after round, until few are left. A final suffix writes its BWT
 *      symbol at its row and, every 32 rows, its SA sample; the tied ones are kept (position, row, group).
 *   Doubling. The tied suffixes are sorted by prefix doubling (Larsson & Sadakane): within its group a suffix
 *      at p is ordered by the rank of p + h. Tied suffixes carry their ranks; the ranks of the other suffixes
 *      needed for the next rounds' p + h are collected by sorting the chunks again (pass 2), which reports the
 *      rows of the marked positions.
 * Ranks are suffix indices (row - 1); a group's rank is that of its first member. The result is the same as
 * libcubwt_bwt / libsais64 and bwt_cal_sa give.
 * Passes 1 and 2 run on all GPUs, a chunk at a time each (every GPU has its own copy of the text); the
 * doubling runs on the current GPU.
 */
#include <cstring>
#include <atomic>
#include <memory>
#include <mutex>
#include <omp.h>
#include "gpu_sa.h"
#include "idxcommon.h"
#include "gpu_sa_common.cuh"

#define MAX_REFINE 4096             /* text rounds: up to 32 + 4096 x 32 symbols (each batch stops at its tie target) */
#define NO_SAMPLE UINT64_MAX
#define PROBE_ROUNDS 256            /* GALN_GPU_BWT_PROBE: refine up to 32 + 256 x 32 symbols and count the ties */
#define STALL_ROUNDS 32             /* pass 1 stops refining a batch whose ties fell by less than 5% in these rounds
                                     * (long exact repeats, which the doubling sorts much faster) */

namespace {

/* where a final suffix goes */
struct Out {
	uint8_t *Lc;            /* pass 1: BWT symbols of the chunk's ranks (index j - row0), else NULL */
	uint64_t *Sc;           /* pass 1: SA samples of the chunk (index k - k_lo) */
	uint64_t k_lo;
	ull *prim;              /* pass 1: BWT row of position 0 (the $) */
	const uint32_t *mark;   /* pass 2: positions whose rank is wanted, else NULL */
	uint64_t *qpos, *qrank; /* pass 2: (position, rank) of the marked suffixes */
	ull *nq;
};

/* suffix p is final with rank j (BWT row j + 1) */
__device__ __forceinline__ void finalize(const Out &o, const uint64_t *__restrict__ T, uint64_t p, uint64_t j, uint64_t row0)
{
	if (o.Lc) {
		o.Lc[j - row0] = p ? sym_at(T, p - 1) : L_DOLLAR;
		if (((j + 1) & 31) == 0) o.Sc[((j + 1) >> 5) - o.k_lo] = p;
		if (p == 0) *o.prim = j + 1;
	}
	if (o.mark && (o.mark[p >> 5] >> (p & 31) & 1)) {
		ull at = append_slot(o.nq);
		o.qpos[at] = p;
		o.qrank[at] = j;
	}
}

/* after the first sort: final suffixes out; S = slot of each element (its rank - row0) */
__global__ void chunk_finish_kernel(const uint64_t *__restrict__ T, const uint64_t *__restrict__ P, const uint8_t *__restrict__ F,
                                    uint64_t c, uint64_t row0, Out o, uint8_t *__restrict__ keep, uint64_t *__restrict__ S)
{
	GRID_STRIDE(i, c) {
		bool single = F[i] && (i + 1 == c || F[i + 1]);
		if (single) finalize(o, T, P[i], row0 + i, row0);
		keep[i] = !single;
		S[i] = i;
	}
}

__global__ void text_key_kernel(const uint64_t *__restrict__ T, const uint64_t *__restrict__ P, uint64_t t, uint64_t off, uint64_t n,
                                uint64_t *__restrict__ K)
{
	GRID_STRIDE(i, t) { uint64_t q = P[i] + off; K[i] = q >= n ? 0 : key_at(T, q); }
}

__global__ void iota_kernel(uint64_t *x, uint64_t t) { GRID_STRIDE(i, t) x[i] = i; }

__global__ void gather_kernel(const uint64_t *__restrict__ src, const uint64_t *__restrict__ idx, uint64_t t, uint64_t *__restrict__ dst)
{
	GRID_STRIDE(i, t) dst[i] = src[idx[i]];
}

/* refinement round: H = index of the group's first element in the list, S = slots; G gets the new group slots */
__global__ void refine_finish_kernel(const uint64_t *__restrict__ T, const uint64_t *__restrict__ P, const uint64_t *__restrict__ H,
                                     const uint64_t *__restrict__ S, const uint8_t *__restrict__ F, uint64_t t, uint64_t row0, Out o,
                                     uint8_t *__restrict__ keep, uint64_t *__restrict__ G)
{
	GRID_STRIDE(i, t) {
		bool single = F[i] && (i + 1 == t || F[i + 1]);
		if (single) finalize(o, T, P[i], row0 + S[i], row0);
		keep[i] = !single;
		G[i] = S[H[i]];
	}
}

/* pass 2: number of tied suffixes that are marked (refinement stops when none are left) */
__global__ void count_marked_kernel(const uint64_t *__restrict__ P, uint64_t t, const uint32_t *__restrict__ mark, ull *cnt)
{
	ull s = 0;
	GRID_STRIDE(i, t) s += mark[P[i] >> 5] >> (P[i] & 31) & 1;
	for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
	if ((threadIdx.x & 31) == 0 && s) atomicAdd(cnt, s);
}

/* pass 2: keep only the tied groups with a marked member (the others' ranks are not wanted). G = group slot
 * (< chunk size), flag: a byte per slot; three steps so that no thread reads a flag another is still clearing */
__global__ void group_clear_kernel(const uint64_t *__restrict__ G, uint64_t t, uint8_t *flag) { GRID_STRIDE(i, t) flag[G[i]] = 0; }
__global__ void group_mark_kernel(const uint64_t *__restrict__ P, const uint64_t *__restrict__ G, const uint8_t *__restrict__ keep,
                                  uint64_t t, const uint32_t *__restrict__ mark, uint8_t *flag)
{
	GRID_STRIDE(i, t) if ((!keep || keep[i]) && (mark[P[i] >> 5] >> (P[i] & 31) & 1)) flag[G[i]] = 1;
}
__global__ void group_keep_kernel(const uint64_t *__restrict__ G, uint64_t t, const uint8_t *__restrict__ flag, uint8_t *keep)
{
	GRID_STRIDE(i, t) keep[i] = (keep[i] != 0) & flag[G[i]];
}
__global__ void fill_kernel(uint8_t *x, uint64_t t, uint8_t v) { GRID_STRIDE(i, t) x[i] = v; }

/* ---------------- doubling ---------------- */
__device__ __forceinline__ int64_t find(const uint64_t *__restrict__ a, uint64_t m, uint64_t x)
{
	uint64_t lo = 0, hi = m;
	while (lo < hi) { uint64_t mid = (lo + hi) >> 1; if (a[mid] < x) lo = mid + 1; else hi = mid; }
	return lo < m && a[lo] == x ? (int64_t)lo : -1;
}

struct Lookup {
	uint64_t n, u, m;
	const uint64_t *Spos, *Suid, *head;   /* tied suffixes: positions sorted, their ids, ranks by id */
	const uint64_t *Tq, *Tr;              /* other suffixes: positions sorted (from pass 2), their ranks */
	ull *err;
};

/* rank + 1 of the suffix at q; 0 for q == n (the $ suffix, smaller than all) */
__device__ __forceinline__ uint64_t rank1(const Lookup &L, uint64_t q)
{
	if (q >= L.n) return 0;
	int64_t k = find(L.Spos, L.u, q);
	if (k >= 0) return L.head[L.Suid[k]] + 1;
	k = find(L.Tq, L.m, q);
	if (k >= 0) return L.Tr[k] + 1;
	atomicAdd(L.err, 1ULL);
	return 0;
}

__global__ void key_kernel(const uint64_t *__restrict__ perm, uint64_t a, const uint64_t *__restrict__ upos, uint64_t h, Lookup L,
                           uint64_t *__restrict__ K)
{
	GRID_STRIDE(i, a) K[i] = rank1(L, upos[perm[i]] + h);
}

/* the positions p + h, p + 2h, ..., p + 2^(nr-1) h that are not tied suffixes */
__global__ void mark_kernel(const uint64_t *__restrict__ perm, uint64_t a, const uint64_t *__restrict__ upos, uint64_t h, int nr,
                            uint64_t n, const uint64_t *__restrict__ Spos, uint64_t u, uint32_t *mark)
{
	GRID_STRIDE(i, a) {
		uint64_t p = upos[perm[i]];
		for (int k = 0; k < nr; ++k) {
			uint64_t q = p + (h << k);
			if (q >= n) break;
			if (find(Spos, u, q) < 0) atomicOr(&mark[q >> 5], 1u << (q & 31));
		}
	}
}

__global__ void popc_kernel(const uint32_t *__restrict__ x, uint64_t nw, ull *cnt)
{
	ull s = 0;
	GRID_STRIDE(i, nw) s += __popc(x[i]);
	for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
	if ((threadIdx.x & 31) == 0 && s) atomicAdd(cnt, s);
}

/* marked positions per bucket (the marked suffixes of each chunk) */
__global__ void mark_bucket_kernel(const uint64_t *__restrict__ T, const uint32_t *__restrict__ mark, uint64_t nmw, uint64_t n,
                                   ull *__restrict__ cnt)
{
	GRID_STRIDE(w, nmw) {
		for (uint32_t x = mark[w]; x; x &= x - 1) {
			uint64_t q = w * 32 + (__ffs(x) - 1);
			if (q < n) atomicAdd(&cnt[key_at(T, q) >> BSHIFT], 1ULL);
		}
	}
}

/* new ranks of this round's suffixes; final ones out as (row << 3 | symbol) and (sample index, position) */
__global__ void round_finish_kernel(const uint64_t *__restrict__ T, const uint64_t *__restrict__ perm, const uint64_t *__restrict__ H,
                                    const uint64_t *__restrict__ Row, const uint8_t *__restrict__ F, uint64_t a,
                                    const uint64_t *__restrict__ upos, uint64_t *__restrict__ head, uint8_t *__restrict__ keep,
                                    uint64_t *Lp, ull *nL, uint64_t *Sp, ull *nS, ull *prim)
{
	GRID_STRIDE(i, a) {
		uint64_t u = perm[i], p = upos[u];
		head[u] = Row[H[i]];
		bool single = F[i] && (i + 1 == a || F[i + 1]);
		keep[i] = !single;
		if (single) {
			uint64_t r = Row[i] + 1;
			Lp[append_slot(nL)] = r << 3 | (p ? sym_at(T, p - 1) : L_DOLLAR);
			if ((r & 31) == 0) { ull at = append_slot(nS); Sp[2 * at] = r >> 5; Sp[2 * at + 1] = p; }
			if (p == 0) *prim = r;
		}
	}
}

__global__ void tied_out_kernel(const uint64_t *__restrict__ P, const uint64_t *__restrict__ S, const uint64_t *__restrict__ G,
                                uint64_t t, uint64_t row0, uint64_t *__restrict__ out)
{
	GRID_STRIDE(i, t) { out[3 * i] = P[i]; out[3 * i + 1] = row0 + S[i]; out[3 * i + 2] = row0 + G[i]; }
}

/* ---------------- driver ---------------- */
struct Chunk { uint64_t b0, b1, c, row0; };

struct Ctx {
	uint64_t n, C, TC, ns;
	const uint64_t *T;
	uint64_t *pool;           /* 4C words: keys A/B, positions A/B */
	uint8_t *F, *keep, *Lc;
	uint64_t *Sc;
	void *tmp; size_t tmp_bytes;
	int *dnum;
	ull *ctr;                 /* [0] select, [1] run-off count, [2] count, [3] prim, [4] nq, [8..] run-off indices */
	int kb_c;                 /* bits of a slot */
	double tied_target;
	double t_sort;
	int probe;                /* count the tied suffixes per depth only (pass 1, nothing kept) */
	std::vector<double> tied_at;   /* probe: tied suffixes after 32 (d + 1) symbols */
	uint64_t not_refined;     /* probe: suffixes in tied groups larger than a refinement batch */
};

static uint64_t dval(const ull *p) { ull v; GCK(cudaMemcpy(&v, p, 8, cudaMemcpyDeviceToHost)); return v; }
static uint64_t dget(const uint64_t *p) { uint64_t v; GCK(cudaMemcpy(&v, p, 8, cudaMemcpyDeviceToHost)); return v; }
/* in a nondecreasing device array: first index in [lo, hi) with a[i] >= v, and first with a[i] > v */
static uint64_t lower_index(const uint64_t *a, uint64_t lo, uint64_t hi, uint64_t v)
{
	while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (dget(a + mid) < v) lo = mid + 1; else hi = mid; }
	return lo;
}
static uint64_t upper_index(const uint64_t *a, uint64_t lo, uint64_t hi, uint64_t v)
{
	while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (dget(a + mid) <= v) lo = mid + 1; else hi = mid; }
	return lo;
}

static int flagged(Ctx &X, const uint64_t *in, const uint8_t *flags, uint64_t *out, uint64_t cnt)
{
	size_t tb = X.tmp_bytes;
	GCK(cub::DeviceSelect::Flagged(X.tmp, tb, in, flags, out, X.dnum, (int)cnt));
	int k;
	GCK(cudaMemcpy(&k, X.dnum, sizeof(int), cudaMemcpyDeviceToHost));
	return k;
}

/* Sort one chunk. Pass 1 (mark == NULL) writes the BWT symbols and SA samples of its final suffixes to Lf and
 * samples and puts the tied ones (position, rank, group rank) in U; returns the refinement rounds done.
 * Pass 2 writes (position, rank) of the chunk's `want` marked suffixes to qpos/qrank (in any order), refining
 * until none of them is tied. */
static int sort_chunk(Ctx &X, const Chunk &ch, const uint32_t *mark, uint64_t *qpos, uint64_t *qrank, uint64_t want,
                      uint8_t *Lf, uint64_t *samples, std::vector<uint64_t> *U)
{
	const uint64_t n = X.n, C = X.C, TC = X.TC, c = ch.c, row0 = ch.row0;
	uint64_t *pool = X.pool;
	GCK(cudaMemset(X.ctr, 0, 8));
	select_kernel<<<nblocks(n), TPB>>>(X.T, n, ch.b0, ch.b1, X.ctr, pool + 2 * C, pool);
	GCK(cudaGetLastError());
	if (dval(X.ctr) != c) { fprintf(stderr, "[galn index] internal error: chunk has %llu suffixes, expected %llu\n", (ull)dval(X.ctr), (ull)c); exit(4); }
	double ts = galn_realtime();
	cub::DoubleBuffer<uint64_t> dk(pool, pool + C), dv(pool + 2 * C, pool + 3 * C);
	size_t tb = X.tmp_bytes;
	GCK(cub::DeviceRadixSort::SortPairs(X.tmp, tb, dk, dv, (int)c, 0, 64));
	if (dk.Current() != pool) GCK(cudaMemcpy(pool, dk.Current(), c * 8, cudaMemcpyDeviceToDevice));
	if (dv.Current() != pool + 2 * C) GCK(cudaMemcpy(pool + 2 * C, dv.Current(), c * 8, cudaMemcpyDeviceToDevice));
	GCK(cudaDeviceSynchronize());
	X.t_sort += galn_realtime() - ts;
	/* fixed layout: keys K [0,C), Ka [C,2C), positions P [2C,3C), Pa [3C,4C) */
	uint64_t *K = pool, *Ka = pool + C, *P = pool + 2 * C, *Pa = pool + 3 * C;
	ull *nidx = X.ctr + 1, *cnt = X.ctr + 2, *idx = X.ctr + 8;
	GCK(cudaMemset(nidx, 0, 16));
	runoff_find_kernel<<<nblocks(c), TPB>>>(P, c, n, KSYM, idx, nidx);
	runoff_fix_kernel<<<1, 1>>>(P, NULL, K, c, n, KSYM, idx, nidx);
	group_flag_kernel<<<nblocks(c), TPB>>>(NULL, K, P, c, n, KSYM, X.F, Ka);
	tb = X.tmp_bytes; GCK(cub::DeviceScan::InclusiveScan(X.tmp, tb, Ka, Pa, MaxOp(), (int)c));   /* group starts (slots) -> Pa */
	Out o = {};
	uint64_t k_lo = 0, k_hi = 0;
	if (!mark) {
		k_lo = (row0 + 32) / 32; k_hi = std::min<uint64_t>((row0 + c) / 32, X.ns);
		GCK(cudaMemset(X.Lc, L_UNSET, c));
		if (k_hi >= k_lo) GCK(cudaMemset(X.Sc, 0xff, (k_hi - k_lo + 1) * 8));
		o.Lc = X.Lc; o.Sc = X.Sc; o.k_lo = k_lo; o.prim = X.ctr + 3;
	} else {
		o.mark = mark; o.qpos = qpos; o.qrank = qrank; o.nq = X.ctr + 4;
		GCK(cudaMemset(o.nq, 0, 8));
	}
	chunk_finish_kernel<<<nblocks(c), TPB>>>(X.T, P, X.F, c, row0, o, X.keep, K);   /* slots -> K */
	count_kernel<<<nblocks(c), TPB>>>(X.keep, c, cnt);
	GCK(cudaGetLastError());
	uint64_t t = dval(cnt);
	int d = 0;
	if (X.probe) {
		if (X.tied_at.size() < 1) X.tied_at.resize(1, 0);
		X.tied_at[0] += t;
	}
	uint64_t *Pt = NULL, *St = NULL, *Gt = NULL;
	int dmin = INT_MAX;   /* fewest text rounds among the batches that leave suffixes tied */
	if (t > 0) {
		/* tied list: positions P -> [C,2C), slots K -> [2C,3C), group slots Pa -> [0,C) */
		Pt = pool + C; St = pool + 2 * C; Gt = pool;
		flagged(X, P, X.keep, Pt, c); flagged(X, K, X.keep, St, c); flagged(X, Pa, X.keep, Gt, c);
		if (mark) {   /* pass 2: only groups with a marked member, compacted through [3C, 4C) */
			group_clear_kernel<<<nblocks(t), TPB>>>(Gt, t, X.Lc);
			group_mark_kernel<<<nblocks(t), TPB>>>(Pt, Gt, NULL, t, mark, X.Lc);
			fill_kernel<<<nblocks(t), TPB>>>(X.keep, t, 1);
			group_keep_kernel<<<nblocks(t), TPB>>>(Gt, t, X.Lc, X.keep);
			uint64_t *tmp3 = pool + 3 * C, nt = 0;
			uint64_t *arr[3] = {Pt, St, Gt};
			for (uint64_t *x : arr) {
				nt = (uint64_t)flagged(X, x, X.keep, tmp3, t);
				if (nt) GCK(cudaMemcpy(x, tmp3, nt * 8, cudaMemcpyDeviceToDevice));
			}
			t = nt;
		}
		/* refined by the next 32 symbols in batches of whole groups (groups are independent); a batch is
		 * worked on in [3C,4C) as 5 arrays of TB; a group larger than a batch stays tied */
		const uint64_t TB = TC;
		uint64_t *wb[5] = {pool + 3 * C, pool + 3 * C + TB, pool + 3 * C + 2 * TB, pool + 3 * C + 3 * TB, pool + 3 * C + 4 * TB};
		const int maxr = X.probe ? PROBE_ROUNDS : MAX_REFINE;
		uint64_t w = 0;   /* tied suffixes kept so far, moved to the front of the list */
		for (uint64_t s0 = 0; s0 < t;) {
			uint64_t e = std::min<uint64_t>(t, s0 + TB);
			if (e < t) e = lower_index(Gt, s0, e, dget(Gt + e));   /* back to the start of the group at e */
			uint64_t b, db = 0;
			if (e == s0) {    /* one group larger than a batch */
				e = upper_index(Gt, s0, t, dget(Gt + s0));
				b = e - s0;
				if (X.probe) X.not_refined += b;
			} else {
				uint64_t *bP = Pt + s0, *bS = St + s0, *bG = Gt + s0;
				const uint64_t b0 = e - s0;
				uint64_t b_ck = b0;   /* ties STALL_ROUNDS rounds ago */
				b = b0;
				for (;;) {
					if ((int)db >= maxr) break;
					if (!mark && !X.probe) {
						if (b <= X.tied_target * b0) break;
						if (db > 0 && db % STALL_ROUNDS == 0) {
							if (b > 0.95 * b_ck) break;
							b_ck = b;
						}
					}
					if (mark) {
						GCK(cudaMemset(cnt, 0, 8));
						count_marked_kernel<<<nblocks(b), TPB>>>(bP, b, mark, cnt);
						if (dval(cnt) == 0) break;
					}
					++db;
					const uint64_t off = (uint64_t)KSYM * db, lim = off + KSYM;
					text_key_kernel<<<nblocks(b), TPB>>>(X.T, bP, b, off, n, wb[0]);
					iota_kernel<<<nblocks(b), TPB>>>(wb[2], b);
					cub::DoubleBuffer<uint64_t> rk(wb[0], wb[1]), ri(wb[2], wb[3]);
					tb = X.tmp_bytes; GCK(cub::DeviceRadixSort::SortPairs(X.tmp, tb, rk, ri, (int)b, 0, 64));
					gather_kernel<<<nblocks(b), TPB>>>(bG, ri.Current(), b, rk.Current());
					tb = X.tmp_bytes; GCK(cub::DeviceRadixSort::SortPairs(X.tmp, tb, rk, ri, (int)b, 0, X.kb_c));   /* stable: by group */
					uint64_t *G2 = rk.Current(), *Kx = rk.Alternate(), *I = ri.Current(), *Hb = ri.Alternate(), *P2 = wb[4];
					gather_kernel<<<nblocks(b), TPB>>>(bP, I, b, P2);
					text_key_kernel<<<nblocks(b), TPB>>>(X.T, P2, b, off, n, Kx);
					GCK(cudaMemset(nidx, 0, 8));
					runoff_find_kernel<<<nblocks(b), TPB>>>(P2, b, n, lim, idx, nidx);
					runoff_fix_kernel<<<1, 1>>>(P2, G2, Kx, b, n, lim, idx, nidx);
					group_flag_kernel<<<nblocks(b), TPB>>>(G2, Kx, P2, b, n, lim, X.F, Hb);
					tb = X.tmp_bytes; GCK(cub::DeviceScan::InclusiveScan(X.tmp, tb, Hb, G2, MaxOp(), (int)b));   /* group starts -> G2 */
					refine_finish_kernel<<<nblocks(b), TPB>>>(X.T, P2, G2, bS, X.F, b, row0, o, X.keep, I);   /* new group slots -> I */
					if (mark) {
						group_clear_kernel<<<nblocks(b), TPB>>>(I, b, X.Lc);
						group_mark_kernel<<<nblocks(b), TPB>>>(P2, I, X.keep, b, mark, X.Lc);
						group_keep_kernel<<<nblocks(b), TPB>>>(I, b, X.Lc, X.keep);
					}
					GCK(cudaGetLastError());
					uint64_t nb = (uint64_t)flagged(X, P2, X.keep, bP, b);
					flagged(X, bS, X.keep, Kx, b);
					if (nb) GCK(cudaMemcpy(bS, Kx, nb * 8, cudaMemcpyDeviceToDevice));
					flagged(X, I, X.keep, bG, b);
					b = nb;
					if (X.probe) {
						if (X.tied_at.size() < (size_t)db + 1) X.tied_at.resize(db + 1, 0);
						X.tied_at[db] += b;
					}
					if (b == 0) break;
				}
			}
			if (b) {   /* keep the batch's tied suffixes: [s0, s0 + b) -> [w, w + b), through wb[0] */
				if (w != s0)
					for (uint64_t k = 0; k < b; k += TB) {
						uint64_t m = std::min<uint64_t>(TB, b - k);
						uint64_t *arr[3] = {Pt, St, Gt};
						for (uint64_t *x : arr) {
							GCK(cudaMemcpy(wb[0], x + s0 + k, m * 8, cudaMemcpyDeviceToDevice));
							GCK(cudaMemcpy(x + w + k, wb[0], m * 8, cudaMemcpyDeviceToDevice));
						}
					}
				w += b;
				dmin = std::min(dmin, (int)db);
			}
			s0 = e;
		}
		t = w;
	}
	d = dmin == INT_MAX ? 0 : dmin;
	if (X.probe) return d;
	if (!mark) {
		GCK(cudaMemcpy(Lf + row0 + 1, X.Lc, c, cudaMemcpyDeviceToHost));
		if (k_hi >= k_lo) GCK(cudaMemcpy(samples + k_lo - 1, X.Sc, (k_hi - k_lo + 1) * 8, cudaMemcpyDeviceToHost));
		if (t) {   /* (position, rank, group rank) triplets, staged in [3C, 4C), which the tied lists never use */
			uint64_t *stage = pool + 3 * C;
			size_t old = U->size();
			U->resize(old + 3 * t);
			for (uint64_t b = 0; b < t; b += C / 3) {
				uint64_t m = std::min<uint64_t>(C / 3, t - b);
				tied_out_kernel<<<nblocks(m), TPB>>>(Pt + b, St + b, Gt + b, m, row0, stage);
				GCK(cudaMemcpy(U->data() + old + 3 * b, stage, m * 24, cudaMemcpyDeviceToHost));
			}
		}
	} else {
		if (t) {
			GCK(cudaMemset(cnt, 0, 8));
			count_marked_kernel<<<nblocks(t), TPB>>>(Pt, t, mark, cnt);
			if (dval(cnt)) { fprintf(stderr, "[galn index] internal error: %llu marked suffixes still tied\n", (ull)dval(cnt)); exit(4); }
		}
		if (dval(o.nq) != want) { fprintf(stderr, "[galn index] internal error: pass 2 found %llu of %llu ranks in a chunk\n", (ull)dval(o.nq), (ull)want); exit(4); }
	}
	return d;
}


/* ---------------- GPUs ---------------- */
/* a GPU of passes 1 and 2: its own copy of the text, chunk buffers and (pass 2) the marks */
struct Dev {
	int id = -1;
	DevMem M, MC, MQ;                            /* MC: chunk buffers (device 0 frees them for the doubling) */
	Ctx X = {};
	uint32_t *mark = NULL;                       /* device 0: the doubling's bitmap */
	uint64_t *qs = NULL, *rs = NULL, qcap = 0;   /* other GPUs, pass 2: a chunk's (position, rank) before they go to device 0 */
};

struct CurDev { int d; ~CurDev() { cudaSetDevice(d); } };

/* a may read and write b's memory (cudaMemcpyPeer then goes directly, not through the host) */
static void enable_peer(int a, int b)
{
	int can = 0;
	if (cudaDeviceCanAccessPeer(&can, a, b) == cudaSuccess && can) {
		GCK(cudaSetDevice(a));
		if (cudaDeviceEnablePeerAccess(b, 0) != cudaSuccess) cudaGetLastError();   /* e.g. already enabled */
	}
	cudaGetLastError();
}

/* consecutive buckets, up to C suffixes per chunk */
static std::vector<Chunk> plan_chunks(const std::vector<ull> &bucket, uint64_t C)
{
	std::vector<Chunk> chunks;
	uint64_t row0 = 0;
	for (uint64_t b0 = 0; b0 < NBUCKET;) {
		uint64_t b1 = b0, c = 0;
		while (b1 < NBUCKET && c + bucket[b1] <= C) c += bucket[b1++];
		if (c == 0) { b0 = b1 + (b1 == b0); continue; }
		chunks.push_back({b0, b1, c, row0});
		row0 += c;
		b0 = b1;
	}
	return chunks;
}

/* temporary storage of the CUB calls on up to C elements */
static size_t cub_bytes(uint64_t C)
{
	size_t a = 0, b = 0, c = 0;
	cub::DoubleBuffer<uint64_t> dk(NULL, NULL), dv(NULL, NULL);
	GCK(cub::DeviceRadixSort::SortPairs(NULL, a, dk, dv, (int)C, 0, 64));
	GCK(cub::DeviceScan::InclusiveScan(NULL, b, (uint64_t*)NULL, (uint64_t*)NULL, MaxOp(), (int)C));
	GCK(cub::DeviceSelect::Flagged(NULL, c, (uint64_t*)NULL, (uint8_t*)NULL, (uint64_t*)NULL, (int*)NULL, (int)C));
	return std::max(a, std::max(b, c));
}

/* chunk buffers for up to C suffixes (about 35.5 C bytes + CUB storage) on the current device */
static bool chunk_buffers(Dev &G, uint64_t C, size_t tmp_bytes)
{
	Ctx &X = G.X;
	X.C = C; X.TC = C / 5; X.kb_c = nbits(C) + 1;   /* TC: refinement batch, 5 arrays in [3C, 4C) */
	X.tmp_bytes = tmp_bytes;
	X.pool = (uint64_t*)G.MC.get(C * 32);
	X.F = (uint8_t*)G.MC.get(C); X.keep = (uint8_t*)G.MC.get(C); X.Lc = (uint8_t*)G.MC.get(C);
	X.Sc = (uint64_t*)G.MC.get((C / 32 + 2) * 8);
	X.tmp = G.MC.get(tmp_bytes);
	if (!X.dnum) X.dnum = (int*)G.M.get(sizeof(int));
	return X.pool && X.F && X.keep && X.Lc && X.Sc && X.tmp && X.dnum;
}

/* Pass 2 over all GPUs: the (position, rank) of the marked suffixes of chunk i go to q1/r1 + at[i] on device 0
 * (want[i] of them). Other GPUs get a copy of the marks and stage a chunk's ranks before copying them over;
 * a chunk with more ranks than a GPU can stage is left to device 0. */
static void pass2(std::vector<Dev*> &act, const std::vector<Chunk> &chunks, const std::vector<uint64_t> &want,
                  const std::vector<uint64_t> &at, const uint32_t *mark, uint64_t nmw, uint64_t *q1, uint64_t *r1)
{
	Dev &D0 = *act[0];
	const int na = (int)act.size();
	uint64_t wmax = 0;
	for (uint64_t w : want) wmax = std::max(wmax, w);
	for (int k = 1; k < na; ++k) {
		Dev &G = *act[k];
		GCK(cudaSetDevice(G.id));
		GCK(cudaMemcpyPeer(G.mark, G.id, mark, D0.id, nmw * 4));
		size_t fr, tot;
		GCK(cudaMemGetInfo(&fr, &tot));
		const size_t margin = (size_t)1 << 30;
		uint64_t cap = std::min<uint64_t>(wmax, fr > margin ? (fr - margin) / 16 : 0);
		G.qcap = 0;
		if (cap) {
			G.qs = (uint64_t*)G.MQ.get(cap * 8); G.rs = (uint64_t*)G.MQ.get(cap * 8);
			if (G.qs && G.rs) G.qcap = cap; else G.MQ.release();
		}
	}
	std::atomic<size_t> next(0);
	std::vector<size_t> left;
	std::mutex mu;
#pragma omp parallel num_threads(na)
	{
		const int k = omp_get_thread_num();
		if (k < na) {
			Dev &G = *act[k];
			GCK(cudaSetDevice(G.id));
			for (size_t i; (i = next++) < chunks.size();) {
				if (!want[i]) continue;
				if (k == 0) sort_chunk(G.X, chunks[i], mark, q1 + at[i], r1 + at[i], want[i], NULL, NULL, NULL);
				else if (want[i] <= G.qcap) {
					sort_chunk(G.X, chunks[i], G.mark, G.qs, G.rs, want[i], NULL, NULL, NULL);
					GCK(cudaMemcpyPeerAsync(q1 + at[i], D0.id, G.qs, G.id, want[i] * 8, 0));   /* ordered before the next chunk */
					GCK(cudaMemcpyPeerAsync(r1 + at[i], D0.id, G.rs, G.id, want[i] * 8, 0));
				} else {
					std::lock_guard<std::mutex> lk(mu);
					left.push_back(i);
				}
			}
			GCK(cudaDeviceSynchronize());
		}
	}
	GCK(cudaSetDevice(D0.id));
	for (size_t i : left) sort_chunk(D0.X, chunks[i], mark, q1 + at[i], r1 + at[i], want[i], NULL, NULL, NULL);
	for (int k = 1; k < na; ++k) { act[k]->MQ.release(); act[k]->qcap = 0; }
	GCK(cudaSetDevice(D0.id));
}

}  // namespace

extern "C" int64_t galn_gpu_bwt(const uint8_t *pac, int64_t l_pac, uint8_t *Lf, uint64_t *samples)
{
	const uint64_t n = 2 * (uint64_t)l_pac, nw = (n + 31) / 32 + 2, ns = (n + 32) / 32 - 1, nmw = n / 32 + 1;
	double t0 = galn_realtime(), t = t0;
	if (n < 2 || n >= (1ULL << 40)) return -1;
	int dev0 = 0, nall = 1;
	GCK(cudaGetDevice(&dev0));
	if (cudaGetDeviceCount(&nall) != cudaSuccess || nall < 1) { cudaGetLastError(); nall = 1; }
	int ngpu = nall;
	const char *env_gpus = getenv("GALN_GPU_BWT_GPUS");   /* testing: at most this many GPUs */
	if (env_gpus) ngpu = std::max(1, std::min(nall, atoi(env_gpus)));
	CurDev restore = {dev0};
	/* D[0]: the current device, which also does the doubling; the others help with passes 1 and 2 */
	std::unique_ptr<Dev[]> D(new Dev[ngpu]);
	D[0].id = dev0; D[0].M.dev = D[0].MC.dev = dev0;
	DevMem &M = D[0].M;
	size_t fr, tot;
	Ctx &X = D[0].X;
	X.n = n; X.ns = ns;
	uint64_t *T = (uint64_t*)M.get(nw * 8);
	ull *cnt = (ull*)M.get(NBUCKET * 8);
	X.ctr = (ull*)M.get(64 * 8);
	if (!T || !cnt || !X.ctr) { GALN_VLOG("[galn-gpu-bwt] n=%llu: the text (%.1f GB) does not fit in device memory\n", (ull)n, nw * 8 / 1e9); return -1; }
	X.T = T;
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
	GCK(cudaMemset(cnt, 0, NBUCKET * 8));
	hist_kernel<<<nblocks(n), TPB>>>(T, n, cnt);
	GCK(cudaGetLastError());
	std::vector<ull> bucket(NBUCKET);
	GCK(cudaMemcpy(bucket.data(), cnt, NBUCKET * 8, cudaMemcpyDeviceToHost));
	M.put(cnt);
	ull bmax = 0;
	for (ull b : bucket) bmax = std::max(bmax, b);

	/* memory of device 0 besides the text and the bitmap (n bits), used in turn by: pass 1 and 2's chunk buffers
	 * (about 36 bytes per suffix, 80%; in pass 2 the rest holds the looked-up ranks), and the doubling (about 90
	 * bytes per tied suffix plus 16 per looked-up rank; its state goes to host memory during pass 2) */
	GCK(cudaMemGetInfo(&fr, &tot));
	const size_t margin = (size_t)1 << 30;
	const double avail = (double)fr - margin - nmw * 4.0;
	if (avail <= 0) { GALN_VLOG("[galn-gpu-bwt] not enough device memory\n"); return -1; }
	X.tied_target = 0.85 * (avail / 250.0) / n;
	const char *env_tied = getenv("GALN_GPU_BWT_TIED");   /* testing: e.g. 0 refines every chunk MAX_REFINE times */
	if (env_tied) X.tied_target = atof(env_tied);
	X.probe = getenv("GALN_GPU_BWT_PROBE") != NULL;
	uint64_t C = (uint64_t)std::min<double>(0.8 * avail / 36.0, (double)INT_MAX - 4096);
	C = std::min<uint64_t>(C, n);
	size_t tmp_bytes;
	for (;;) {
		tmp_bytes = cub_bytes(C);
		if (35.5 * C + tmp_bytes <= 0.8 * avail || C < bmax) break;
		C = C / 20 * 19;
	}
	if (C < bmax || C < 1024) { GALN_VLOG("[galn-gpu-bwt] not enough device memory to sort (largest bucket %llu suffixes)\n", bmax); return -1; }

	/* the other GPUs: the text, marks and chunk buffers (no doubling there; 15% kept for pass 2's ranks) */
	std::vector<Dev*> act(1, &D[0]);
	for (int i = 0, k = 1; i < nall && k < ngpu; ++i) {
		if (i == dev0) continue;
		Dev &G = D[k];
		G.id = i; G.M.dev = G.MC.dev = G.MQ.dev = i;
		enable_peer(i, dev0); enable_peer(dev0, i);
		GCK(cudaSetDevice(i));
		uint64_t *Tg = (uint64_t*)G.M.get(nw * 8);
		G.X.T = Tg;
		G.X.ctr = (ull*)G.M.get(64 * 8);
		G.mark = (uint32_t*)G.M.get(nmw * 4);
		bool ok = Tg && G.X.ctr && G.mark;
		if (ok) {
			GCK(cudaMemGetInfo(&fr, &tot));
			double cg = (0.85 * ((double)fr - margin) - (double)tmp_bytes) / 35.5;
			ok = cg >= C / 2.0 && cg >= bmax && cg >= 1024;
			if (ok) C = std::min<uint64_t>(C, (uint64_t)cg);
		}
		if (!ok) { G.M.release(); G.id = -1; G.X = Ctx(); continue; }
		G.X.n = n; G.X.ns = ns; G.X.tied_target = X.tied_target; G.X.probe = X.probe;
		GCK(cudaMemcpyPeer(Tg, i, T, dev0, nw * 8));
		act.push_back(&G);
		++k;
	}
	for (size_t k = 0; k < act.size();) {
		Dev &G = *act[k];
		GCK(cudaSetDevice(G.id));
		if (chunk_buffers(G, C, tmp_bytes)) { ++k; continue; }
		if (k == 0) { GCK(cudaSetDevice(dev0)); GALN_VLOG("[galn-gpu-bwt] allocation failed\n"); return -1; }
		G.MC.release(); G.M.release();
		act.erase(act.begin() + k);
	}
	GCK(cudaSetDevice(dev0));
	const int na = (int)act.size();
	fprintf(stderr, "[galn index] large reference: memory-light builder on %d GPU%s\n", na, na > 1 ? "s" : "");
	double t_text = galn_realtime() - t;

	/* ---- pass 1 (all GPUs, a chunk at a time each) ---- */
	t = galn_realtime();
	const std::vector<Chunk> chunks = plan_chunks(bucket, C);
	{
		uint64_t rows = 0;
		for (const Chunk &ch : chunks) rows += ch.c;
		if (rows != n) { fprintf(stderr, "[galn index] internal error: %llu rows, expected %llu\n", (ull)rows, (ull)n); exit(4); }
	}
	std::vector<std::vector<uint64_t>> Uc(chunks.size());   /* tied suffixes of each chunk: (position, rank, group rank), in rank order */
	std::vector<int> depth(chunks.size(), 0);
	for (Dev *G : act) { GCK(cudaSetDevice(G->id)); GCK(cudaMemset(G->X.ctr + 3, 0xff, 8)); }
	{
		std::atomic<size_t> next(0);
#pragma omp parallel num_threads(na)
		{
			const int k = omp_get_thread_num();
			if (k < na) {
				Dev &G = *act[k];
				GCK(cudaSetDevice(G.id));
				for (size_t i; (i = next++) < chunks.size();)
					depth[i] = sort_chunk(G.X, chunks[i], NULL, NULL, NULL, 0, Lf, samples, &Uc[i]);
				GCK(cudaDeviceSynchronize());
			}
		}
	}
	uint64_t prim = UINT64_MAX;   /* the row of the $ (set by the GPU whose chunk has position 0, or later by the doubling) */
	for (Dev *G : act) {
		GCK(cudaSetDevice(G->id));
		prim = std::min<uint64_t>(prim, dval(G->X.ctr + 3));
		if (G != act[0]) {
			X.t_sort += G->X.t_sort; X.not_refined += G->X.not_refined;
			if (X.tied_at.size() < G->X.tied_at.size()) X.tied_at.resize(G->X.tied_at.size(), 0);
			for (size_t d = 0; d < G->X.tied_at.size(); ++d) X.tied_at[d] += G->X.tied_at[d];
		}
	}
	GCK(cudaSetDevice(dev0));
	GCK(cudaMemcpy(X.ctr + 3, &prim, 8, cudaMemcpyHostToDevice));
	int min_depth = INT_MAX, max_rounds = 0;
	std::vector<uint64_t> uat(chunks.size() + 1, 0);
	for (size_t i = 0; i < chunks.size(); ++i) {
		if (!Uc[i].empty()) min_depth = std::min(min_depth, KSYM * (depth[i] + 1));
		max_rounds = std::max(max_rounds, depth[i]);
		uat[i + 1] = uat[i] + Uc[i].size() / 3;
	}
	const uint64_t u = uat[chunks.size()];
	double t_p1 = galn_realtime() - t;
	if (X.probe) {
		fprintf(stderr, "[galn-gpu-bwt] probe: n=%llu, pass 1 %.2f s on %d GPUs, %zu chunks of up to %llu suffixes; %llu suffixes in groups too large to refine\n",
		        (ull)n, t_p1, na, chunks.size(), (ull)C, (ull)X.not_refined);
		for (size_t d = 0; d < X.tied_at.size(); ++d)
			if (d < 8 || (d & (d + 1)) == 0 || d % 32 == 31 || d + 1 == X.tied_at.size())
				fprintf(stderr, "[galn-gpu-bwt] probe: tied after %6zu symbols: %14.0f (%.4f%%)\n", 32 * (d + 1), X.tied_at[d], 100.0 * X.tied_at[d] / n);
		exit(0);
	}
	GALN_VLOG("[galn-gpu-bwt] n=%llu: pass 1 %.2f s on %d GPU(s), %zu chunks of up to %llu suffixes (radix sort %.2f s), up to %d text rounds; "
	        "%llu suffixes tied (%.3f%%)\n", (ull)n, t_p1, na, chunks.size(), (ull)C, X.t_sort, max_rounds, (ull)u, 100.0 * u / n);

	/* ---- doubling (device 0), with pass 2 on all GPUs ---- */
	t = galn_realtime();
	int rounds = 0, epochs = 0;
	uint64_t h = min_depth == INT_MAX ? KSYM : (uint64_t)min_depth;
	double t_p2 = 0, t_park = 0;
	D[0].MC.release();   /* device 0's chunk buffers come back for each pass 2 */
	if (u) {
		uint32_t *mark = (uint32_t*)M.get(nmw * 4);
		D[0].mark = mark;
		if (!mark || u >= (uint64_t)INT_MAX - 4096) { GALN_VLOG("[galn-gpu-bwt] %llu tied suffixes: too many\n", (ull)u); return -1; }
		/* the doubling state: the tied suffixes by id (positions, group ranks), their positions sorted with their
		 * ids, and the a active ones (ids and ranks in list order, three more buffers for sorting, flags, output) */
		DevMem MS;
		uint64_t *upos = NULL, *head = NULL, *Spos = NULL, *Suid = NULL, *bf[5] = {}, *Lp = NULL, *Sp = NULL;
		uint8_t *F = NULL, *keep = NULL;
		void *utmp = NULL;
		size_t ub = 0;
		auto state_alloc = [&](uint64_t a) -> bool {
			upos = (uint64_t*)MS.get(u * 8); head = (uint64_t*)MS.get(u * 8); Spos = (uint64_t*)MS.get(u * 8); Suid = (uint64_t*)MS.get(u * 8);
			bool ok = upos && head && Spos && Suid;
			for (int i = 0; i < 5; ++i) { bf[i] = (uint64_t*)MS.get(a * 8); ok = ok && bf[i]; }
			F = (uint8_t*)MS.get(a); keep = (uint8_t*)MS.get(a);
			Lp = (uint64_t*)MS.get(a * 8); Sp = (uint64_t*)MS.get((a + 4) * 8);   /* at most one sample per two tied suffixes */
			ub = cub_bytes(a);
			utmp = MS.get(ub);
			ok = ok && F && keep && Lp && Sp && utmp;
			if (!ok) MS.release();
			return ok;
		};
		if (!state_alloc(u)) { GALN_VLOG("[galn-gpu-bwt] %llu tied suffixes do not fit in device memory\n", (ull)u); return -1; }
		/* ids are the chunks' order (rank order): perm = identity, Row = ranks */
		{
			std::vector<uint64_t> a(u);
			const int fld[3] = {0, 2, 1};
			uint64_t *dst[3] = {upos, head, bf[4]};
			for (int f = 0; f < 3; ++f) {
#pragma omp parallel for schedule(dynamic)
				for (int64_t i = 0; i < (int64_t)chunks.size(); ++i)
					for (uint64_t j = 0; j < uat[i + 1] - uat[i]; ++j) a[uat[i] + j] = Uc[i][3 * j + fld[f]];
				GCK(cudaMemcpy(dst[f], a.data(), u * 8, cudaMemcpyHostToDevice));
			}
		}
		std::vector<std::vector<uint64_t>>().swap(Uc);
		{   /* tied positions sorted, with their ids */
			GCK(cudaMemcpy(bf[0], upos, u * 8, cudaMemcpyDeviceToDevice));
			iota_kernel<<<nblocks(u), TPB>>>(bf[2], u);
			cub::DoubleBuffer<uint64_t> dk(bf[0], bf[1]), dv(bf[2], bf[3]);
			size_t tb = ub; GCK(cub::DeviceRadixSort::SortPairs(utmp, tb, dk, dv, (int)u, 0, nbits(n) + 1));
			GCK(cudaMemcpy(Spos, dk.Current(), u * 8, cudaMemcpyDeviceToDevice));
			GCK(cudaMemcpy(Suid, dv.Current(), u * 8, cudaMemcpyDeviceToDevice));
		}
		uint64_t *perm = bf[0], *perm2 = bf[1], *K = bf[2], *K2 = bf[3], *Row = bf[4];
		iota_kernel<<<nblocks(u), TPB>>>(perm, u);
		const int kb = nbits(n) + 1;
		ull *err = X.ctr + 5, *nL = X.ctr + 6, *nS = X.ctr + 7;
		GCK(cudaMemset(err, 0, 8));
		std::vector<uint64_t> hp, park;
		uint64_t a = u;
		const char *env_k = getenv("GALN_GPU_BWT_ROUNDS");   /* testing: doubling rounds per pass 2 */
		while (a) {
			/* rounds for this epoch: as many as the table of looked-up ranks allows (16 bytes per entry, next to the
			 * doubling state, and in pass 2 next to the chunk buffers: 20% of device 0) */
			GCK(cudaMemGetInfo(&fr, &tot));
			const uint64_t m_max = (uint64_t)std::max(0.0, std::min({((double)fr - margin) / 16.0, 0.2 * avail / 16.0, (double)INT_MAX - 4096}));
			int nr = (int)std::max<uint64_t>(1, std::min<uint64_t>(40, m_max / a));
			if (env_k) nr = std::max(1, atoi(env_k));
			while (nr > 1 && (h << (nr - 1)) >= n) --nr;
			uint64_t m;
			for (;;) {
				GCK(cudaMemset(mark, 0, nmw * 4));
				mark_kernel<<<nblocks(a), TPB>>>(perm, a, upos, h, nr, n, Spos, u, mark);
				GCK(cudaMemset(X.ctr + 2, 0, 8));
				popc_kernel<<<nblocks(nmw), TPB>>>(mark, nmw, X.ctr + 2);
				GCK(cudaGetLastError());
				m = dval(X.ctr + 2);
				if (m <= m_max || nr == 1) break;
				nr = std::max(1, nr / 2);
			}
			if (m > m_max) { GALN_VLOG("[galn-gpu-bwt] %llu looked-up ranks do not fit in device memory\n", (ull)m); return -1; }
			uint64_t *Tq = NULL, *Tr = NULL;
			DevMem MT;
			if (m) {
				double ts = galn_realtime();
				/* the marked suffixes of each chunk and their place in the table */
				std::vector<uint64_t> want(chunks.size()), at(chunks.size());
				{
					ull *bc = (ull*)MT.get(NBUCKET * 8);
					if (!bc) { GALN_VLOG("[galn-gpu-bwt] not enough device memory for pass 2\n"); return -1; }
					GCK(cudaMemset(bc, 0, NBUCKET * 8));
					mark_bucket_kernel<<<nblocks(nmw), TPB>>>(T, mark, nmw, n, bc);
					GCK(cudaGetLastError());
					std::vector<ull> mb(NBUCKET);
					GCK(cudaMemcpy(mb.data(), bc, NBUCKET * 8, cudaMemcpyDeviceToHost));
					MT.put(bc);
					uint64_t s = 0;
					for (size_t i = 0; i < chunks.size(); ++i) {
						uint64_t w = 0;
						for (uint64_t b = chunks[i].b0; b < chunks[i].b1; ++b) w += mb[b];
						want[i] = w; at[i] = s; s += w;
					}
					if (s != m) { fprintf(stderr, "[galn index] internal error: %llu of %llu marked suffixes in the chunks\n", (ull)s, (ull)m); exit(4); }
				}
				/* the doubling state goes to host memory while the chunk buffers are back */
				double tk = galn_realtime();
				uint64_t *const live[6] = {upos, head, Spos, Suid, perm, Row};
				const uint64_t len[6] = {u, u, u, u, a, a};
				park.resize(4 * u + 2 * a);
				for (uint64_t i = 0, o = 0; i < 6; o += len[i], ++i) GCK(cudaMemcpy(park.data() + o, live[i], len[i] * 8, cudaMemcpyDeviceToHost));
				MS.release();
				t_park += galn_realtime() - tk;
				uint64_t *q1 = (uint64_t*)MT.get(m * 8), *r1 = (uint64_t*)MT.get(m * 8);
				if (!q1 || !r1 || !chunk_buffers(D[0], C, tmp_bytes)) { GALN_VLOG("[galn-gpu-bwt] not enough device memory for pass 2\n"); return -1; }
				pass2(act, chunks, want, at, mark, nmw, q1, r1);
				D[0].MC.release();
				uint64_t *q2 = (uint64_t*)MT.get(m * 8), *r2 = (uint64_t*)MT.get(m * 8);
				size_t mb = 0;
				{ cub::DoubleBuffer<uint64_t> dk(NULL, NULL), dv(NULL, NULL); GCK(cub::DeviceRadixSort::SortPairs(NULL, mb, dk, dv, (int)m, 0, kb)); }
				void *mtmp = MT.get(mb);
				if (!q2 || !r2 || !mtmp) { GALN_VLOG("[galn-gpu-bwt] %llu looked-up ranks do not fit in device memory\n", (ull)m); return -1; }
				cub::DoubleBuffer<uint64_t> dk(q1, q2), dv(r1, r2);
				size_t tb = mb; GCK(cub::DeviceRadixSort::SortPairs(mtmp, tb, dk, dv, (int)m, 0, kb));
				Tq = dk.Current(); Tr = dv.Current();
				GCK(cudaDeviceSynchronize());
				MT.put(dk.Alternate()); MT.put(dv.Alternate()); MT.put(mtmp);
				tk = galn_realtime();
				if (!state_alloc(a)) { GALN_VLOG("[galn-gpu-bwt] the doubling state does not fit back in device memory\n"); return -1; }
				perm = bf[0]; perm2 = bf[1]; K = bf[2]; K2 = bf[3]; Row = bf[4];
				uint64_t *const back[6] = {upos, head, Spos, Suid, perm, Row};
				for (uint64_t i = 0, o = 0; i < 6; o += len[i], ++i) GCK(cudaMemcpy(back[i], park.data() + o, len[i] * 8, cudaMemcpyHostToDevice));
				t_park += galn_realtime() - tk;
				t_p2 += galn_realtime() - ts;
			}
			++epochs;
			Lookup L = {n, u, m, Spos, Suid, head, Tq, Tr, err};
			for (int k = 0; k < nr && a; ++k) {
				size_t tb;
				/* order within groups by the rank h further on: sort by that rank, then (stable) by group */
				key_kernel<<<nblocks(a), TPB>>>(perm, a, upos, h, L, K);
				cub::DoubleBuffer<uint64_t> dk(K, K2), dv(perm, perm2);
				tb = ub; GCK(cub::DeviceRadixSort::SortPairs(utmp, tb, dk, dv, (int)a, 0, kb));
				gather_kernel<<<nblocks(a), TPB>>>(head, dv.Current(), a, dk.Current());
				tb = ub; GCK(cub::DeviceRadixSort::SortPairs(utmp, tb, dk, dv, (int)a, 0, kb));
				uint64_t *G = dk.Current(), *Kx = dk.Alternate(), *Pc = dv.Current(), *Pf = dv.Alternate();
				key_kernel<<<nblocks(a), TPB>>>(Pc, a, upos, h, L, Kx);
				group_flag_kernel<<<nblocks(a), TPB>>>(G, Kx, G, a, n, 0, F, Pf);
				tb = ub; GCK(cub::DeviceScan::InclusiveScan(utmp, tb, Pf, G, MaxOp(), (int)a));   /* group starts -> G */
				GCK(cudaMemset(nL, 0, 16));
				round_finish_kernel<<<nblocks(a), TPB>>>(T, Pc, G, Row, F, a, upos, head, keep, Lp, nL, Sp, nS, X.ctr + 3);
				GCK(cudaGetLastError());
				/* final suffixes: their BWT symbols and samples to the host */
				uint64_t fl = dval(nL), fs = dval(nS);
				hp.resize(std::max(fl, 2 * fs));
				if (fl) {
					GCK(cudaMemcpy(hp.data(), Lp, fl * 8, cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
					for (int64_t i = 0; i < (int64_t)fl; ++i) Lf[hp[i] >> 3] = (uint8_t)(hp[i] & 7);
				}
				if (fs) {
					GCK(cudaMemcpy(hp.data(), Sp, fs * 16, cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
					for (int64_t i = 0; i < (int64_t)fs; ++i) samples[hp[2 * i] - 1] = hp[2 * i + 1];
				}
				size_t tb2 = ub;
				GCK(cub::DeviceSelect::Flagged(utmp, tb2, Pc, keep, Pf, X.dnum, (int)a));
				tb2 = ub;
				GCK(cub::DeviceSelect::Flagged(utmp, tb2, Row, keep, Kx, X.dnum, (int)a));
				int na2;
				GCK(cudaMemcpy(&na2, X.dnum, sizeof(int), cudaMemcpyDeviceToHost));
				/* next round: ids in Pf, ranks in Kx; the other three buffers are free */
				uint64_t *rest[3], *all[5] = {perm, perm2, K, K2, Row};
				int r = 0;
				for (int i = 0; i < 5; ++i) if (all[i] != Pf && all[i] != Kx) rest[r++] = all[i];
				perm = Pf; Row = Kx; perm2 = rest[0]; K = rest[1]; K2 = rest[2];
				a = (uint64_t)na2;
				h <<= 1;
				++rounds;
				if (h >= (1ULL << 41) && a) { fprintf(stderr, "[galn index] internal error: prefix doubling did not finish\n"); exit(4); }
			}
			if (dval(err)) { fprintf(stderr, "[galn index] internal error: %llu ranks not found\n", (ull)dval(err)); exit(4); }
		}
	}
	double t_b = galn_realtime() - t;

	/* ---- the $ row and checks ---- */
	t = galn_realtime();
	const uint64_t primary = dval(X.ctr + 3);
	Lf[0] = (uint8_t)(3 - (pac[0] >> 6 & 3));   /* L of the $ row is T[n-1] = complement of F[0] */
	uint64_t bad_l = 0, bad_s = 0;
#pragma omp parallel for schedule(static) reduction(+:bad_l)
	for (int64_t i = 1; i <= (int64_t)n; ++i) bad_l += Lf[i] == L_UNSET;
#pragma omp parallel for schedule(static) reduction(+:bad_s)
	for (int64_t i = 0; i < (int64_t)ns; ++i) bad_s += samples[i] == NO_SAMPLE;
	if (bad_l || bad_s || primary > n || Lf[primary] != L_DOLLAR) {
		fprintf(stderr, "[galn index] internal error: %llu rows and %llu samples missing, primary %llu\n", (ull)bad_l, (ull)bad_s, (ull)primary);
		exit(4);
	}
	GALN_VLOG("[galn-gpu-bwt] text + buckets %.2f s, pass 1 %.2f s, doubling %.2f s (%d rounds in %d epochs, pass 2 %.2f s incl. "
	        "%.2f s moving the doubling state, longest tie < %llu symbols), checks %.2f s, total %.2f s\n", t_text, t_p1, t_b, rounds,
	        epochs, t_p2, t_park, (ull)h, galn_realtime() - t, galn_realtime() - t0);
	return (int64_t)primary;
}
