/* gpu_index.cu: the GPU index builder of `galn index`, writing the .bwt and .sa files of BWA 0.7.19 `bwa index`.
 *
 *   1. FASTA -> .pac/.ann/.amb as upstream bns_fasta2bntseq(for_only=1) writes them               (CPU, OpenMP)
 *   2. forward .pac -> T = F . revcomp(F), one symbol per byte                                       (CPU, OpenMP)
 *   3. BWT of T$; returns `primary`                                                                   (GPU)
 *      -A gpu: n < 2^31: libcubwt (prefix doubling + DC3, ~20.5n bytes of device memory); longer
 *              texts: galn_gpu_bwt_ranks (gpu_sa_ranks.cu, a rank per position, ~6.25n bytes);
 *      -A gpu-lowmem: galn_gpu_bwt (gpu_sa.cu, ~0.4n bytes + working space, passes on all GPUs;
 *              also gives the SA samples of step 5);
 *      -A auto (default): the first of these that fits. If none does, galn index stops and asks for -a cpu.
 *   4. pack BWT 2 bits/symbol + interleave Occ every 128 symbols (bwt_bwtupdate_core layout)        (GPU + CUB scan)
 *   5. SA samples SA'[32k]: from every row r = 32k walk LF until the next row that is a multiple
 *      of 32 (each walk ~32 steps, all walks independent), then list-rank the resulting linked
 *      list of n/32 nodes by pointer jumping. SA'[r] = (#LF steps from r back to row 0) - 1.          (GPU)
 *   6. write .bwt / .sa (.bwt written by a host thread while step 5 runs)                            (CPU)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "idxcommon.h"
#include "libcubwt.cuh"
#include "gpu_sa.h"
extern "C" int host_mem_cpu_nodes_only(void);   /* host_io.c */

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
	fprintf(stderr, "[galn index] CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static const uint32_t END = 0xffffffffu;
enum { ALGO_AUTO, ALGO_GPU, ALGO_LOWMEM };   /* -A: auto, gpu (libcubwt / galn_gpu_bwt_ranks), gpu-lowmem (galn_gpu_bwt) */

struct LFParams {
	const uint32_t *bwt; /* bwa interleaved BWT+Occ body */
	uint64_t primary;
	uint64_t L2[5];
};

/* ---------------- step 4: pack + per-block counts ---------------- */
__global__ void pack_kernel(const uint8_t *__restrict__ L, uint64_t n, uint64_t nb, uint32_t *__restrict__ buf, uint64_t *__restrict__ cnt)
{
	uint64_t b = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (b >= nb) return;
	uint64_t s = b * 128, e = s + 128 < n ? s + 128 : n;
	uint32_t c[4] = {0, 0, 0, 0};
	uint32_t *w = buf + b * 16 + 8;
	int nwb = (int)((e - s + 15) / 16);
	for (int j = 0; j < nwb; ++j) {
		uint64_t ws = s + (uint64_t)j * 16;
		uint32_t word = 0;
		if (ws + 16 <= e) {
			uint4 v = *(const uint4 *)(L + ws);
			uint32_t x[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
			for (int q = 0; q < 16; ++q) {
				uint32_t sym = (x[q >> 2] >> ((q & 3) * 8)) & 3; /* little-endian byte q */
				word |= sym << ((15 - q) << 1);
				c[sym]++;
			}
		} else {
			for (uint64_t i = ws; i < e; ++i) {
				uint32_t sym = L[i] & 3;
				word |= sym << ((15 - (i & 15)) << 1);
				c[sym]++;
			}
		}
		w[j] = word;
	}
	for (int k = 0; k < 4; ++k) cnt[k * (nb + 1) + b] = c[k];
}

__global__ void write_occ_kernel(const uint64_t *__restrict__ occ, uint64_t n, uint64_t nb, uint32_t *__restrict__ buf)
{
	uint64_t b = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (b > nb) return;
	uint32_t *dst = b < nb ? buf + b * 16 : buf + 8 * nb + (n + 15) / 16; /* last element after the last (partial) block */
	for (int k = 0; k < 4; ++k) {
		uint64_t v = occ[k * (nb + 1) + b];
		dst[2 * k] = (uint32_t)v;
		dst[2 * k + 1] = (uint32_t)(v >> 32);
	}
}

/* ---------------- step 5: LF walks + list ranking ---------------- */
/* LF(k) exactly as bwa's bwt_invPsi(): row k -> row of the suffix one position to the left; LF(primary) = 0. */
__device__ __forceinline__ uint64_t lf_step(const LFParams &p, uint64_t k)
{
	if (k == p.primary) return 0;
	uint64_t x = k - (k > p.primary); /* position in the $-removed BWT string */
	const uint32_t *blk = p.bwt + ((x >> 7) << 4);
	const uint32_t *w = blk + 8;
	uint32_t wi = (uint32_t)((x & 127) >> 4);
	uint32_t wx = __ldg(w + wi);
	uint32_t c = (wx >> ((~x & 15) << 1)) & 3;
	uint64_t occ = (uint64_t)__ldg(blk + 2 * c) | ((uint64_t)__ldg(blk + 2 * c + 1) << 32);
	uint32_t pat = c * 0x55555555u, acc = 0;
	for (uint32_t j = 0; j < wi; ++j) {
		uint32_t y = __ldg(w + j) ^ pat;
		y = (y | (y >> 1)) & 0x55555555u;
		acc += 16 - __popc(y);
	}
	uint32_t m = (uint32_t)(x & 15) + 1;
	uint32_t y = wx ^ pat;
	y = (y | (y >> 1)) & 0x55555555u;
	uint32_t mask = m == 16 ? 0xffffffffu : ~(0xffffffffu >> (2 * m));
	acc += m - __popc(y & mask);
	return p.L2[c] + occ + acc;
}

__global__ void walk_kernel(LFParams p, uint64_t m, uint32_t *__restrict__ next, uint32_t *__restrict__ dist)
{
	uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (t >= m) return;
	uint64_t r = t * GALN_SA_INTV;
	uint32_t d = 0;
	do {
		r = lf_step(p, r);
		++d;
	} while (r & (GALN_SA_INTV - 1));
	uint32_t s = (uint32_t)(r / GALN_SA_INTV);
	next[t] = s == 0 ? END : s; /* the walk that wraps to row 0 ($) is the tail of the list */
	dist[t] = d;
}

__global__ void init_rank_kernel(const uint32_t *__restrict__ dist, uint64_t m, uint64_t *__restrict__ R)
{
	uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (t < m) R[t] = dist[t];
}

__global__ void jump_kernel(const uint32_t *__restrict__ nx_in, const uint64_t *__restrict__ R_in,
							uint32_t *__restrict__ nx_out, uint64_t *__restrict__ R_out, uint64_t m, int *changed)
{
	uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (t >= m) return;
	uint32_t s = nx_in[t];
	if (s == END) {
		nx_out[t] = END;
		R_out[t] = R_in[t];
	} else {
		uint32_t s2 = nx_in[s];
		R_out[t] = R_in[t] + R_in[s];
		nx_out[t] = s2;
		if (s2 != END) *changed = 1;
	}
}

/* samples[k-1] = SA'[32k] = R[k] - 1, k = 1..m-1 */
__global__ void finish_sa_kernel(const uint64_t *__restrict__ R, uint64_t m, uint64_t *__restrict__ samples)
{
	uint64_t t = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x + 1;
	if (t < m) samples[t - 1] = R[t] - 1;
}

static inline unsigned nblk(uint64_t n, unsigned bs) { return (unsigned)((n + bs - 1) / bs); }

/* Step 5: SA'[32k] for k = 1..m-1 from the bwa BWT+Occ body already on the device (dBuf).
 * Equivalent to bwt_cal_sa(bwt, 32) but parallel: LF walks between sampled rows + list ranking. */
static void sample_sa_gpu(const uint32_t *dBuf, uint64_t n, uint64_t primary, const uint64_t L2[5], uint64_t *hSamples)
{
	const uint64_t m = (n + GALN_SA_INTV) / GALN_SA_INTV; /* n_sa: rows 0, 32, ..., 32*(m-1) <= n */
	if (m - 1 >= END) { fprintf(stderr, "[galn index] internal error: too many SA samples for 32-bit list nodes\n"); exit(4); }
	cudaEvent_t ev0, ev1;
	float ms, ms_walk;
	CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
	uint32_t *dNext, *dDist, *dNx2; uint64_t *dR, *dR2; int *dChanged;
	CK(cudaMalloc(&dNext, m * 4)); CK(cudaMalloc(&dDist, m * 4)); CK(cudaMalloc(&dNx2, m * 4));
	CK(cudaMalloc(&dR, m * 8)); CK(cudaMalloc(&dR2, m * 8)); CK(cudaMalloc(&dChanged, sizeof(int)));
	LFParams p; p.bwt = dBuf; p.primary = primary; for (int k = 0; k < 5; ++k) p.L2[k] = L2[k];
	CK(cudaEventRecord(ev0));
	walk_kernel<<<nblk(m, 128), 128>>>(p, m, dNext, dDist);
	CK(cudaGetLastError());
	CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1)); CK(cudaEventElapsedTime(&ms_walk, ev0, ev1));
	CK(cudaEventRecord(ev0));
	init_rank_kernel<<<nblk(m, 256), 256>>>(dDist, m, dR);
	int iters = 0, changed = 1;
	uint32_t *nxA = dNext, *nxB = dNx2; uint64_t *RA = dR, *RB = dR2;
	while (changed) {
		CK(cudaMemset(dChanged, 0, sizeof(int)));
		jump_kernel<<<nblk(m, 256), 256>>>(nxA, RA, nxB, RB, m, dChanged);
		CK(cudaGetLastError());
		CK(cudaMemcpy(&changed, dChanged, sizeof(int), cudaMemcpyDeviceToHost));
		std::swap(nxA, nxB); std::swap(RA, RB);
		++iters;
		if (iters > 70) { fprintf(stderr, "[galn index] internal error: list ranking did not converge\n"); exit(4); }
	}
	uint64_t R0;
	CK(cudaMemcpy(&R0, RA, 8, cudaMemcpyDeviceToHost));
	if (R0 != n + 1) { fprintf(stderr, "[galn index] internal error: LF cycle length %llu != n+1\n", (unsigned long long)R0); exit(4); }
	uint64_t *dSamples = RB; /* reuse the free ping-pong buffer */
	finish_sa_kernel<<<nblk(m, 256), 256>>>(RA, m, dSamples);
	CK(cudaGetLastError());
	if (m > 1) CK(cudaMemcpy(hSamples, dSamples, (m - 1) * 8, cudaMemcpyDeviceToHost));
	CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1)); CK(cudaEventElapsedTime(&ms, ev0, ev1));
	GALN_VLOG("[galn-index-gpu] SA sampling: LF walks %.3f ms over %llu rows (%llu walks), list ranking %d rounds + D2H %.3f ms\n",
			ms_walk, (unsigned long long)n + 1, (unsigned long long)m, iters, ms);
	CK(cudaFree(dNext)); CK(cudaFree(dDist)); CK(cudaFree(dNx2)); CK(cudaFree(dR)); CK(cudaFree(dR2)); CK(cudaFree(dChanged));
	CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1));
}

#ifdef main   /* built into galn: entry point galn_index_gpu_main(), callable from C */
extern "C" int main(int argc, char *argv[]);
#endif
int main(int argc, char *argv[])
{
	int c, threads = 0;
	const char *fa = NULL;
	int algo = ALGO_AUTO;
	while ((c = getopt(argc, argv, "f:t:A:v")) >= 0) {
		if (c == 'v') galn_index_verbose = 1;
		else if (c == 'A') {
			if (!strcmp(optarg, "gpu")) algo = ALGO_GPU;
			else if (!strcmp(optarg, "gpu-lowmem")) algo = ALGO_LOWMEM;
			else if (strcmp(optarg, "auto")) { fprintf(stderr, "[galn index] unknown builder %s\n", optarg); return 1; }
		}
		else if (c == 'f') fa = optarg;
		else if (c == 't') threads = atoi(optarg);
		else return 1;
	}
	if (optind >= argc) {
		fprintf(stderr, "Usage: galn-index-gpu [-f in.fasta] [-t threads] [-A auto|gpu|gpu-lowmem] [-v] <prefix>\n");
		return 1;
	}
	std::string prefix = argv[optind];
	if (threads > 0) omp_set_num_threads(threads);
	if (host_mem_cpu_nodes_only()) GALN_VLOG("[galn-index-gpu] host memory bound to the NUMA nodes with CPUs\n");
	double t_all = galn_realtime(), t;
	cudaEvent_t ev0, ev1;
	float ms;

	cudaDeviceProp prop;
	{
		int dev, ndev = 0;
		t = galn_realtime();
		cudaError_t e = cudaGetDeviceCount(&ndev);
		if (e != cudaSuccess || ndev == 0) {
			fprintf(stderr, "[galn index] no usable GPU (%s): to index on the CPU, use galn index -a cpu\n",
					e != cudaSuccess ? cudaGetErrorString(e) : "no CUDA device");
			return 1;
		}
		CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev)); CK(cudaFree(0));
		size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
		GALN_VLOG("[galn-index-gpu] device %s, %.1f GB total / %.1f GB free; CUDA init %.2f s; %d host threads\n",
				prop.name, tot / 1e9, fr / 1e9, galn_realtime() - t, omp_get_max_threads());
	}
	CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));

	int64_t l_pac;
	uint8_t *pac = NULL;
	if (fa) {
		t = galn_realtime();
		l_pac = galn_fasta_to_pac(fa, prefix.c_str(), &pac);
		GALN_VLOG("[galn-index-gpu] FASTA -> .pac/.ann/.amb (l_pac=%lld): %.2f s\n", (long long)l_pac, galn_realtime() - t);
		fprintf(stderr, "[galn index] %s: %.2f Gbp\n", fa, l_pac / 1e9);
	}

	/* step 2 */
	t = galn_realtime();
	if (!pac) pac = galn_read_pac((prefix + ".pac").c_str(), &l_pac);
	const uint64_t n = 2 * (uint64_t)l_pac;
	if (n < 16) {
		fprintf(stderr, "[galn index] the reference is too short for the GPU builders: use galn index -a cpu\n");
		return 1;
	}
	uint8_t *hT;
	CK(cudaHostAlloc((void **)&hT, n + 128, cudaHostAllocDefault));
	GALN_VLOG("[galn-index-gpu] .pac read, n=%llu: %.2f s\n", (unsigned long long)n, galn_realtime() - t);

	fprintf(stderr, "[galn index] building the index on the GPU (%s)\n", prop.name);
	/* step 3: BWT into hT: L with $ removed (libcubwt, galn_gpu_bwt_ranks), or, from galn_gpu_bwt, by row
	 * with the $ row kept (at primary) and the SA samples already made */
	t = galn_realtime();
	uint64_t primary = 0;
	const uint64_t m = (n + GALN_SA_INTV) / GALN_SA_INTV; /* n_sa: rows 0, 32, ..., 32*(m-1) <= n */
	uint64_t *hSamples = NULL;
	int done = 0, have_samples = 0;
	const bool small = n < ((uint64_t)1 << 31);
	if (algo != ALGO_LOWMEM && small) {   /* libcubwt */
		galn_pac_to_text(pac, l_pac, hT);
		void *storage = NULL;
		int64_t rc = libcubwt_allocate_device_storage(&storage, (int64_t)n);
		if (rc != LIBCUBWT_NO_ERROR) {
			GALN_VLOG("[galn-index-gpu] libcubwt cannot handle n=%llu (error %lld)\n", (unsigned long long)n, (long long)rc);
		} else {
			double t1 = galn_realtime();
			int64_t idx = libcubwt_bwt(storage, hT, hT, (int64_t)n);
			if (idx < 0) { fprintf(stderr, "[galn index] libcubwt_bwt failed: %lld\n", (long long)idx); return 3; }
			double t2 = galn_realtime();
			libcubwt_free_device_storage(storage);
			primary = (uint64_t)idx; done = 1;
			GALN_VLOG("[galn-index-gpu] libcubwt BWT: %.3f s (alloc %.3f s, bwt %.3f s incl. H2D/D2H), primary=%llu\n",
					galn_realtime() - t, t1 - t, t2 - t1, (unsigned long long)primary);
		}
	}
	if (!done && algo != ALGO_LOWMEM && !small) {   /* a rank per position, if it fits: the faster builder */
		size_t fr, tot;
		CK(cudaMemGetInfo(&fr, &tot));
		if (algo == ALGO_GPU || 6.25 * n + (4.0 * (1ULL << 30)) <= 0.6 * fr) {
			int64_t pr = galn_gpu_bwt_ranks(pac, l_pac, hT);
			if (pr >= 0) {
				primary = (uint64_t)pr; done = 1;
				GALN_VLOG("[galn-index-gpu] BWT (galn_gpu_bwt_ranks): %.3f s, primary=%llu\n", galn_realtime() - t, (unsigned long long)primary);
			}
		}
	}
	if (!done && algo != ALGO_GPU) {   /* memory-light, on all GPUs */
		CK(cudaHostAlloc((void **)&hSamples, (m > 1 ? m - 1 : 1) * 8, cudaHostAllocDefault));
		int64_t pr = galn_gpu_bwt(pac, l_pac, hT, hSamples);
		if (pr >= 0) {
			primary = (uint64_t)pr; done = have_samples = 1;
			GALN_VLOG("[galn-index-gpu] BWT and SA samples (galn_gpu_bwt): %.3f s, primary=%llu\n", galn_realtime() - t, (unsigned long long)primary);
		} else {
			CK(cudaFreeHost(hSamples)); hSamples = NULL;
			GALN_VLOG("[galn-index-gpu] galn_gpu_bwt cannot handle n=%llu\n", (unsigned long long)n);
		}
	}
	if (!done) {
		if (algo == ALGO_GPU)
			fprintf(stderr, "[galn index] -a gpu: not enough GPU memory for this reference: try galn index -a gpu-lowmem\n");
		else
			fprintf(stderr, "[galn index] %s: not enough GPU memory for this reference: index it on the CPU with galn index -a cpu\n",
					algo == ALGO_LOWMEM ? "-a gpu-lowmem" : "the GPU builders");
		return 1;
	}
	free(pac);

	/* step 4: pack the BWT 2 bits/symbol with Occ interleaved, on the GPU if it fits there (always when the SA
	 * samples still have to be made on the GPU), otherwise on the host cores */
	t = galn_realtime();
	const uint64_t nb = (n + 127) / 128, n_words = galn_bwt_words(n);
	uint64_t L2[5];
	uint32_t *dBuf = NULL, *hBuf;
	CK(cudaHostAlloc((void **)&hBuf, n_words * 4, cudaHostAllocDefault));
	size_t fr4, tot4;
	CK(cudaMemGetInfo(&fr4, &tot4));
	const double need4 = (double)(nb * 128 + 16) + n_words * 4.0 + 64.0 * (nb + 1) + (1ULL << 30);
	if (!have_samples || need4 <= fr4) {
		uint8_t *dL; uint64_t *dCnt, *dOcc;
		CK(cudaMalloc(&dL, nb * 128 + 16));
		CK(cudaMemset(dL, 0, nb * 128 + 16));
		CK(cudaMalloc(&dBuf, n_words * 4));
		CK(cudaMalloc(&dCnt, 4 * (nb + 1) * 8));
		CK(cudaMalloc(&dOcc, 4 * (nb + 1) * 8));
		CK(cudaMemset(dCnt, 0, 4 * (nb + 1) * 8));
		CK(cudaEventRecord(ev0));
		if (have_samples) {   /* leave out the $ row */
			CK(cudaMemcpy(dL, hT, primary, cudaMemcpyHostToDevice));
			CK(cudaMemcpy(dL + primary, hT + primary + 1, n - primary, cudaMemcpyHostToDevice));
		} else CK(cudaMemcpy(dL, hT, n, cudaMemcpyHostToDevice));
		pack_kernel<<<nblk(nb, 256), 256>>>(dL, n, nb, dBuf, dCnt);
		CK(cudaGetLastError());
		{
			size_t tmp_bytes = 0; void *dTmp = NULL;
			CK(cub::DeviceScan::ExclusiveSum(dTmp, tmp_bytes, dCnt, dOcc, (int64_t)(nb + 1)));
			CK(cudaMalloc(&dTmp, tmp_bytes));
			for (int k = 0; k < 4; ++k)
				CK(cub::DeviceScan::ExclusiveSum(dTmp, tmp_bytes, dCnt + k * (nb + 1), dOcc + k * (nb + 1), (int64_t)(nb + 1)));
			CK(cudaFree(dTmp));
		}
		write_occ_kernel<<<nblk(nb + 1, 256), 256>>>(dOcc, n, nb, dBuf);
		CK(cudaGetLastError());
		uint64_t tot[4];
		for (int k = 0; k < 4; ++k) CK(cudaMemcpy(&tot[k], dOcc + k * (nb + 1) + nb, 8, cudaMemcpyDeviceToHost));
		CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1)); CK(cudaEventElapsedTime(&ms, ev0, ev1));
		CK(cudaFree(dL)); CK(cudaFree(dCnt)); CK(cudaFree(dOcc));
		L2[0] = 0; L2[1] = tot[0]; L2[2] = L2[1] + tot[1]; L2[3] = L2[2] + tot[2]; L2[4] = L2[3] + tot[3];
		CK(cudaMemcpy(hBuf, dBuf, n_words * 4, cudaMemcpyDeviceToHost));
		if (have_samples) { CK(cudaFree(dBuf)); dBuf = NULL; }
		GALN_VLOG("[galn-index-gpu] pack + Occ interleave on the GPU: %.3f s (GPU %.3f ms incl. H2D of L), L2=%llu,%llu,%llu,%llu\n",
				galn_realtime() - t, ms, (unsigned long long)L2[1], (unsigned long long)L2[2], (unsigned long long)L2[3], (unsigned long long)L2[4]);
	} else {
		galn_pack_bwt_skip(hT, n, primary, hBuf, L2);
		GALN_VLOG("[galn-index-gpu] pack + Occ interleave on %d host threads: %.3f s, L2=%llu,%llu,%llu,%llu\n", omp_get_max_threads(),
				galn_realtime() - t, (unsigned long long)L2[1], (unsigned long long)L2[2], (unsigned long long)L2[3], (unsigned long long)L2[4]);
	}
	CK(cudaFreeHost(hT));
	if (L2[4] != n) { fprintf(stderr, "[galn index] internal error: L2[4]=%llu != n\n", (unsigned long long)L2[4]); return 4; }

	/* .bwt: written by a background thread while the SA sampling runs */
	double t_bwt_write = 0;
	std::thread wr([&]() { double a = galn_realtime(); galn_write_bwt((prefix + ".bwt").c_str(), primary, L2, hBuf, n_words); t_bwt_write = galn_realtime() - a; });

	/* step 5 */
	if (!have_samples) {
		t = galn_realtime();
		CK(cudaHostAlloc((void **)&hSamples, (m > 1 ? m - 1 : 1) * 8, cudaHostAllocDefault));
		sample_sa_gpu(dBuf, n, primary, L2, hSamples);
		GALN_VLOG("[galn-index-gpu] SA sampling total: %.3f s\n", galn_realtime() - t);
	}

	t = galn_realtime();
	galn_write_sa((prefix + ".sa").c_str(), primary, L2, n, hSamples, m - 1);
	double t_sa_write = galn_realtime() - t;
	wr.join();
	GALN_VLOG("[galn-index-gpu] write .sa %.3f s; write .bwt (overlapped) %.3f s; wait %.3f s\n", t_sa_write, t_bwt_write, galn_realtime() - t - t_sa_write);
	if (dBuf) CK(cudaFree(dBuf));
	CK(cudaFreeHost(hBuf)); CK(cudaFreeHost(hSamples));
	GALN_VLOG("[galn-index-gpu] total wall time %.2f s\n", galn_realtime() - t_all);
	fprintf(stderr, "[galn index] done in %.1f s: %s.{pac,ann,amb,bwt,sa}\n", galn_realtime() - t_all, prefix.c_str());
	return 0;
}
