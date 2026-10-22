/* galn-index-gpu: GPU builder for bwa-compatible (BWA 0.7.19 `bwa index`) .bwt and .sa files.
 *
 * Pipeline (single GPU):
 *   1. [-f] FASTA -> .pac/.ann/.amb with the unmodified upstream bns_fasta2bntseq(for_only=1)   (CPU)
 *   2. forward .pac -> T = F . revcomp(F), one symbol per byte                                  (CPU, OpenMP)
 *   3. BWT of T$ on the GPU; returns `primary`                                                     (GPU)
 *      n < 2^31: libcubwt (prefix doubling + DC3, ~20.5n bytes of device memory);
 *      longer texts: galn_gpu_bwt (gpu_sa.cu: bucketed radix sort + prefix doubling, ~7.2n bytes);
 *      if neither fits in device memory (or with -C), libsais64 on the host cores                (CPU fallback)
 *   4. pack BWT 2 bits/symbol + interleave Occ every 128 symbols (bwt_bwtupdate_core layout)      (GPU + CUB scan)
 *   5. SA samples SA'[32k]: from every row r = 32k walk LF until the next row that is a multiple
 *      of 32 (each walk ~32 steps, all walks independent), then list-rank the resulting linked
 *      list of n/32 nodes by pointer jumping. SA'[r] = (#LF steps from r back to row 0) - 1.         (GPU)
 *   6. write .bwt / .sa (.bwt written by a host thread while step 5 runs)                         (CPU)
 *
 * Usage: galn-index-gpu [-f in.fasta] [-t threads] [-C] <prefix>
 *        galn-index-gpu -B in.bwt <out_prefix>   (GPU replacement of `bwa bwt2sa`, any n < 2^37)
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
#include "libsais64.h"
#include "gpu_sa.h"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
	fprintf(stderr, "[galn-index-gpu] CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static const uint32_t END = 0xffffffffu;

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
	if (m - 1 >= END) { fprintf(stderr, "text too long for 32-bit list nodes\n"); exit(4); }
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
		if (iters > 70) { fprintf(stderr, "internal error: list ranking did not converge\n"); exit(4); }
	}
	uint64_t R0;
	CK(cudaMemcpy(&R0, RA, 8, cudaMemcpyDeviceToHost));
	if (R0 != n + 1) { fprintf(stderr, "internal error: LF cycle length %llu != n+1\n", (unsigned long long)R0); exit(4); }
	uint64_t *dSamples = RB; /* reuse the free ping-pong buffer */
	finish_sa_kernel<<<nblk(m, 256), 256>>>(RA, m, dSamples);
	CK(cudaGetLastError());
	if (m > 1) CK(cudaMemcpy(hSamples, dSamples, (m - 1) * 8, cudaMemcpyDeviceToHost));
	CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1)); CK(cudaEventElapsedTime(&ms, ev0, ev1));
	fprintf(stderr, "[galn-index-gpu] SA sampling: LF walks %.3f ms over %llu rows (%llu walks), list ranking %d rounds + D2H %.3f ms\n",
			ms_walk, (unsigned long long)n + 1, (unsigned long long)m, iters, ms);
	CK(cudaFree(dNext)); CK(cudaFree(dDist)); CK(cudaFree(dNx2)); CK(cudaFree(dR)); CK(cudaFree(dR2)); CK(cudaFree(dChanged));
	CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1));
}

/* -B mode: GPU replacement of `bwa bwt2sa`: read an existing bwa .bwt, write <out>.sa */
static int bwt2sa_mode(const char *fn_bwt, const std::string &out)
{
	double t = galn_realtime(), t0 = t;
	FILE *fp = fopen(fn_bwt, "rb");
	if (!fp) { fprintf(stderr, "cannot open %s\n", fn_bwt); return 1; }
	fseeko(fp, 0, SEEK_END);
	uint64_t fsize = (uint64_t)ftello(fp), primary, L2[5] = {0, 0, 0, 0, 0};
	fseeko(fp, 0, SEEK_SET);
	uint64_t n_words = (fsize - 40) / 4;
	if (fread(&primary, 8, 1, fp) != 1 || fread(L2 + 1, 8, 4, fp) != 4) { fprintf(stderr, "read error\n"); return 1; }
	const uint64_t n = L2[4];
	if (n_words != galn_bwt_words(n)) { fprintf(stderr, "unexpected .bwt size\n"); return 1; }
	uint32_t *hBuf;
	CK(cudaHostAlloc((void **)&hBuf, n_words * 4, cudaHostAllocDefault));
	for (uint64_t got = 0; got < n_words;) {
		size_t r = fread(hBuf + got, 4, std::min<uint64_t>(n_words - got, 1ULL << 26), fp);
		if (r == 0) { fprintf(stderr, "short read\n"); return 1; }
		got += r;
	}
	fclose(fp);
	fprintf(stderr, "[galn-index-gpu] read %s (n=%llu, %.2f GB): %.2f s\n", fn_bwt, (unsigned long long)n, n_words * 4 / 1e9, galn_realtime() - t);
	t = galn_realtime();
	uint32_t *dBuf;
	CK(cudaMalloc(&dBuf, n_words * 4));
	CK(cudaMemcpy(dBuf, hBuf, n_words * 4, cudaMemcpyHostToDevice));
	CK(cudaFreeHost(hBuf));
	fprintf(stderr, "[galn-index-gpu] H2D .bwt: %.3f s\n", galn_realtime() - t);
	const uint64_t m = (n + GALN_SA_INTV) / GALN_SA_INTV;
	uint64_t *hSamples;
	CK(cudaHostAlloc((void **)&hSamples, (m > 1 ? m - 1 : 1) * 8, cudaHostAllocDefault));
	t = galn_realtime();
	sample_sa_gpu(dBuf, n, primary, L2, hSamples);
	fprintf(stderr, "[galn-index-gpu] SA sampling total: %.3f s\n", galn_realtime() - t);
	t = galn_realtime();
	galn_write_sa((out + ".sa").c_str(), primary, L2, n, hSamples, m - 1);
	fprintf(stderr, "[galn-index-gpu] write .sa: %.3f s; total %.2f s\n", galn_realtime() - t, galn_realtime() - t0);
	CK(cudaFree(dBuf)); CK(cudaFreeHost(hSamples));
	return 0;
}

/* Fallback when libcubwt cannot be used (n < 16, n >= ~2^31, or not enough device memory):
 * suffix array with libsais64 on the host CPU cores (64-bit, OpenMP), then BWT in place of T. */
static uint64_t cpu_bwt_libsais(uint8_t *T, uint64_t n, int threads)
{
	double t = galn_realtime();
	int64_t *SA = (int64_t *)malloc(sizeof(int64_t) * (n ? n : 1));
	uint8_t *L = (uint8_t *)malloc(n ? n : 1);
	if (!SA || !L) { fprintf(stderr, "out of host memory\n"); exit(2); }
	if (libsais64_omp(T, SA, (int64_t)n, 0, NULL, threads) != 0) { fprintf(stderr, "libsais64 failed\n"); exit(3); }
	double t1 = galn_realtime();
	int64_t j0 = -1;
#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < (int64_t)n; ++i)
		if (SA[i] == 0) j0 = i;
	uint64_t primary = (uint64_t)j0 + 1;
#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < (int64_t)n; ++i) {
		uint64_t r = (uint64_t)i + ((uint64_t)i >= primary);
		L[i] = r == 0 ? T[n - 1] : T[SA[r - 1] - 1];
	}
	free(SA);
#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < (int64_t)n; ++i) T[i] = L[i];
	free(L);
	fprintf(stderr, "[galn-index-gpu] CPU fallback: libsais64 SA %.2f s + BWT %.2f s (%d threads)\n", t1 - t, galn_realtime() - t1, threads);
	return primary;
}

#ifdef main   /* built into galn: entry point galn_index_gpu_main(), callable from C */
extern "C" int main(int argc, char *argv[]);
#endif
int main(int argc, char *argv[])
{
	int c, threads = 0;
	const char *fa = NULL, *from_bwt = NULL;
	int force_cpu = 0;
	while ((c = getopt(argc, argv, "f:t:B:C")) >= 0) {
		if (c == 'f') fa = optarg;
		else if (c == 'C') force_cpu = 1;
		else if (c == 'B') from_bwt = optarg;
		else if (c == 't') threads = atoi(optarg);
		else return 1;
	}
	if (optind >= argc) {
		fprintf(stderr, "Usage: galn-index-gpu [-f in.fasta] [-t threads] [-C] <prefix>\n");
		fprintf(stderr, "       -C  force the CPU (libsais64) suffix sort (automatic if libcubwt cannot handle n)\n");
		fprintf(stderr, "       galn-index-gpu -B in.bwt <out_prefix>     (GPU bwt2sa: writes <out_prefix>.sa)\n");
		return 1;
	}
	std::string prefix = argv[optind];
	if (threads > 0) omp_set_num_threads(threads);
	double t_all = galn_realtime(), t;
	cudaEvent_t ev0, ev1;
	float ms;

	{
		int dev; cudaDeviceProp prop;
		t = galn_realtime();
		CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev)); CK(cudaFree(0));
		size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
		fprintf(stderr, "[galn-index-gpu] device %s, %.1f GB total / %.1f GB free; CUDA init %.2f s; %d host threads\n",
				prop.name, tot / 1e9, fr / 1e9, galn_realtime() - t, omp_get_max_threads());
	}
	CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
	if (from_bwt) return bwt2sa_mode(from_bwt, prefix);

	if (fa) {
		t = galn_realtime();
		int64_t lp = galn_fasta_to_pac(fa, prefix.c_str());
		fprintf(stderr, "[galn-index-gpu] FASTA -> .pac/.ann/.amb (l_pac=%lld): %.2f s\n", (long long)lp, galn_realtime() - t);
	}

	/* step 2 */
	t = galn_realtime();
	int64_t l_pac;
	uint8_t *pac = galn_read_pac((prefix + ".pac").c_str(), &l_pac);
	const uint64_t n = 2 * (uint64_t)l_pac;
	uint8_t *hT;
	CK(cudaHostAlloc((void **)&hT, n + 128, cudaHostAllocDefault));
	fprintf(stderr, "[galn-index-gpu] .pac read, n=%llu: %.2f s\n", (unsigned long long)n, galn_realtime() - t);

	/* step 3: BWT into hT (L, $ removed) */
	t = galn_realtime();
	uint64_t primary = 0;
	const char *env_gsa = getenv("GALN_GPU_BWT");   /* =1: galn_gpu_bwt for any n (testing) */
	int use_gsa = !force_cpu && n >= 16 && (n >= ((uint64_t)1 << 31) || (env_gsa && *env_gsa == '1'));
	if (use_gsa) {
		int64_t pr = galn_gpu_bwt(pac, l_pac, hT);
		if (pr >= 0) fprintf(stderr, "[galn-index-gpu] BWT (galn_gpu_bwt): %.3f s, primary=%llu\n", galn_realtime() - t, (unsigned long long)(primary = pr));
		else fprintf(stderr, "[galn-index-gpu] galn_gpu_bwt cannot handle n=%llu: falling back to CPU libsais64\n", (unsigned long long)n);
		use_gsa = pr >= 0;
	}
	if (!use_gsa) {
		galn_pac_to_text(pac, l_pac, hT);
		fprintf(stderr, "[galn-index-gpu] text expanded: %.2f s\n", galn_realtime() - t);
	}
	free(pac);
	int use_cubwt = n >= 16 && !force_cpu && !use_gsa;
	if (use_cubwt) {
		void *storage = NULL;
		int64_t rc = libcubwt_allocate_device_storage(&storage, (int64_t)n);
		if (rc != LIBCUBWT_NO_ERROR) {
			fprintf(stderr, "[galn-index-gpu] libcubwt cannot handle n=%llu (error %lld): falling back to CPU libsais64\n", (unsigned long long)n, (long long)rc);
			use_cubwt = 0;
		} else {
			double t1 = galn_realtime();
			int64_t idx = libcubwt_bwt(storage, hT, hT, (int64_t)n);
			if (idx < 0) { fprintf(stderr, "libcubwt_bwt failed: %lld\n", (long long)idx); return 3; }
			double t2 = galn_realtime();
			libcubwt_free_device_storage(storage);
			primary = (uint64_t)idx;
			fprintf(stderr, "[galn-index-gpu] libcubwt BWT: %.3f s (alloc %.3f s, bwt %.3f s incl. H2D/D2H), primary=%llu\n",
					galn_realtime() - t, t1 - t, t2 - t1, (unsigned long long)primary);
		}
	}
	if (!use_cubwt && !use_gsa) {
		primary = cpu_bwt_libsais(hT, n, omp_get_max_threads());
		fprintf(stderr, "[galn-index-gpu] BWT (CPU fallback): %.3f s, primary=%llu\n", galn_realtime() - t, (unsigned long long)primary);
	}

	/* step 4 */
	t = galn_realtime();
	const uint64_t nb = (n + 127) / 128, n_words = galn_bwt_words(n);
	const uint64_t m = (n + GALN_SA_INTV) / GALN_SA_INTV; /* n_sa: rows 0, 32, ..., 32*(m-1) <= n */
	uint8_t *dL; uint32_t *dBuf; uint64_t *dCnt, *dOcc;
	CK(cudaMalloc(&dL, nb * 128 + 16));
	CK(cudaMemset(dL, 0, nb * 128 + 16));
	CK(cudaMalloc(&dBuf, n_words * 4));
	CK(cudaMalloc(&dCnt, 4 * (nb + 1) * 8));
	CK(cudaMalloc(&dOcc, 4 * (nb + 1) * 8));
	CK(cudaMemset(dCnt, 0, 4 * (nb + 1) * 8));
	CK(cudaEventRecord(ev0));
	CK(cudaMemcpy(dL, hT, n, cudaMemcpyHostToDevice));
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
	CK(cudaFreeHost(hT));
	uint64_t L2[5] = {0, tot[0], tot[0] + tot[1], tot[0] + tot[1] + tot[2], tot[0] + tot[1] + tot[2] + tot[3]};
	if (L2[4] != n) { fprintf(stderr, "internal error: L2[4]=%llu != n\n", (unsigned long long)L2[4]); return 4; }
	fprintf(stderr, "[galn-index-gpu] pack + Occ interleave: %.3f s (GPU %.3f ms incl. H2D of L), L2=%llu,%llu,%llu,%llu\n",
			galn_realtime() - t, ms, (unsigned long long)L2[1], (unsigned long long)L2[2], (unsigned long long)L2[3], (unsigned long long)L2[4]);

	/* .bwt: copy to host and write it in a background thread while the SA sampling runs */
	t = galn_realtime();
	uint32_t *hBuf;
	CK(cudaHostAlloc((void **)&hBuf, n_words * 4, cudaHostAllocDefault));
	CK(cudaMemcpy(hBuf, dBuf, n_words * 4, cudaMemcpyDeviceToHost));
	double t_bwt_write = 0;
	std::thread wr([&]() { double a = galn_realtime(); galn_write_bwt((prefix + ".bwt").c_str(), primary, L2, hBuf, n_words); t_bwt_write = galn_realtime() - a; });
	fprintf(stderr, "[galn-index-gpu] D2H .bwt body: %.3f s\n", galn_realtime() - t);

	/* step 5 */
	t = galn_realtime();
	uint64_t *hSamples;
	CK(cudaHostAlloc((void **)&hSamples, (m > 1 ? m - 1 : 1) * 8, cudaHostAllocDefault));
	sample_sa_gpu(dBuf, n, primary, L2, hSamples);
	fprintf(stderr, "[galn-index-gpu] SA sampling total: %.3f s\n", galn_realtime() - t);

	t = galn_realtime();
	galn_write_sa((prefix + ".sa").c_str(), primary, L2, n, hSamples, m - 1);
	double t_sa_write = galn_realtime() - t;
	wr.join();
	fprintf(stderr, "[galn-index-gpu] write .sa %.3f s; write .bwt (overlapped) %.3f s; wait %.3f s\n", t_sa_write, t_bwt_write, galn_realtime() - t - t_sa_write);
	CK(cudaFree(dBuf)); CK(cudaFreeHost(hBuf)); CK(cudaFreeHost(hSamples));
	fprintf(stderr, "[galn-index-gpu] total wall time %.2f s\n", galn_realtime() - t_all);
	return 0;
}
