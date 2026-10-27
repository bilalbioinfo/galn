/* galn-index-cpu: CPU reference builder for bwa-compatible .bwt/.sa (plain suffix array via libsais64).
 *
 * Usage: galn-index-cpu [-t threads] [-f in.fasta] <prefix>
 *   -f  first write <prefix>.pac/.ann/.amb from the FASTA (as upstream bns_fasta2bntseq, for_only=1)
 *   otherwise <prefix>.pac must exist (forward-only .pac as written by `bwa index`).
 * Writes <prefix>.bwt and <prefix>.sa (byte-identical to `bwa index` 0.7.19).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "idxcommon.h"
#include "libsais64.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[])
{
	int c, threads = 0;
	const char *fa = NULL, *prefix;
	char *fn;
	int64_t l_pac, n, i, j0 = -1, n_samples;
	uint8_t *pac, *T, *L;
	int64_t *SA;
	uint64_t L2[5], primary, n_words, *samples;
	uint32_t *buf;
	double t0 = galn_realtime(), t;

	while ((c = getopt(argc, argv, "t:f:v")) >= 0) {
		if (c == 'v') galn_index_verbose = 1;
		else if (c == 't') threads = atoi(optarg);
		else if (c == 'f') fa = optarg;
		else return 1;
	}
	if (optind >= argc) {
		fprintf(stderr, "Usage: galn-index-cpu [-t threads] [-f in.fasta] [-v] <prefix>\n");
		return 1;
	}
	prefix = argv[optind];
	fn = (char*)malloc(strlen(prefix) + 16);
#ifdef _OPENMP
	if (threads > 0) omp_set_num_threads(threads);
	threads = omp_get_max_threads();
#else
	threads = 1;
#endif
	GALN_VLOG("[galn-index-cpu] threads=%d\n", threads);

	pac = NULL;
	if (fa) {
		t = galn_realtime();
		l_pac = galn_fasta_to_pac(fa, prefix, &pac);
		GALN_VLOG("[galn-index-cpu] FASTA -> .pac/.ann/.amb (l_pac=%lld): %.2f s\n", (long long)l_pac, galn_realtime() - t);
		fprintf(stderr, "[galn index] %s: %.2f Gbp\n", fa, l_pac / 1e9);
	}
	fprintf(stderr, "[galn index] building the index on the CPU (%d threads)\n", threads);

	t = galn_realtime();
	sprintf(fn, "%s.pac", prefix);
	if (!pac) pac = galn_read_pac(fn, &l_pac);
	n = 2 * l_pac;
	T = (uint8_t*)malloc(n);
	galn_pac_to_text(pac, l_pac, T);
	free(pac);
	GALN_VLOG("[galn-index-cpu] text n=%lld (l_pac=%lld) loaded: %.2f s\n", (long long)n, (long long)l_pac, galn_realtime() - t);

	t = galn_realtime();
	SA = (int64_t*)malloc(sizeof(int64_t) * n);
	if (!SA || !T) { fprintf(stderr, "out of memory\n"); return 2; }
#ifdef _OPENMP
	if (libsais64_omp(T, SA, n, 0, NULL, threads) != 0) { fprintf(stderr, "libsais64 failed\n"); return 3; }
#else
	if (libsais64(T, SA, n, 0, NULL) != 0) { fprintf(stderr, "libsais64 failed\n"); return 3; }
#endif
	GALN_VLOG("[galn-index-cpu] libsais64 suffix array: %.2f s\n", galn_realtime() - t);

	/* SA'[0] = n (the $ suffix), SA'[i] = SA[i-1]; primary = row with SA' == 0. */
	t = galn_realtime();
#pragma omp parallel for schedule(static)
	for (i = 0; i < n; ++i)
		if (SA[i] == 0) j0 = i;
	primary = (uint64_t)j0 + 1;
	L = (uint8_t*)malloc(n);
	/* L[x] = BWT char of row x + (x >= primary); row r>0 has char T[SA[r-1]-1]; row 0 has T[n-1]. */
#pragma omp parallel for schedule(static)
	for (i = 0; i < n; ++i) {
		uint64_t r = (uint64_t)i + ((uint64_t)i >= primary);
		L[i] = r == 0 ? T[n - 1] : T[SA[r - 1] - 1];
	}
	n_samples = (n + GALN_SA_INTV) / GALN_SA_INTV - 1;
	samples = (uint64_t*)malloc(sizeof(uint64_t) * (n_samples > 0 ? n_samples : 1));
#pragma omp parallel for schedule(static)
	for (i = 1; i <= n_samples; ++i)
		samples[i - 1] = (uint64_t)SA[i * GALN_SA_INTV - 1];
	free(SA); free(T);
	n_words = galn_bwt_words(n);
	buf = (uint32_t*)malloc(sizeof(uint32_t) * n_words);
	galn_pack_bwt(L, n, buf, L2);
	free(L);
	GALN_VLOG("[galn-index-cpu] BWT/Occ/SA-sample derivation: %.2f s (primary=%llu, L2=%llu,%llu,%llu,%llu)\n",
			galn_realtime() - t, (unsigned long long)primary, (unsigned long long)L2[1], (unsigned long long)L2[2],
			(unsigned long long)L2[3], (unsigned long long)L2[4]);

	t = galn_realtime();
	sprintf(fn, "%s.bwt", prefix);
	galn_write_bwt(fn, primary, L2, buf, n_words);
	sprintf(fn, "%s.sa", prefix);
	galn_write_sa(fn, primary, L2, n, samples, n_samples);
	GALN_VLOG("[galn-index-cpu] write .bwt/.sa: %.2f s\n", galn_realtime() - t);
	free(buf); free(samples); free(fn);
	GALN_VLOG("[galn-index-cpu] total wall time %.2f s\n", galn_realtime() - t0);
	fprintf(stderr, "[galn index] done in %.1f s: %s.{pac,ann,amb,bwt,sa}\n", galn_realtime() - t0, prefix);
	return 0;
}
