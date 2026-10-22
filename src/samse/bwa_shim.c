/* bwa_shim.c — the few functions of BWA 0.7.19's bwa.c and bwtaln.c that bwase.c and bwaseqio.c
 * need, copied unchanged (the rest of those files pulls in the whole BWA program). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "bntseq.h"
#include "bwtaln.h"
#include "utils.h"

int bwa_verbose = 3;
char bwa_rg_id[256];
char *bwa_pg;

/* bwa.c */
void bwa_fill_scmat(int a, int b, int8_t mat[25])
{
	int i, j, k;
	for (i = k = 0; i < 4; ++i) {
		for (j = 0; j < 4; ++j)
			mat[k++] = i == j? a : -b;
		mat[k++] = -1; // ambiguous base
	}
	for (j = 0; j < 5; ++j) mat[k++] = -1;
}

/* bwa.c */
char *bwa_idx_infer_prefix(const char *hint)
{
	char *prefix;
	int l_hint;
	FILE *fp;
	l_hint = strlen(hint);
	prefix = malloc(l_hint + 3 + 4 + 1);
	strcpy(prefix, hint);
	strcpy(prefix + l_hint, ".64.bwt");
	if ((fp = fopen(prefix, "rb")) != 0) {
		fclose(fp);
		prefix[l_hint + 3] = 0;
		return prefix;
	} else {
		strcpy(prefix + l_hint, ".bwt");
		if ((fp = fopen(prefix, "rb")) == 0) {
			free(prefix);
			return 0;
		} else {
			fclose(fp);
			prefix[l_hint] = 0;
			return prefix;
		}
	}
}

/* bwa.c */
void bwa_print_sam_hdr(const bntseq_t *bns, const char *hdr_line)
{
	int i, n_HD = 0, n_SQ = 0;
	extern char *bwa_pg;

	if (hdr_line) {
		// check for HD line
		const char *p = hdr_line;
		while ((p = strstr(p, "@HD\t")) != 0) {
			if (p == hdr_line || *(p-1) == '\n') ++n_HD;
			p += 4;
		}
		// check for SQ lines
		p = hdr_line;
		while ((p = strstr(p, "@SQ\t")) != 0) {
			if (p == hdr_line || *(p-1) == '\n') ++n_SQ;
			p += 4;
		}
	}
	if (n_HD == 0) err_printf("@HD\tVN:1.5\tSO:unsorted\tGO:query\n");
	else if (bwa_verbose >= 2)
		fprintf(stderr, "[W::%s] please don't include @HD with option -H. Continue anyway.\n", __func__);
	if (n_SQ == 0) {
		for (i = 0; i < bns->n_seqs; ++i) {
			err_printf("@SQ\tSN:%s\tLN:%d", bns->anns[i].name, bns->anns[i].len);
			if (bns->anns[i].is_alt) err_printf("\tAH:*\n");
			else err_fputc('\n', stdout);
		}
	} else if (n_SQ != bns->n_seqs && bwa_verbose >= 2)
		fprintf(stderr, "[W::%s] %d @SQ lines provided with -H; %d sequences in the index. Continue anyway.\n", __func__, n_SQ, bns->n_seqs);
	if (hdr_line) err_printf("%s\n", hdr_line);
	if (bwa_pg) err_printf("%s\n", bwa_pg);
}

/* bwa.c */
static char *bwa_escape(char *s)
{
	char *p, *q;
	for (p = q = s; *p; ++p) {
		if (*p == '\\') {
			++p;
			if (*p == 't') *q++ = '\t';
			else if (*p == 'n') *q++ = '\n';
			else if (*p == 'r') *q++ = '\r';
			else if (*p == '\\') *q++ = '\\';
		} else *q++ = *p;
	}
	*q = '\0';
	return s;
}

/* bwa.c */
char *bwa_set_rg(const char *s)
{
	char *p, *q, *r, *rg_line = 0;
	memset(bwa_rg_id, 0, 256);
	if (strstr(s, "@RG") != s) {
		if (bwa_verbose >= 1) fprintf(stderr, "[E::%s] the read group line is not started with @RG\n", __func__);
		goto err_set_rg;
	}
	if (strstr(s, "\t") != NULL) {
		if (bwa_verbose >= 1) fprintf(stderr, "[E::%s] the read group line contained literal <tab> characters -- replace with escaped tabs: \\t\n", __func__);
		goto err_set_rg;
	}
	rg_line = strdup(s);
	bwa_escape(rg_line);
	if ((p = strstr(rg_line, "\tID:")) == 0) {
		if (bwa_verbose >= 1) fprintf(stderr, "[E::%s] no ID within the read group line\n", __func__);
		goto err_set_rg;
	}
	p += 4;
	for (q = p; *q && *q != '\t' && *q != '\n'; ++q);
	if (q - p + 1 > 256) {
		if (bwa_verbose >= 1) fprintf(stderr, "[E::%s] @RG:ID is longer than 255 characters\n", __func__);
		goto err_set_rg;
	}
	for (q = p, r = bwa_rg_id; *q && *q != '\t' && *q != '\n'; ++q)
		*r++ = *q;
	return rg_line;

err_set_rg:
	free(rg_line);
	return 0;
}

/* bwtaln.c */
bwa_seqio_t *bwa_open_reads(int mode, const char *fn_fa)
{
	bwa_seqio_t *ks;
	if (mode & BWA_MODE_BAM) { // open BAM
		int which = 0;
		if (mode & BWA_MODE_BAM_SE) which |= 4;
		if (mode & BWA_MODE_BAM_READ1) which |= 1;
		if (mode & BWA_MODE_BAM_READ2) which |= 2;
		if (which == 0) which = 7; // then read all reads
		ks = bwa_bam_open(fn_fa, which);
	} else ks = bwa_seq_open(fn_fa);
	return ks;
}

/* bwtaln.c */
int bwa_cal_maxdiff(int l, double err, double thres)
{
	double elambda = exp(-l * err);
	double sum, y = 1.0;
	int k, x = 1;
	for (k = 1, sum = elambda; k < 1000; ++k) {
		y *= l * err;
		x *= k;
		sum += elambda * y / x;
		if (1.0 - sum < thres) return k;
	}
	return 2;
}
