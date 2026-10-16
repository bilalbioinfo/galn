/* opt_build.h — reconstruct BWA's gap_opt_t option block for the .sai header.
 * Mirrors gap_init_opt() + the option handling of bwa_aln() in bwtaln.c
 * (0.7.19-r1273). The resulting block is written verbatim into the .sai.
 *
 * NOTE: when -n is given as a FRACTION (contains '.'), upstream sets
 * max_diff=-1 and fnr=frac, and recomputes max_diff per read at run time (on a
 * local copy) — the value written to the header stays -1. Integer -n sets
 * max_diff=N and fnr=-1. */
#ifndef OPT_BUILD_H
#define OPT_BUILD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sai_format.h"

/* upstream mode bits */
#define OPT_MODE_GAPE      0x01
#define OPT_MODE_COMPREAD  0x02
#define OPT_MODE_LOGGAP    0x04
#define OPT_MODE_NONSTOP   0x10

static inline sai_gap_opt_t opt_default(void)
{
    sai_gap_opt_t o;
    o.s_mm = 3; o.s_gapo = 11; o.s_gape = 4;
    o.max_diff = -1; o.max_gapo = 1; o.max_gape = 6;
    o.indel_end_skip = 5; o.max_del_occ = 10; o.max_entries = 2000000;
    o.mode = OPT_MODE_GAPE | OPT_MODE_COMPREAD;
    o.seed_len = 32; o.max_seed_diff = 2;
    o.fnr = 0.04f;
    o.n_threads = 1;
    o.max_top2 = 30;
    o.trim_qual = 0;
    return o;
}

/* Driver-only knobs (not part of the .sai header). */
typedef struct {
    const char *tiers; /* GPU: comma-separated per-thread stack capacities, tried in order; "max" == max_entries+16 */
    double pool_gb;    /* GPU: device memory for the per-thread stack pools; 0 = all free memory minus 2 GiB */
    int aln_cap;       /* GPU: alignment records per thread in the first tier (x4 per further tier) */
    int cpu_workers;   /* hybrid: Grace worker threads (0 = all available cores minus one) */
    const char *heavy_tiers; /* hybrid: GPU stack caps for heavy reads ("" = heavy reads only on CPU) */
    int heavy_lanes;   /* hybrid: reads per warp in the heavy tiers (1 = one read per warp) */
    int gpu_only, cpu_only;
    int reserve;       /* hybrid: heavy reads the GPU always leaves for the CPU workers (0 = 2 x workers) */
    int heavy_engine;  /* hybrid: 1 = warp-per-read engine (gapw.cuh), 0 = thread-per-read heavy tiers */
    int verify;        /* hybrid: re-run every warp-engine read on the CPU and compare per read */
    int static_split;  /* hybrid: give the warp engine a fixed share of the heavy reads instead of a shared queue */
    double warp_pool_gb; /* hybrid: warp-engine chunk pool size (0 = all free device memory but 2 GiB) */
    int warp_max_warps;  /* hybrid: cap on resident warps of the warp engine (0 = occupancy limit) */
    int warp_shared_pool; /* hybrid: warps share one chunk pool instead of private arenas */
    int stats;         /* print a histogram of per-read peak stack entries to stderr */
    int batch;      /* reads per batch: fixed at upstream's 0x40000, because the batch
                     * max read length feeds the max_gapo clamp in bwa_cal_sa_reg_gap */
} aln_extra_t;

static inline aln_extra_t extra_default(void)
{
    aln_extra_t x;
    x.tiers = "4096,65536,262144,1048576,max";
    x.pool_gb = 0.0;
    x.aln_cap = 32;
    x.stats = 0;
    x.cpu_workers = 0;
    x.heavy_tiers = "262144";
    x.heavy_lanes = 1;
    x.gpu_only = 0;
    x.cpu_only = 0;
    x.reserve = 0;
    x.heavy_engine = 1;
    x.verify = 0;
    x.static_split = 0;
    x.warp_pool_gb = 0;
    x.warp_max_warps = 0;
    x.warp_shared_pool = 0;
    x.batch = 0x40000;
    return x;
}

/* Parse bwa-aln options from argv[start..argc-1] (after <prefix> <reads.fq>).
 * Same semantics as bwa_aln()'s getopt loop, plus --tiers/--pool-gb/--alncap (GPU only)
 * and --stats (CPU only).
 * Returns 0 on success, -1 on a bad or unsupported option. */
static inline int opt_parse(int argc, char **argv, int start, sai_gap_opt_t *o, aln_extra_t *x)
{
    int i, opte = -1;
    *o = opt_default();
    *x = extra_default();
    for (i = start; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc)? argv[i + 1] : NULL;
        if (!strcmp(a, "-L")) { o->mode |= OPT_MODE_LOGGAP; continue; }
        if (!strcmp(a, "-N")) { o->mode |= OPT_MODE_NONSTOP; o->max_top2 = 0x7fffffff; continue; }
        if (!strcmp(a, "--stats")) { x->stats = 1; continue; }
        if (!strcmp(a, "--gpu-only")) { x->gpu_only = 1; continue; }
        if (!strcmp(a, "--cpu-only")) { x->cpu_only = 1; continue; }
        if (!strcmp(a, "--verify")) { x->verify = 1; continue; }
        if (!strcmp(a, "--static-split")) { x->static_split = 1; continue; }
        if (!strcmp(a, "--warp-shared-pool")) { x->warp_shared_pool = 1; continue; }
        if (!v) { fprintf(stderr, "[opt_parse] option %s needs a value\n", a); return -1; }
        if (!strcmp(a, "-n")) {
            if (strstr(v, ".")) o->fnr = atof(v), o->max_diff = -1;
            else o->max_diff = atoi(v), o->fnr = -1.0;
        }
        else if (!strcmp(a, "-o")) o->max_gapo = atoi(v);
        else if (!strcmp(a, "-e")) opte = atoi(v);
        else if (!strcmp(a, "-M")) o->s_mm = atoi(v);
        else if (!strcmp(a, "-O")) o->s_gapo = atoi(v);
        else if (!strcmp(a, "-E")) o->s_gape = atoi(v);
        else if (!strcmp(a, "-d")) o->max_del_occ = atoi(v);
        else if (!strcmp(a, "-i")) o->indel_end_skip = atoi(v);
        else if (!strcmp(a, "-l")) o->seed_len = atoi(v);
        else if (!strcmp(a, "-k")) o->max_seed_diff = atoi(v);
        else if (!strcmp(a, "-m")) o->max_entries = atoi(v);
        else if (!strcmp(a, "-t")) o->n_threads = atoi(v);
        else if (!strcmp(a, "-R")) o->max_top2 = atoi(v);
        else if (!strcmp(a, "-q")) {
            o->trim_qual = atoi(v);
            if (o->trim_qual >= 1) { fprintf(stderr, "[opt_parse] -q (quality trimming) is not ported yet\n"); return -1; }
        }
        else if (!strcmp(a, "--tiers"))   x->tiers = v;
        else if (!strcmp(a, "--pool-gb")) x->pool_gb = atof(v);
        else if (!strcmp(a, "--alncap")) x->aln_cap = atoi(v);
        else if (!strcmp(a, "--cpu-workers")) x->cpu_workers = atoi(v);
        else if (!strcmp(a, "--heavy-tiers")) x->heavy_tiers = v;
        else if (!strcmp(a, "--heavy-lanes")) x->heavy_lanes = atoi(v);
        else if (!strcmp(a, "--reserve")) x->reserve = atoi(v);
        else if (!strcmp(a, "--warp-pool-gb")) x->warp_pool_gb = atof(v);
        else if (!strcmp(a, "--warp-max-warps")) x->warp_max_warps = atoi(v);
        else if (!strcmp(a, "--heavy-engine")) {
            if (!strcmp(v, "warp")) x->heavy_engine = 1;
            else if (!strcmp(v, "thread")) x->heavy_engine = 0;
            else { fprintf(stderr, "[opt_parse] --heavy-engine must be warp or thread\n"); return -1; }
        }
        else { fprintf(stderr, "[opt_parse] unsupported option %s\n", a); return -1; }
        ++i;
    }
    if (opte > 0) {
        o->max_gape = opte;
        o->mode &= ~OPT_MODE_GAPE;
    }
    if (o->n_threads < 1) o->n_threads = 1;
    if (x->pool_gb < 0) x->pool_gb = 0.0;
    if (x->aln_cap < 1) x->aln_cap = 1;
    if (x->heavy_lanes < 1 || x->heavy_lanes > 32) x->heavy_lanes = 1;
    if (x->cpu_workers < 0) x->cpu_workers = 0;
    return 0;
}

#endif
