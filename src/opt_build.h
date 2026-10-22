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

#endif
