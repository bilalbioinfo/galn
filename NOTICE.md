# Notices

G-ALN (GPU BWA-ALN) is distributed under the GNU General Public License v3 (see `LICENSE`).

- Its alignment search (`src/bwtgap_hd.h`, `src/gapw.cuh`, `src/bwtsearch.h`) is a port of BWA's
  `bwt_match_gap`, `bwt_occ4`/`bwt_2occ4`, `bwt_cal_width` and related code from BWA 0.7.19
  (Heng Li; https://github.com/lh3/bwa; GPLv3).
- `src/kseq.h` and `src/index/bwa_upstream/` (`bntseq.[ch]`, `utils.[ch]`, `kseq.h`, `khash.h`,
  `ksort.h`) are unmodified copies from BWA 0.7.19 carrying MIT licence headers.
- `galn samse` / `galn sampe` use BWA 0.7.19's own code: `src/samse/bwa_upstream/` holds unmodified copies
  of `bwase.c`, `bwape.c`, `bwaseqio.c`, `bwt.c`, `ksw.c`, `bamlite.c`, `kstring.c` and their headers
  (GPLv3, some files with MIT licence headers); `src/samse/bwa_shim.c` copies a few functions of `bwa.c` and
  `bwtaln.c`, and `src/samse/sam_common.c` adapts `bwa_print_sam1`, `bwa_print_sam_hdr` and
  `bwa_refine_gapped` to work on one read at a time.
- `third_party/libsais` (v2.10.4) and `third_party/libcubwt` (v1.6.3), by Ilya Grebnov, are used
  unmodified under the Apache License 2.0 (see their `LICENSE` files).
- `third_party/zlib-ng` (v2.2.4, without its tests and documentation) is used unmodified under the zlib
  licence (see its `LICENSE.md`), built in zlib-compatible mode.
