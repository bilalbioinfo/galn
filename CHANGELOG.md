# Changelog

## 0.2.4 (2026-10-22)

- `galn index` builds the BWT on the GPU also for genomes over 1 Gbp (before, the suffix array was built on the
  CPU): the Asian elephant genome is indexed in 19 s instead of 82 s on one GH200.
- Faster read input: reads are decompressed with zlib-ng (included) on a separate thread while they are parsed.
- New `galn samse` and `galn sampe`: `bwa samse/sampe` on the CPU cores of the GPU node, writing BAM directly.
- New `galn aln -dt`(dummy threads): sets the thread count written to the `.sai` header, for comparing files with `bwa aln -t`.

## 0.2.1 (2026-10-18)

- Version described in the bioRxiv preprint (Sharif et al. 2026).
