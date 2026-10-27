# Changelog

## 0.2.6 (2026-10-27)

- `galn index` builds the index of large references on the GPU: when a genome does not fit the fastest GPU builder,
  a memory-light builder runs on all GPUs of the node instead of falling back to the CPU.
- New `galn index -a auto|gpu|gpu-lowmem|cpu` to choose the index builder (default `auto`: the fastest that fits).
- `galn index` reads and parses the FASTA file on all CPU cores
- `galn aln` searches on the GPUs also for references over 34 Gbp (up to ~137 Gbp - work in progress for larger references).

## 0.2.4 (2026-10-22)

- `galn index` builds the BWT on the GPU also for genomes over 1 Gbp (before, the suffix array was built on the
  CPU): the Asian elephant genome is indexed in 19 s instead of 82 s on one GH200.
- Faster read input: reads are decompressed with zlib-ng (included) on a separate thread while they are parsed.
- New `galn samse` and `galn sampe`: `bwa samse/sampe` on the CPU cores of the GPU node, writing BAM directly.
- New `galn aln -dt`(dummy threads): sets the thread count written to the `.sai` header, for comparing files with `bwa aln -t`.

## 0.2.1 (2026-10-18)

- Version described in the bioRxiv preprint (Sharif et al. 2026).
