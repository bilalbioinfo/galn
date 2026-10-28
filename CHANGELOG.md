# Changelog

## 0.2.7 (2026-10-28)

- New `galn samse -k INT`: keeps multi-mapping reads in the BAM file like `bowtie2 -k`. Up to INT records per read, best
  first: the main alignment (the same as `bwa samse`'s) and up to INT-1 more hits as secondary alignments (flag 256,
  MAPQ 0, NM and MD tags). Hits that run past the end of a reference sequence are left out.
- New `galn samse --no-unal`: leaves out reads that did not align.
- `galn index` no longer falls back to the CPU when the reference does not fit in GPU memory: it stops and asks for
  `galn index -a cpu`. Every GPU error is now reported.
- `galn samse` and `galn sampe` print one summary line at the end; `-v` gives the details.
- `galn aln` on GH200 nodes: the host-memory budget now counts only the CPU memory (the GPU memory is a NUMA node
  too), so large indexes no longer run out of memory (50M reads against a 67 Gbp index: 234 GB, before: killed).

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
