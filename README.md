# G-ALN

G-ALN runs `bwa aln` on NVIDIA GPUs. Its output (`.sai`) is byte-identical to that of BWA 0.7.19,
for example for ancient-DNA mapping with `galn aln -l 1024 -n 0.01 -o 2 ref reads.fq.gz > reads.sai`,
and is used with the usual `bwa samse` / `bwa sampe`.

G-ALN also builds the BWA index (`galn index`), with files identical to those of `bwa index`.


## Requirements

- An NVIDIA GPU of compute capability 7.0 or newer (tested on NVIDIA GH200)
- CUDA toolkit 11.8 or newer
- CMake 3.18 or newer, a C/C++ compiler with OpenMP (e.g. GCC), zlib

## Install

```
git clone https://github.com/bilalbioinfo/galn.git
cd galn
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16
```

The program is `build/galn`. To build faster for one GPU type only, add for example
`-DCMAKE_CUDA_ARCHITECTURES=90` (H100/GH200) or `=80` (A100) to the first cmake command.

## Run

1. Index the reference (or use an existing index made by `bwa index`):

```
galn index -p ref ref.fa
```

2. Map the reads:

```
galn aln -l 1024 -n 0.01 -o 2 ref reads.fq.gz > reads.sai
```

3. Make SAM with BWA, as usual:

```
bwa samse ref reads.sai reads.fq.gz > reads.sam
```

G-ALN uses all GPUs and all CPU cores given to the job, e.g. `sbatch --gpus=4 --exclusive ...`.
All `bwa aln` options have the same meaning and defaults as in BWA 0.7.19 (not supported: `-q`, BAM input).
Run `galn aln` without arguments to see all options, including `--gpus`, `--cpu-only` and `--gpu-only`.

## Checking that the output is identical to bwa aln

`bwa aln` writes its `-t` value (number of threads) into the header of the `.sai` file. G-ALN does the
same, but in G-ALN `-t` only sets this header value; it does not change how many threads G-ALN uses.
To get a file identical to your `bwa aln` run, give `galn aln` the same `-t` value:

```
bwa  aln -t 16 -l 1024 -n 0.01 -o 2 ref reads.fq.gz > bwa.sai
galn aln -t 16 -l 1024 -n 0.01 -o 2 ref reads.fq.gz > galn.sai
md5sum bwa.sai galn.sai          # or: cmp bwa.sai galn.sai
```

## License

GPLv3.

## Citation

G-ALN manuscript in preparation. G-ALN is adapted from the BWA source code, so please also cite BWA:

Li H. and Durbin R. (2009) Fast and accurate short read alignment with Burrows-Wheeler transform.
*Bioinformatics*, 25, 1754-1760.
