/* galn_index.c: `galn index`, the front end of the G-ALN index builders.
 *
 *   galn index [-a auto|gpu|gpu-lowmem|cpu] [-p prefix] [-t threads] [-v] <in.fasta>
 *
 * Writes <prefix>.{pac,ann,amb,bwt,sa}, byte-identical to `bwa index` (BWA 0.7.19): the BWT of a text is
 * unique, so every builder gives the same files.
 *   -a auto        (default) the first GPU builder that fits: gpu, then gpu-lowmem; if neither fits,
 *                  galn index stops and asks for -a cpu
 *   -a gpu         libcubwt up to about 2^31 symbols, longer texts galn_gpu_bwt_ranks (a rank per position)
 *   -a gpu-lowmem  galn_gpu_bwt: 2-bit text and a bitmap on each GPU, passes over all GPUs
 *   -a cpu         CPU-only builder (libsais64, no GPU needed); --cpu is the same
 * bwa's -a is, bwtsw and rb2 are accepted and mean auto. The prefix defaults to the FASTA file name, as in bwa index.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int galn_index_gpu_main(int argc, char *argv[]);   /* gpu_index.cu */
int galn_index_cpu_main(int argc, char *argv[]);   /* cpu_ref.c */

static int index_usage(void)
{
    fprintf(stderr,
        "\nUsage:   galn index [-a STR] [-p prefix] [-t threads] [-v] <in.fasta>\n\n"
        "Options: -a STR   index builder [auto]:\n"
        "                    auto        the fastest that fits in GPU memory: gpu, then gpu-lowmem\n"
        "                    gpu         fastest, but needs more memory\n"
        "                    gpu-lowmem  for large references\n"
        "                    cpu         CPU only (no GPU)\n"
        "         -p STR   prefix of the index files [same as the FASTA file name]\n"
        "         -t INT   CPU threads for the CPU parts [all available]\n"
        "         -v       print the time of each step\n\n");
    return 1;
}

int galn_index(int argc, char *argv[])
{
    const char *prefix = NULL, *fa = NULL, *threads = NULL, *algo = NULL;
    int cpu = 0, verbose = 0;
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc) prefix = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = argv[++i];
        else if (!strcmp(argv[i], "--cpu")) cpu = 1;
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "-a") && i + 1 < argc) {
            const char *a = argv[++i];
            if (!strcmp(a, "cpu")) cpu = 1;
            else if (!strcmp(a, "gpu") || !strcmp(a, "gpu-lowmem")) algo = a;
            else if (strcmp(a, "auto") && strcmp(a, "is") && strcmp(a, "bwtsw") && strcmp(a, "rb2")) {
                fprintf(stderr, "[galn index] unknown builder -a %s\n", a);
                return index_usage();
            }
        }
        else if (argv[i][0] == '-') { fprintf(stderr, "[galn index] unknown option %s\n", argv[i]); return index_usage(); }
        else if (!fa) fa = argv[i];
        else return index_usage();
    }
    if (!fa) return index_usage();
    if (!prefix) prefix = fa;
    char *av[12]; int ac = 0;
    av[ac++] = (char*)(cpu? "galn-index-cpu" : "galn-index-gpu");
    if (threads) { av[ac++] = (char*)"-t"; av[ac++] = (char*)threads; }
    if (verbose) av[ac++] = (char*)"-v";
    if (algo && !cpu) { av[ac++] = (char*)"-A"; av[ac++] = (char*)algo; }
    av[ac++] = (char*)"-f"; av[ac++] = (char*)fa;
    av[ac++] = (char*)prefix;
    av[ac] = NULL;
    optind = 1;
    return cpu? galn_index_cpu_main(ac, av) : galn_index_gpu_main(ac, av);
}
