/* galn_index.c — `galn index`: bwa-index-compatible front end of the G-ALN index builders.
 *
 *   galn index [-p prefix] [-t threads] [--cpu] <in.fasta>
 *
 * Writes <prefix>.{pac,ann,amb,bwt,sa}, byte-identical to `bwa index` (BWA 0.7.19; the
 * BWT of a text is unique, so `-a is`, `-a bwtsw` and `-a rb2` all give these files).
 *   default  GPU builder: libcubwt BWT on the GPU when the text fits (about 2^31 symbols
 *            and GPU memory), otherwise a multi-threaded CPU suffix sort (libsais64);
 *            BWT packing, occurrence counts and suffix-array sampling on the GPU.
 *   --cpu    CPU-only builder (no GPU needed).
 * The prefix defaults to the FASTA file name, as in bwa index.
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
        "\nUsage:   galn index [-p prefix] [-t threads] [--cpu] <in.fasta>\n\n"
        "Options: -p STR   prefix of the index files [same as the FASTA file name]\n"
        "         -t INT   CPU threads for the CPU parts [all available]\n"
        "         --cpu    build on the CPU only (no GPU)\n\n"
        "Writes <prefix>.pac .ann .amb .bwt .sa, byte-identical to `bwa index` (BWA 0.7.19).\n"
        "Host memory: about 10 bytes per reference base (~65 GB for a human genome).\n\n");
    return 1;
}

int galn_index(int argc, char *argv[])
{
    const char *prefix = NULL, *fa = NULL, *threads = NULL;
    int cpu = 0;
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc) prefix = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = argv[++i];
        else if (!strcmp(argv[i], "--cpu")) cpu = 1;
        else if (!strcmp(argv[i], "-a") && i + 1 < argc) ++i;   /* bwa's algorithm choice: same result, ignored */
        else if (argv[i][0] == '-') { fprintf(stderr, "[galn index] unknown option %s\n", argv[i]); return index_usage(); }
        else if (!fa) fa = argv[i];
        else return index_usage();
    }
    if (!fa) return index_usage();
    if (!prefix) prefix = fa;
    char *av[8]; int ac = 0;
    av[ac++] = (char*)(cpu? "galn-index-cpu" : "galn-index-gpu");
    if (threads) { av[ac++] = (char*)"-t"; av[ac++] = (char*)threads; }
    av[ac++] = (char*)"-f"; av[ac++] = (char*)fa;
    av[ac++] = (char*)prefix;
    av[ac] = NULL;
    optind = 1;
    return cpu? galn_index_cpu_main(ac, av) : galn_index_gpu_main(ac, av);
}
