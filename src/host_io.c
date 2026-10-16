#define _GNU_SOURCE
#include "host_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

/* Standard BWA nucleotide table: A/a=0 C/c=1 G/g=2 T/t=3, everything else 4 ('-'=5). */
static const unsigned char nt4[256] = {
    4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4,
    4,4,4,4, 4,4,4,4, 4,4,4,4, 4,5,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4,
    4,0,4,1, 4,4,4,2, 4,4,4,4, 4,4,4,4, 4,4,4,4, 3,4,4,4, 4,4,4,4, 4,4,4,4,
    4,0,4,1, 4,4,4,2, 4,4,4,4, 4,4,4,4, 4,4,4,4, 3,4,4,4, 4,4,4,4, 4,4,4,4,
    4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4,
    4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4,
    4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4,
    4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4, 4,4,4,4
};

/* Load the .bwt into a 2 MB-aligned buffer (huge pages where available, so a
 * 64-byte occ block never straddles cache lines), reading in parallel chunks. */
uint32_t *load_bwt(const char *fn, bwtgpu_meta_t *m)
{
    int fd = open(fn, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[load_bwt] cannot open %s\n", fn); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    uint64_t fsize = (uint64_t)st.st_size;
    /* header = 5 * sizeof(uint64_t) = primary + L2[1..4] */
    uint64_t hdr[5];
    if (fsize < sizeof(hdr) || pread(fd, hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        fprintf(stderr, "[load_bwt] short read on %s\n", fn); close(fd); return NULL;
    }
    m->primary = hdr[0];
    m->L2[0] = 0;
    memcpy(m->L2 + 1, hdr + 1, 4 * sizeof(uint64_t));
    m->seq_len = m->L2[4];
    m->bwt_size = (fsize - sizeof(hdr)) >> 2;

    size_t bytes = (size_t)m->bwt_size * 4, align = (size_t)2 << 20;
    size_t alloc = (bytes + align - 1) / align * align;
    void *mem = NULL;
    if (posix_memalign(&mem, align, alloc ? alloc : align) != 0) { close(fd); return NULL; }
#ifdef MADV_HUGEPAGE
    madvise(mem, alloc, MADV_HUGEPAGE);
#endif
    const size_t chunk = (size_t)64 << 20;
    long nchunk = (long)((bytes + chunk - 1) / chunk);
    int ok = 1, nthr = 16;
    const char *env = getenv("BWAGPU_LOAD_THREADS");
    if (env && atoi(env) > 0) nthr = atoi(env);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthr)
    for (long c = 0; c < nchunk; ++c) {
        size_t off = (size_t)c * chunk, len = bytes - off < chunk? bytes - off : chunk, done = 0;
        while (done < len) {
            ssize_t r = pread(fd, (char*)mem + off + done, len - done, (off_t)(sizeof(hdr) + off + done));
            if (r <= 0) { ok = 0; break; }
            done += (size_t)r;
        }
    }
    close(fd);
    if (!ok) { fprintf(stderr, "[load_bwt] short read on %s\n", fn); free(mem); return NULL; }
    return (uint32_t*)mem;
}

/* bwa's FASTQ/FASTA parser; gzread errors are recorded instead of being taken for EOF */
static int g_gz_failed = 0;
static int gzread_chk(gzFile f, void *buf, unsigned len)
{
    int r = gzread(f, buf, len);
    if (r < 0) { g_gz_failed = 1; return 0; }
    return r;
}
#include "kseq.h"
KSEQ_INIT(gzFile, gzread_chk)

read_t *read_fastq(const char *fn, int *n)
{
    gzFile fp = gzopen(fn, "rb");   /* reads plain or gzip-compressed input */
    if (!fp) { fprintf(stderr, "[read_fastq] cannot open %s\n", fn); return NULL; }
    gzbuffer(fp, 1 << 20);
    g_gz_failed = 0;
    kseq_t *ks = kseq_init(fp);
    size_t cap = 1024; int cnt = 0, batch = 0, in_batch = 0;
    read_t *reads = (read_t*)malloc(cap * sizeof(read_t));
    for (;;) {
        int l = kseq_read(ks);
        if (l < 0) {
            /* bwa_read_seq: the batch ends here; an empty batch ends the run */
            if (in_batch == 0 || l == -1) break;
            ++batch; in_batch = 0;           /* -2 (malformed record): the next batch continues after it */
            continue;
        }
        if (ks->seq.l == 0) continue;          /* seq.l <= l_bc: skipped */
        int len = (int)ks->seq.l;
        char *name = strdup(ks->name.s);
        uint8_t *q = (uint8_t*)malloc((size_t)len);
        /* query = complement(nt4(seq[len-1-j])) : reverse + complement in one pass */
        for (int j = 0; j < len; ++j) {
            uint8_t b = nt4[(unsigned char)ks->seq.s[len - 1 - j]];
            q[j] = (b > 3) ? 4 : (uint8_t)(3 - b);
        }
        if ((size_t)cnt >= cap) { cap <<= 1; reads = (read_t*)realloc(reads, cap * sizeof(read_t)); }
        reads[cnt].name = name; reads[cnt].len = len; reads[cnt].batch = batch; reads[cnt].query = q;
        ++cnt;
        if (++in_batch == HOST_ALN_BATCH) { ++batch; in_batch = 0; }
    }
    kseq_destroy(ks);
    int zc = gzclose(fp);
    if (g_gz_failed || zc != Z_OK) {
        fprintf(stderr, "[read_fastq] error reading %s (%s)\n", fn, g_gz_failed? "decompression error" : "truncated or corrupt input");
        free_reads(reads, cnt);
        return NULL;
    }
    *n = cnt;
    return reads;
}

int read_batch_ranges(const read_t *r, int n, int **start)
{
    int nb = 0;
    *start = (int*)malloc((size_t)(n + 2) * sizeof(int));
    for (int i = 0; i < n; ++i)
        if (i == 0 || r[i].batch != r[i - 1].batch) (*start)[nb++] = i;
    (*start)[nb] = n;
    return nb;
}

void free_reads(read_t *r, int n)
{
    for (int i = 0; i < n; ++i) { free(r[i].name); free(r[i].query); }
    free(r);
}
