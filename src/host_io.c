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
#include <sys/syscall.h>
#include <dirent.h>
#include <pthread.h>

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
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   /* the index is in our buffer now: no page-cache copy */
    close(fd);
    if (!ok) { fprintf(stderr, "[load_bwt] short read on %s\n", fn); free(mem); return NULL; }
    return (uint32_t*)mem;
}

/* ---- FASTQ input: a decompression thread fills a ring of chunks, bwa's kseq parser reads them ---- */
#define IN_CHUNK (4 << 20)
#define IN_NCHUNK 16
typedef struct {
    gzFile fp;
    char *buf[IN_NCHUNK];
    int len[IN_NCHUNK];        /* 0: end of input */
    long head, tail;           /* chunks filled, chunks released by the parser */
    int failed, stop;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    char *cur;                 /* parser side: chunk being read */
    int cur_len, cur_pos, eof;
} instream_t;

static void *inflate_main(void *arg)
{
    instream_t *s = (instream_t*)arg;
    for (;;) {
        pthread_mutex_lock(&s->mu);
        while (s->head - s->tail == IN_NCHUNK && !s->stop) pthread_cond_wait(&s->cv, &s->mu);
        int k = (int)(s->head % IN_NCHUNK), stop = s->stop;
        pthread_mutex_unlock(&s->mu);
        if (stop) break;
        int r = gzread(s->fp, s->buf[k], IN_CHUNK);
        pthread_mutex_lock(&s->mu);
        if (r < 0) { s->failed = 1; r = 0; }   /* a decompression error is not taken for EOF */
        s->len[k] = r; ++s->head;
        pthread_cond_signal(&s->cv);
        pthread_mutex_unlock(&s->mu);
        if (r == 0) break;
    }
    return NULL;
}

/* kseq's read function: copies from the current chunk; returns 0 only at the end of input */
static int in_read(instream_t *s, void *dst, int n)
{
    if (s->cur_pos == s->cur_len) {
        if (s->eof) return 0;
        pthread_mutex_lock(&s->mu);
        if (s->cur) { ++s->tail; pthread_cond_signal(&s->cv); }
        while (s->head == s->tail) pthread_cond_wait(&s->cv, &s->mu);
        int k = (int)(s->tail % IN_NCHUNK);
        s->cur = s->buf[k]; s->cur_len = s->len[k]; s->cur_pos = 0;
        pthread_mutex_unlock(&s->mu);
        if (s->cur_len == 0) { s->eof = 1; return 0; }
    }
    int m = s->cur_len - s->cur_pos < n ? s->cur_len - s->cur_pos : n;
    memcpy(dst, s->cur + s->cur_pos, m);
    s->cur_pos += m;
    return m;
}
#include "kseq.h"
KSEQ_INIT(instream_t*, in_read)

/* Read names and queries live in large blocks (galn reads one FASTQ per run; free_reads frees them all). */
#define ARENA_BLOCK ((size_t)64 << 20)
typedef struct arena_blk { struct arena_blk *next; size_t used, cap; char data[]; } arena_blk;
static arena_blk *g_arena = NULL;
static void *arena_alloc(size_t n)
{
    if (!g_arena || g_arena->cap - g_arena->used < n) {
        size_t cap = n > ARENA_BLOCK ? n : ARENA_BLOCK;
        arena_blk *b = (arena_blk*)malloc(sizeof(arena_blk) + cap);
        if (!b) return NULL;
        b->next = g_arena; b->used = 0; b->cap = cap; g_arena = b;
    }
    void *p = g_arena->data + g_arena->used;
    g_arena->used += n;
    return p;
}

read_t *read_fastq(const char *fn, int *n)
{
    instream_t s;
    memset(&s, 0, sizeof(s));
    if (!(s.fp = gzopen(fn, "rb"))) { fprintf(stderr, "[read_fastq] cannot open %s\n", fn); return NULL; }   /* plain or gzip */
    gzbuffer(s.fp, 1 << 20);
    for (int i = 0; i < IN_NCHUNK; ++i) s.buf[i] = (char*)malloc(IN_CHUNK);
    pthread_mutex_init(&s.mu, NULL); pthread_cond_init(&s.cv, NULL);
    pthread_t tid;
    pthread_create(&tid, NULL, inflate_main, &s);
    kseq_t *ks = kseq_init(&s);
    size_t cap = 1024; int cnt = 0, batch = 0, in_batch = 0, oom = 0;
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
        char *name = (char*)arena_alloc(ks->name.l + 1);
        uint8_t *q = (uint8_t*)arena_alloc((size_t)len);
        if (!name || !q) { oom = 1; break; }
        memcpy(name, ks->name.s, ks->name.l + 1);
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
    pthread_mutex_lock(&s.mu);                 /* parsing may stop before the end: release the thread */
    s.stop = 1;
    pthread_cond_signal(&s.cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(tid, NULL);
    kseq_destroy(ks);
    int zc = gzclose(s.fp);
    for (int i = 0; i < IN_NCHUNK; ++i) free(s.buf[i]);
    pthread_mutex_destroy(&s.mu); pthread_cond_destroy(&s.cv);
    if (s.failed || zc != Z_OK || oom) {
        fprintf(stderr, "[read_fastq] error reading %s (%s)\n", fn,
                oom? "out of memory" : s.failed? "decompression error" : "truncated or corrupt input");
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
    (void)n;
    while (g_arena) { arena_blk *b = g_arena; g_arena = b->next; free(b); }
    free(r);
}

/* Keep this process's host memory (and the page cache it fills) on the NUMA nodes that have CPUs. On a
 * GH200 each GPU's HBM is also a NUMA node without CPUs, where Linux would otherwise put host pages once
 * the CPU memory is full, taking that memory from the GPU. Does nothing when every node has CPUs. */
int host_mem_cpu_nodes_only(void)
{
    unsigned long mask[16] = {0};
    int ncpu_nodes = 0, nmem_only = 0;
    DIR *d = opendir("/sys/devices/system/node");
    if (!d) return 0;
    for (struct dirent *e; (e = readdir(d)) != NULL; ) {
        if (strncmp(e->d_name, "node", 4) || e->d_name[4] < '0' || e->d_name[4] > '9') continue;
        int node = atoi(e->d_name + 4);
        if (node < 0 || node >= 16 * 64) continue;
        char path[300], buf[64] = "";
        snprintf(path, sizeof(path), "/sys/devices/system/node/%s/cpulist", e->d_name);
        FILE *fp = fopen(path, "r");
        if (fp) { if (!fgets(buf, sizeof(buf), fp)) buf[0] = 0; fclose(fp); }
        if (buf[0] >= '0' && buf[0] <= '9') { mask[node / 64] |= 1UL << (node % 64); ++ncpu_nodes; }
        else ++nmem_only;
    }
    closedir(d);
    if (!ncpu_nodes || !nmem_only) return 0;
    return syscall(SYS_set_mempolicy, 2 /* MPOL_BIND */, mask, (unsigned long)(16 * 64)) == 0 ? ncpu_nodes : 0;
}
