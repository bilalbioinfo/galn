/* galn.cu — G-ALN (GPU BWA-ALN): `bwa aln` on an NVIDIA GPU, byte-identical .sai.
 *
 *   galn aln [options] <prefix> <reads.fq[.gz]> > out.sai
 *
 * Portable design: uses only device memory and ordinary (asynchronous) copies, so it
 * runs on any CUDA GPU with any host CPU (PCIe or NVLink-C2C, x86 or Arm). The host
 * CPU cores help as much as there are of them; with --gpu-only the GPU does all of it.
 *
 *   light tier   every read runs on the GPU, one thread per read, with a small stack
 *                (--light-cap, default 4096). ~80% of reads finish here.
 *   phase W      reads that overflowed are placed in one list. A persistent kernel
 *                (warp-per-read engine, gapw.cuh) claims positions from the BACK through
 *                a counter in device memory; CPU worker threads claim from the FRONT.
 *                About once per millisecond the driver exchanges the two frontiers with
 *                small async copies on a second stream, so each side stops where the
 *                other one is.
 *   phase G      reads that do not fit in a warp's stack arena ("giant" reads) are
 *                reported back during phase W and taken by the CPU workers first; the
 *                rest run in a second GPU pass with fewer warps and arenas big enough
 *                for any read, again shared with the CPU from the other end.
 *
 * Several GPUs: every GPU runs this pipeline on an interleaved share of the reads
 * (read r goes to GPU r mod n) with its own driver thread; one pool of CPU workers serves
 * all GPUs, each worker preferring the GPU on its own NUMA node and using a copy of the
 * index in its node's memory.
 *
 * Every read is searched from scratch to completion with the same algorithm by
 * whichever engine takes it, and the first complete result is kept (a per-read flag),
 * so the output never depends on timing or on the CPU/GPU split. Reads are processed
 * in bwa's batches (bwa_read_seq), whose options travel with each read.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <deque>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <memory>
#include <algorithm>
#include <cstdarg>
#include <sched.h>
#include <unistd.h>
#include <pthread.h>
#include <dirent.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

extern "C" {
#include "host_io.h"
}
#include "bwtsearch.h"
#include "bwtgap_hd.h"
#include "aln_host.h"
#include "aln_il.h"
#include "sai_writer.h"
#include "opt_build.h"

#include "version.h"

/* fatal error from any thread: the first caller reports and ends the process with _exit
 * (no atexit / CUDA runtime teardown while other threads are still inside CUDA calls) */
static std::atomic<int> g_dying{0};
static void galn_die(const char *fmt, ...)
{
    if (g_dying.exchange(1)) for (;;) pause();
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
    _exit(10);
}

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) galn_die("[galn] CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    } while (0)

/* index descriptor in constant memory: small, read-only, broadcast to all threads */
__constant__ bwtgpu_meta_t g_meta;

#include "gapw.cuh"

/* ------------------------------------------------------------------------- */
/* light tier: one thread per read                                           */

struct light_args {
    const uint32_t *bwt, *cnt_table;
    const uint8_t  *query;
    const uint64_t *qoff;
    const int      *qlen, *maxdiff, *bid, *idx;
    const uint64_t *woff;
    bwtgap_width_t *wbuf, *swbuf;
    bwtgap_entry_t *pool;
    int            *heads;
    bwtgap_aln_t   *aln;
    int            *n_aln;
    unsigned long long *n_pops;
    const bwtgap_opt_t *bopts;
    const int      *nstacks;
    int heads_stride, n, pool_cap, aln_cap, seed_len;
};

__global__ void light_kernel(light_args a)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= a.n) return;
    int r = a.idx[t], b = a.bid[r], len = a.qlen[r];
    const uint8_t *q = a.query + a.qoff[r];
    bwtgap_opt_t o = a.bopts[b];
    o.max_diff = a.maxdiff[r];
    o.seed_len = a.seed_len < len? a.seed_len : 0x7fffffff;
    bwtgap_width_t *w = a.wbuf + a.woff[t], *sw = NULL;
    bwtgap_cal_width(a.bwt, &g_meta, len, q, 1, w);
    if (len > a.seed_len) {
        sw = a.swbuf + a.woff[t];
        bwtgap_cal_width(a.bwt, &g_meta, a.seed_len, q + (len - a.seed_len), 1, sw);
    }
    bwtgap_stack_t st;
    st.pool = a.pool + (size_t)t * a.pool_cap;
    st.cap = a.pool_cap;
    st.heads = a.heads + (size_t)t * a.heads_stride;
    st.n_stacks = a.nstacks[b];
    st.n_pops = 0;
    st.trace = BWTGAP_TRACE_INIT;
    a.n_aln[t] = bwtgap_match_gap(a.bwt, &g_meta, a.cnt_table, len, q, w, sw, &o, &st,
                                  a.aln + (size_t)t * a.aln_cap, a.aln_cap);
    a.n_pops[t] = st.n_pops;
}

/* ------------------------------------------------------------------------- */

static double now_s()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int usage(void)
{
    fprintf(stderr,
        "\nProgram: G-ALN (GPU BWA-ALN), version " GALN_VERSION "\n"
        "         byte-identical replacement for `bwa aln` (BWA 0.7.19) on NVIDIA GPUs\n\n"
        "Usage:   galn index [-p prefix] [-t threads] [--cpu] <in.fasta>   (see `galn index`)\n"
        "         galn aln [options] <prefix> <in.fq[.gz]>\n"
        "         galn samse [options] <prefix> <in.sai> <in.fq> > out.bam   (see `galn samse`)\n"
        "         galn sampe [options] <prefix> <in1.sai> <in2.sai> <in1.fq> <in2.fq> > out.bam   (see `galn sampe`)\n\n"
        "bwa aln options (same meaning and defaults as BWA 0.7.19):\n"
        "         -n NUM    max #diff (int) or missing prob under 0.02 err rate (float) [0.04]\n"
        "         -o INT    maximum number of gap opens [1]\n"
        "         -e INT    maximum number of gap extensions, -1 for disabling long gaps [-1]\n"
        "         -i INT    do not put an indel within INT bp towards the ends [5]\n"
        "         -d INT    maximum occurrences for extending a long deletion [10]\n"
        "         -l INT    seed length [32]\n"
        "         -k INT    maximum differences in the seed [2]\n"
        "         -m INT    maximum entries in the queue [2000000]\n"
        "         -dt INT   dummy threads: only the thread count written to the .sai header, as bwa aln -t [1]\n"
        "                   (G-ALN always uses all GPUs and CPU cores of the job)\n"
        "         -M INT    mismatch penalty [3]\n"
        "         -O INT    gap open penalty [11]\n"
        "         -E INT    gap extension penalty [4]\n"
        "         -R INT    stop searching when there are >INT equally best hits [30]\n"
        "         -L        log-scaled gap penalty for long deletions\n"
        "         -N        non-iterative mode: search for all n-difference hits (slooow)\n"
        "         -f FILE   file to write output to instead of stdout\n"
        "         (not supported: -q, -b/-0/-1/-2, -I, -Y, -B)\n\n"
        "G-ALN options:\n"
        "         --cpu-workers INT  CPU threads helping the GPUs [all cores of the job minus one per GPU]\n"
        "         --cpu-interleave INT  reads each CPU worker advances in turn, overlapping memory waits\n"
        "                            [8, or fewer if the stacks would need more than half the job's memory]\n"
        "         --handoff INT      give a GPU read whose stack passes INT entries to the CPU while the CPU has\n"
        "                            room (0: the GPU keeps every read it can) [1048576]\n"
        "         --handoff-when MODE  endgame: only after the GPU's list of heavy reads is used up; always [endgame]\n"
        "         --device LIST      CUDA device(s) to use, e.g. 0 or 0,1,2,3 [all visible devices]\n"
        "         --gpus INT         use the first INT visible devices [all]\n"
        "         --gpu-only         the GPU does all reads it can (CPU only for reads it cannot take)\n"
        "         --cpu-only         no GPU at all\n"
        "         --light-cap INT    stack entries per read in the light tier [4096]\n"
        "         --warps INT        cap on resident warps of the heavy-read engine [occupancy limit]\n"
        "         --pool-gb FLOAT    GPU memory for search stacks [all free memory but 1 GiB]\n"
        "         --alloc MODE       stack memory of the heavy-read engine: arena (per warp) or shared [arena]\n"
        "         --verify           re-run every heavy read finished on the GPU on the CPU and compare\n"
        "         -v                 print timing and engine statistics to stderr\n\n");
    return 1;
}

/* parse `galn aln` arguments: options may come before or after the two positional arguments */
static int parse_args(int argc, char **argv, sai_gap_opt_t *o, const char **prefix, const char **fq, const char **outfn,
                      int *cpu_workers, int *il_n, long *handoff_entries, int *handoff_endgame, const char **devspec, int *ngpus, int *gpu_only, int *cpu_only, long *light_cap, int *max_warps,
                      double *pool_gb, int *verify, int *verbose, int *alloc_mode)
{
    int opte = -1, npos = 0;
    *o = opt_default();
    for (int i = 0; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc)? argv[i + 1] : NULL;
        if (a[0] != '-' || !a[1]) {
            if (npos == 0) *prefix = a; else if (npos == 1) *fq = a;
            else { fprintf(stderr, "[galn] unexpected argument '%s'\n", a); return -1; }
            ++npos; continue;
        }
        if (!strcmp(a, "-L")) { o->mode |= OPT_MODE_LOGGAP; continue; }
        if (!strcmp(a, "-N")) { o->mode |= OPT_MODE_NONSTOP; o->max_top2 = 0x7fffffff; continue; }
        if (!strcmp(a, "--gpu-only")) { *gpu_only = 1; continue; }
        if (!strcmp(a, "--cpu-only")) { *cpu_only = 1; continue; }
        if (!strcmp(a, "--verify")) { *verify = 1; continue; }
        if (!strcmp(a, "-v")) { *verbose = 1; continue; }
        if (!v) { fprintf(stderr, "[galn] option %s needs a value\n", a); return -1; }
        if (!strcmp(a, "-n")) {
            if (strstr(v, ".")) o->fnr = atof(v), o->max_diff = -1;
            else o->max_diff = atoi(v), o->fnr = -1.0;
        }
        else if (!strcmp(a, "-o")) o->max_gapo = atoi(v);
        else if (!strcmp(a, "-e")) opte = atoi(v);
        else if (!strcmp(a, "-M")) o->s_mm = atoi(v);
        else if (!strcmp(a, "-O")) o->s_gapo = atoi(v);
        else if (!strcmp(a, "-E")) o->s_gape = atoi(v);
        else if (!strcmp(a, "-d")) o->max_del_occ = atoi(v);
        else if (!strcmp(a, "-i")) o->indel_end_skip = atoi(v);
        else if (!strcmp(a, "-l")) o->seed_len = atoi(v);
        else if (!strcmp(a, "-k")) o->max_seed_diff = atoi(v);
        else if (!strcmp(a, "-m")) o->max_entries = atoi(v);
        else if (!strcmp(a, "-dt") || !strcmp(a, "-t")) o->n_threads = atoi(v);   /* -dt is -t: only the .sai header value */
        else if (!strcmp(a, "-R")) o->max_top2 = atoi(v);
        else if (!strcmp(a, "-f")) *outfn = v;
        else if (!strcmp(a, "-q")) {
            if (atoi(v) >= 1) { fprintf(stderr, "[galn] -q (quality trimming) is not supported\n"); return -1; }
        }
        else if (!strcmp(a, "--cpu-workers")) *cpu_workers = atoi(v);
        else if (!strcmp(a, "--cpu-interleave")) *il_n = atoi(v);
        else if (!strcmp(a, "--handoff")) *handoff_entries = atol(v);
        else if (!strcmp(a, "--handoff-when")) {
            if (!strcmp(v, "endgame")) *handoff_endgame = 1; else if (!strcmp(v, "always")) *handoff_endgame = 0;
            else { fprintf(stderr, "[galn] --handoff-when must be endgame or always\n"); return -1; }
        }
        else if (!strcmp(a, "--device")) *devspec = v;
        else if (!strcmp(a, "--gpus")) *ngpus = atoi(v);
        else if (!strcmp(a, "--light-cap")) *light_cap = atol(v);
        else if (!strcmp(a, "--warps")) *max_warps = atoi(v);
        else if (!strcmp(a, "--pool-gb")) *pool_gb = atof(v);
        else if (!strcmp(a, "--alloc")) {
            if (!strcmp(v, "arena")) *alloc_mode = 1; else if (!strcmp(v, "shared")) *alloc_mode = 2;
            else if (!strcmp(v, "auto")) *alloc_mode = 0;
            else { fprintf(stderr, "[galn] --alloc must be arena, shared or auto\n"); return -1; }
        }
        else { fprintf(stderr, "[galn] unsupported option %s\n", a); return -1; }
        ++i;
    }
    if (opte > 0) { o->max_gape = opte; o->mode &= ~OPT_MODE_GAPE; }
    if (o->n_threads < 1) o->n_threads = 1;
    if (npos != 2) return -1;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* a list shared by the CPU (claims from the front) and the GPU (from the back) */

struct TwoEnd {
    std::vector<int> list;
    std::atomic<int> front{0};   /* next position for the CPU */
    std::atomic<int> limit{0};   /* the CPU may claim positions < limit (GPU frontier as last seen) */
    bool active = false;
    int claim() {
        int f = front.load();
        for (;;) {
            if (f >= limit.load() || f >= (int)list.size()) return -1;
            if (front.compare_exchange_weak(f, f + 1)) return f;
        }
    }
    bool claimable() { int f = front.load(); return active && f < limit.load() && f < (int)list.size(); }
    bool drained() { return !active || front.load() >= (int)list.size(); }
    /* end game: once nothing is left to claim, idle CPU workers re-run reads the GPU is
     * still working on (positions >= front); whichever engine finishes first is kept and
     * the GPU abandons its copy. Each position is handed to at most one CPU worker. */
    std::atomic<int> dscan{-1};
    int steal() {
        int d = dscan.load();
        if (d < 0) { int f = front.load(); dscan.compare_exchange_strong(d, f); }
        int p = dscan.fetch_add(1);
        return p < (int)list.size()? p : -1;
    }
};

/* per-GPU state: the reads the device owns and its two shared lists */
struct DevState {
    int dev = 0, numa = -1, drv_cpu = -1;
    std::vector<int> reads;     /* reads this GPU owns (interleaved share of the input) */
    std::deque<int> heavy;      /* light-tier overflow before phase W starts */
    std::deque<int> giant;      /* phase-W give-ups not yet handed to phase G */
    TwoEnd W, G;
    TwoEnd *running = nullptr;  /* the list whose kernel is running (for the end game) */
    bool gpu_running = false;
    bool gpu_done = false;
    std::atomic<bool> cpu_help{true};     /* CPU workers may take this GPU's reads (false under --gpu-only until done) */
    const uint32_t *host_bwt = nullptr;   /* host copy the index is uploaded from */
    /* statistics */
    double t_light = 0, t_W = 0, t_G = 0, k_light = 0, k_W = 0, k_G = 0, pops_W = 0, pops_G = 0, epops = 0;
    long n_light_done = 0, n_W_gpu = 0, n_G_gpu = 0, n_giant = 0, n_unsup = 0, nwW = 0, nwG = 0;
};

struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::deque<int> cpu_only;   /* reads only the CPU can take */
    std::vector<std::unique_ptr<DevState> > devs;
    int ndone = 0;              /* GPUs that finished all their phases */
    bool workers_may_start = true;
    bool any_running() const { for (auto &d : devs) if (d->gpu_running) return true; return false; }
    bool finished() const {
        if (ndone < (int)devs.size() || !cpu_only.empty()) return false;
        for (auto &d : devs) if (!d->W.drained() || !d->G.drained()) return false;
        return true;
    }
};

struct Results {
    std::vector<int> n;
    std::vector<std::vector<bwtgap_aln_t> > aln;
    std::unique_ptr<std::atomic<uint8_t>[]> done;
    bool claim(int r) { uint8_t z = 0; return done[r].compare_exchange_strong(z, 1); }
};

/* ------------------------------------------------------------------------- */
/* CPU topology (Linux sysfs): NUMA node of a CPU and of a GPU                 */

static int cpu_numa_node(int cpu)
{
    char path[96];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu);
    DIR *d = opendir(path);
    if (!d) return 0;
    int node = 0;
    for (struct dirent *e; (e = readdir(d)) != NULL; )
        if (!strncmp(e->d_name, "node", 4) && e->d_name[4] >= '0' && e->d_name[4] <= '9') { node = atoi(e->d_name + 4); break; }
    closedir(d);
    return node;
}

static int gpu_numa_node(int dev)
{
    char bus[64];
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), dev) != cudaSuccess) { cudaGetLastError(); return -1; }
    unsigned dom = 0, b = 0, s = 0, f = 0;
    if (sscanf(bus, "%x:%x:%x.%x", &dom, &b, &s, &f) != 4) return -1;
    char path[160];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%x/numa_node", dom & 0xffff, b, s, f);
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    int n = -1;
    if (fscanf(fp, "%d", &n) != 1) n = -1;
    fclose(fp);
    return n;
}

/* memory this process may still use: the cgroup limit (v2, then v1) or MemAvailable, whichever is smaller */
static double host_mem_avail()
{
    double avail = -1;
    FILE *fp = fopen("/proc/meminfo", "r");
    if (fp) {
        char key[64]; double kb;
        while (fscanf(fp, "%63s %lf kB", key, &kb) == 2)
            if (!strcmp(key, "MemAvailable:")) { avail = kb * 1024.0; break; }
        fclose(fp);
    }
    const char *lim[] = { "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes" };
    const char *use[] = { "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes" };
    for (int i = 0; i < 2; ++i) {
        FILE *a = fopen(lim[i], "r"), *b = fopen(use[i], "r");
        double l = -1, u = 0;
        if (a && fscanf(a, "%lf", &l) != 1) l = -1;
        if (b && fscanf(b, "%lf", &u) != 1) u = 0;
        if (a) fclose(a);
        if (b) fclose(b);
        if (l > 0 && l < 1e18) { double left = l - u; if (avail < 0 || left < avail) avail = left; break; }
    }
    return avail;
}

static void pin_self(int cpu)
{
    if (cpu < 0) return;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

/* ------------------------------------------------------------------------- */

int galn_aln(int argc, char **argv)
{
    sai_gap_opt_t opt;
    const char *prefix = NULL, *fq = NULL, *outfn = NULL, *devspec = NULL;
    int cpu_workers = 0, il_n = 0, ngpus = 0, gpu_only = 0, cpu_only = 0, max_warps = 0, verify = 0, verbose = 0, alloc_mode = 0;
    long light_cap = 4096, handoff_entries = 1L << 20;
    int handoff_endgame = 1;
    double pool_gb = 0;
    if (parse_args(argc, argv, &opt, &prefix, &fq, &outfn, &cpu_workers, &il_n, &handoff_entries, &handoff_endgame, &devspec, &ngpus, &gpu_only, &cpu_only,
                   &light_cap, &max_warps, &pool_gb, &verify, &verbose, &alloc_mode) < 0) return usage();
    if (light_cap < 16) light_cap = 16;

    /* ---- GPUs: --device list, else the first --gpus, else every visible device ---- */
    std::vector<int> devlist;
    if (!cpu_only) {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            fprintf(stderr, "[galn] no CUDA device found; use --cpu-only to run without a GPU\n");
            return 10;
        }
        if (devspec) {
            for (const char *p = devspec; *p; ) {
                char *e;
                long d = strtol(p, &e, 10);
                if (e == p || d < 0 || d >= ndev) { fprintf(stderr, "[galn] --device: no CUDA device '%s' (%d visible)\n", devspec, ndev); return 1; }
                if (std::find(devlist.begin(), devlist.end(), (int)d) == devlist.end()) devlist.push_back((int)d);
                p = *e == ','? e + 1 : e;
                if (*e && *e != ',') { fprintf(stderr, "[galn] --device: bad list '%s'\n", devspec); return 1; }
            }
        } else {
            int n = ngpus > 0? std::min(ngpus, ndev) : ndev;
            for (int d = 0; d < n; ++d) devlist.push_back(d);
        }
        if (devlist.empty()) { fprintf(stderr, "[galn] --device: empty device list\n"); return 1; }
        for (int d : devlist) {   /* before any context exists on the device */
            CUDA_CHECK(cudaSetDevice(d));
            if (cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess) cudaGetLastError();   /* a hint only */
        }
    }
    const int ndev = (int)devlist.size();

    /* ---- CPUs of the job: one driver thread per GPU (on a core next to it), the rest are workers ---- */
    std::vector<int> cpus;
    {
        cpu_set_t cs; CPU_ZERO(&cs);
        if (sched_getaffinity(0, sizeof(cs), &cs) == 0)
            for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &cs)) cpus.push_back(c);
        if (cpus.empty()) for (int c = 0; c < (int)std::thread::hardware_concurrency(); ++c) cpus.push_back(c);
    }
    std::vector<int> cpu_node(cpus.size());
    for (size_t i = 0; i < cpus.size(); ++i) cpu_node[i] = cpu_numa_node(cpus[i]);
    std::vector<char> cpu_taken(cpus.size(), 0);
    std::vector<int> gnuma(ndev, -1), gdrv(ndev, -1);
    for (int i = 0; i < ndev; ++i) {
        gnuma[i] = gpu_numa_node(devlist[i]);
        if ((int)cpus.size() <= ndev) continue;          /* too few CPUs to reserve any */
        int pick = -1, any = -1;
        for (int j = (int)cpus.size() - 1; j >= 0; --j) {   /* last free CPU on the GPU's node, else the last free one */
            if (cpu_taken[j]) continue;
            if (any < 0) any = j;
            if (cpu_node[j] == gnuma[i]) { pick = j; break; }
        }
        if (pick < 0) pick = any;
        if (pick >= 0) { cpu_taken[pick] = 1; gdrv[i] = cpus[pick]; }
    }
    /* worker CPUs: free CPUs taken round-robin over NUMA nodes, so any worker count spreads evenly */
    std::vector<int> wcpu_order;
    {
        std::vector<int> nodes;
        for (size_t j = 0; j < cpus.size(); ++j) if (!cpu_taken[j] && std::find(nodes.begin(), nodes.end(), cpu_node[j]) == nodes.end()) nodes.push_back(cpu_node[j]);
        std::sort(nodes.begin(), nodes.end());
        std::vector<std::vector<int> > per(nodes.size());
        for (size_t j = 0; j < cpus.size(); ++j)
            if (!cpu_taken[j]) per[std::find(nodes.begin(), nodes.end(), cpu_node[j]) - nodes.begin()].push_back((int)j);
        for (size_t k = 0;; ++k) {
            bool any = false;
            for (auto &v : per) if (k < v.size()) { wcpu_order.push_back(v[k]); any = true; }
            if (!any) break;
        }
    }
    int workers = cpu_workers > 0? cpu_workers : std::max(1, (int)wcpu_order.size() - (ndev == 0? 1 : 0));
    const bool pin = workers <= (int)wcpu_order.size() && !getenv("GALN_NO_PIN");
    std::vector<int> wcpu(workers, -1), wnode(workers, -1), whome(workers, 0);
    {
        std::vector<int> node_rank, nolocal(1, 0);
        for (int w = 0; w < workers; ++w) {
            if (pin) { wcpu[w] = cpus[wcpu_order[w]]; wnode[w] = cpu_node[wcpu_order[w]]; }
            if (ndev == 0) continue;
            std::vector<int> local;
            for (int i = 0; i < ndev; ++i) if (wnode[w] >= 0 && gnuma[i] == wnode[w]) local.push_back(i);
            if (!local.empty()) {
                if ((int)node_rank.size() <= wnode[w]) node_rank.resize(wnode[w] + 1, 0);
                whome[w] = local[node_rank[wnode[w]]++ % local.size()];
            } else whome[w] = nolocal[0]++ % ndev;
        }
    }

    FILE *out = stdout;
    if (outfn && !(out = fopen(outfn, "wb"))) { fprintf(stderr, "[galn] cannot write %s\n", outfn); return 1; }

    double t0 = now_s();
    /* ---- load: FASTQ on a second thread while the index loads ---- */
    char bwtfn[4096];
    snprintf(bwtfn, sizeof(bwtfn), "%s.bwt", prefix);
    int nreads = 0;
    read_t *reads = NULL;
    std::thread fq_thread([&] { reads = read_fastq(fq, &nreads); });
    bwtgpu_meta_t meta;
    uint32_t *bwt = load_bwt(bwtfn, &meta);
    fq_thread.join();
    if (!bwt) { if (reads) free_reads(reads, nreads); return 2; }
    if (!reads) { free(bwt); return 3; }
    uint32_t cnt_table[256];
    bwtgap_gen_cnt_table(cnt_table);
    size_t bwt_bytes = (size_t)meta.bwt_size * sizeof(uint32_t);
    /* interleaving hides memory latency (8 reads per worker by default); every slot may grow a
     * stack of up to max_entries entries, so cap the depth at half of the memory the job may use */
    {
        bool user = il_n > 0;
        if (!user) il_n = 8;
        il_n = std::max(1, std::min(il_n, ALN_IL_MAX));
        double slot_bytes = ((double)opt.max_entries + 16) * sizeof(bwtgap_entry_t);
        double avail = host_mem_avail();
        if (!user && avail > 0) {
            int cap = (int)(0.5 * avail / (slot_bytes * std::max(1, workers)));
            if (cap < il_n) il_n = std::max(1, cap);
        }
    }

    /* ---- one host copy of the index per NUMA node that has workers (CPU search is latency-bound) ---- */
    std::vector<int> rep_node;
    std::vector<uint32_t*> rep_ptr;
    if (pin) {
        for (int w = 0; w < workers; ++w)
            if (std::find(rep_node.begin(), rep_node.end(), wnode[w]) == rep_node.end()) rep_node.push_back(wnode[w]);
        if (rep_node.size() < 2) rep_node.clear();
        rep_ptr.assign(rep_node.size(), nullptr);
        std::vector<std::thread> rt;
        for (size_t k = 0; k < rep_node.size(); ++k) rt.emplace_back([&, k] {
            for (int w = 0; w < workers; ++w) if (wnode[w] == rep_node[k]) { pin_self(wcpu[w]); break; }
            void *mem = NULL;
            size_t align = (size_t)2 << 20, alloc = (bwt_bytes + align - 1) / align * align;
            if (posix_memalign(&mem, align, alloc ? alloc : align) != 0) return;
            madvise(mem, alloc, MADV_HUGEPAGE);
            memcpy(mem, bwt, bwt_bytes);        /* first touch from this node: pages land here */
            rep_ptr[k] = (uint32_t*)mem;
        });
        for (auto &t : rt) t.join();
        bool all = !rep_ptr.empty();
        for (auto p : rep_ptr) all = all && p;
        if (all) { free(bwt); bwt = rep_ptr[0]; }   /* the replicas replace the loaded copy */
        else for (auto &p : rep_ptr) { free(p); p = nullptr; }
        if (!all) rep_node.clear();
    }
    auto bwt_for_node = [&](int node) -> const uint32_t* {
        for (size_t k = 0; k < rep_node.size(); ++k) if (rep_node[k] == node && rep_ptr[k]) return rep_ptr[k];
        return bwt;
    };
    double t_loaded = now_s();

    /* ---- per-batch options (bwa_read_seq batches), per-read max_diff ---- */
    int *bstart = NULL;
    int nbatch = read_batch_ranges(reads, nreads, &bstart);
    std::vector<bwtgap_opt_t> bopts(nbatch > 0? nbatch : 1);
    std::vector<int> bstacks(nbatch > 0? nbatch : 1, 1), h_maxdiff(nreads > 0? nreads : 1), h_bid(nreads > 0? nreads : 1);
    int max_stacks = 1, max_len = 1;
    for (int b = 0; b < nbatch; ++b) {
        int base = bstart[b], n = bstart[b + 1] - bstart[b];
        bstacks[b] = aln_batch_opt(&opt, reads + base, n, &bopts[b]);
        max_stacks = std::max(max_stacks, bstacks[b]);
        for (int t = 0; t < n; ++t) {
            h_maxdiff[base + t] = aln_read_maxdiff(&opt, &bopts[b], reads[base + t].len);
            h_bid[base + t] = b;
            max_len = std::max(max_len, reads[base + t].len);
        }
    }
    free(bstart);

    Results res;
    res.n.assign(nreads, INT_MIN);
    res.aln.resize(nreads);
    res.done.reset(new std::atomic<uint8_t>[nreads > 0? nreads : 1]);
    for (int r = 0; r < nreads; ++r) res.done[r].store(0);
    Shared S;
    for (int i = 0; i < ndev; ++i) {
        S.devs.emplace_back(new DevState);
        DevState &D = *S.devs.back();
        D.dev = devlist[i]; D.numa = gnuma[i]; D.drv_cpu = gdrv[i];
        D.host_bwt = bwt_for_node(gnuma[i]);
    }
    for (int r = 0; r < nreads; ++r) if (ndev > 0) S.devs[r % ndev]->reads.push_back(r);   /* interleaved shares */

    std::vector<unsigned long long> ovf_pops(nreads > 0? nreads : 1, 0);
    std::vector<uint8_t> queued(nreads > 0? nreads : 1, 0);   /* read handed to a CPU queue (under S.m) */

    /* ---- CPU workers: one global pool; each prefers the queues of the GPU next to it ---- */
    std::vector<long> wreads(workers, 0), wdup(workers, 0);
    std::vector<double> wbusy(workers, 0), wpops(workers, 0);
    const bool il_check = getenv("GALN_IL_CHECK") && atoi(getenv("GALN_IL_CHECK")) > 0;
    std::atomic<long> il_bad(0), il_checked(0);
    /* take the next read for a CPU worker: r >= 0, -1 if none right now (only when !block), -2 if all is done */
    auto take = [&](int home, bool block) -> int {   /* block: the worker has no read in flight */
        int r = -1;
        std::unique_lock<std::mutex> lk(S.m);
        for (;;) {
                    {
                        if (S.workers_may_start && !S.cpu_only.empty()) { r = S.cpu_only.front(); S.cpu_only.pop_front(); break; }
                        const int nd = (int)S.devs.size();
                        int p;
                        /* in order of preference over all GPUs, starting with the home GPU */
                        for (int k = 0; k < nd && r < 0; ++k) { DevState &D = *S.devs[(home + k) % nd];
                            if (!D.cpu_help) continue;
                            if (!D.giant.empty()) { r = D.giant.front(); D.giant.pop_front(); } }
                        for (int k = 0; k < nd && r < 0; ++k) { DevState &D = *S.devs[(home + k) % nd];
                            if (!D.cpu_help) continue;
                            if (D.G.active && (p = D.G.claim()) >= 0) r = D.G.list[p]; }
                        for (int k = 0; k < nd && r < 0; ++k) { DevState &D = *S.devs[(home + k) % nd];
                            if (!D.cpu_help) continue;
                            if (D.W.active && (p = D.W.claim()) >= 0) r = D.W.list[p]; }
                        for (int k = 0; k < nd && r < 0; ++k) { DevState &D = *S.devs[(home + k) % nd];
                            if (!D.cpu_help) continue;
                            if (!D.heavy.empty()) { r = D.heavy.front(); D.heavy.pop_front(); } }
                        if (r >= 0) break;
                        bool retry = false;
                        for (int k = 0; k < nd && r < 0 && !retry && block; ++k) { DevState &D = *S.devs[(home + k) % nd];
                            if (!D.cpu_help) continue;
                            if (D.running && D.gpu_running && !D.running->claimable() && (p = D.running->steal()) >= 0) {
                                r = D.running->list[p];
                                if (res.done[r].load() || queued[r]) { r = -1; retry = true; }
                                else queued[r] = 1;   /* CPU-owned: a later GPU give-up of it is dropped */
                            } }
                        if (r >= 0) break;
                        if (retry) continue;
                        if (S.finished()) { r = -2; break; }   /* nothing left, ever */
                    }
                    if (!block) { r = -1; break; }
                    if (S.any_running()) S.cv.wait_for(lk, std::chrono::milliseconds(2));
                    else S.cv.wait(lk);
        }
        return r;
    };
    auto worker = [&](int wid) {
        pin_self(wcpu[wid]);
        const uint32_t *wbwt = wnode[wid] >= 0? bwt_for_node(wnode[wid]) : bwt;
        const int home = whome[wid];
        std::unique_ptr<aln_il_t> il(new aln_il_t);
        aln_il_init(il.get(), wbwt, &meta, cnt_table, il_n, opt.max_entries + 16, max_stacks, max_len, opt.seed_len);
        /* test aid: GALN_IL_CHECK=1 re-runs every read with the plain search and compares */
        std::unique_ptr<aln_ws_t> chk;
        if (il_check) { chk.reset(new aln_ws_t); aln_ws_init(chk.get()); }
        int active = 0;
        bool all_done = false;
        double ts = 0;
        long refill_mask = 31;
        for (long round = 0;; ++round) {
            /* every 32 rounds: drop slots whose read another engine has finished */
            if (active > 0 && (round & 31) == 0)
                for (int s = 0; s < il->n; ++s) {
                    aln_il_slot_t &sl = il->slot[s];
                    if ((sl.status == ALN_IL_RUN || sl.status == ALN_IL_OCC) && res.done[sl.tag].load()) {
                        wpops[wid] += (double)sl.st.n_pops; ++wdup[wid];
                        sl.status = ALN_IL_IDLE;
                        if (--active == 0) wbusy[wid] += now_s() - ts;
                    }
                }
            /* refill idle slots; while others run, poll the shared lock rarely and back off when nothing comes */
            if (!all_done && active < il->n && (active == 0 || (round & refill_mask) == 0)) {
                bool got_any = false;
                for (int s = 0; s < il->n && !all_done; ++s) {
                    if (il->slot[s].status != ALN_IL_IDLE) continue;
                    int r;
                    for (;;) {
                        r = take(home, active == 0);
                        if (r < 0 || !res.done[r].load()) break;   /* skip reads a GPU finished already */
                    }
                    if (r == -2) { all_done = true; break; }
                    if (r < 0) break;
                    if (active == 0) ts = now_s();
                    int b = h_bid[r];
                    aln_il_start(il.get(), s, &opt, &bopts[b], bstacks[b], &reads[r], r);
                    ++active;
                    got_any = true;
                }
                refill_mask = got_any? 31 : std::min<long>(refill_mask * 2 + 1, 4095);
            }
            if (active == 0) { if (all_done) break; continue; }
            aln_il_round(il.get());
            for (int s = 0; s < il->n; ++s) {
                aln_il_slot_t &sl = il->slot[s];
                if (sl.status != ALN_IL_DONE) continue;
                int r = sl.tag, rr = sl.ret;
                if (chk) {
                    int b = h_bid[r];
                    int cr = aln_search_host(wbwt, &meta, cnt_table, &opt, &bopts[b], bstacks[b], &reads[r], chk.get());
                    bool ok = cr == rr && (cr <= 0 || !memcmp(chk->aln, sl.ws.aln, (size_t)cr * sizeof(bwtgap_aln_t)))
                              && chk->last_pops == sl.st.n_pops && chk->last_trace == sl.st.trace;
                    if (!ok && il_bad++ < 10)
                        fprintf(stderr, "[galn il-check] read %d (%s): interleaved n=%d pops=%llu | plain n=%d pops=%llu\n", r,
                                reads[r].name, rr, (unsigned long long)sl.st.n_pops, cr, (unsigned long long)chk->last_pops);
                    ++il_checked;
                }
                if (res.claim(r)) {
                    if (rr > 0) res.aln[r].assign(sl.ws.aln, sl.ws.aln + rr);
                    res.n[r] = rr;
                    ++wreads[wid];
                } else ++wdup[wid];
                wpops[wid] += (double)sl.st.n_pops;
                sl.status = ALN_IL_IDLE;
                if (--active == 0) wbusy[wid] += now_s() - ts;
            }
        }
        aln_il_free(il.get());
        if (chk) aln_ws_free(chk.get());
    };

    if (ndev > 0) {
        uint64_t tq = 0;
        for (int i = 0; i < nreads; ++i) tq += reads[i].len;
        for (auto &d : S.devs) {
            size_t free_b = 0, total_b = 0;
            CUDA_CHECK(cudaSetDevice(d->dev));
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            if (bwt_bytes + tq + (size_t)nreads * 32 + ((size_t)1 << 30) > free_b) {
                fprintf(stderr, "[galn] GPU %d: the index (%.2f GB) and reads do not fit in GPU memory (%.2f GB free); use --cpu-only\n",
                        d->dev, bwt_bytes / 1e9, free_b / 1e9);
                return 10;
            }
        }
    }

    std::vector<std::thread> pool;
    std::vector<char> by_warp;
    std::vector<unsigned long long> warp_pops, warp_trace;
    if (verify) { by_warp.assign(nreads, 0); warp_pops.assign(nreads, 0); warp_trace.assign(nreads, 0); }

    /* host arrays every GPU uploads (all reads, indexed by read number) */
    uint64_t total = 0;
    std::vector<uint64_t> h_off(nreads > 0? nreads : 1);
    std::vector<int> h_len(nreads > 0? nreads : 1);
    std::vector<uint8_t> h_q;
    if (ndev > 0) {
        for (int i = 0; i < nreads; ++i) { h_off[i] = total; h_len[i] = reads[i].len; total += reads[i].len; }
        h_q.resize(total? total : 1);
        for (int i = 0; i < nreads; ++i) memcpy(&h_q[h_off[i]], reads[i].query, reads[i].len);
    }
    bool any_long = false;
    for (int i = 0; i < nreads; ++i) if (reads[i].len > opt.seed_len) { any_long = true; break; }
    /* a read the warp engine cannot take goes straight to the CPU */
    auto warp_ok = [&](int r) { return reads[r].len <= GAPW_MAXLEN && bstacks[h_bid[r]] <= GAPW_MAXSTACK; };

    /* ---- one GPU: upload, light tier, phases W and G over the reads it owns ---- */
    auto run_device = [&](DevState &D) {
        pin_self(D.drv_cpu);
        CUDA_CHECK(cudaSetDevice(D.dev));
        const int device = D.dev;
        const int nmine = (int)D.reads.size();
        cudaStream_t ks, cs;   /* kernels, and progress copies that overlap them */
        CUDA_CHECK(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        uint32_t *d_bwt, *d_cnt; uint8_t *d_q; uint64_t *d_off; int *d_len, *d_maxdiff, *d_bid, *d_nstacks;
        bwtgap_opt_t *d_bopts;
        CUDA_CHECK(cudaMalloc(&d_bwt, bwt_bytes + GAPW_BWT_PAD));   /* zero pad: the warp engine reads whole blocks */
        CUDA_CHECK(cudaMemset((char*)d_bwt + bwt_bytes, 0, GAPW_BWT_PAD));
        CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(cnt_table)));
        CUDA_CHECK(cudaMalloc(&d_q, total? total : 1));
        CUDA_CHECK(cudaMalloc(&d_off, (size_t)(nreads + 1) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_len, (size_t)(nreads + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_maxdiff, (size_t)(nreads + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bid, (size_t)(nreads + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bopts, bopts.size() * sizeof(bwtgap_opt_t)));
        CUDA_CHECK(cudaMalloc(&d_nstacks, bstacks.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bwt, D.host_bwt, bwt_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cnt, cnt_table, sizeof(cnt_table), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), total? total : 1, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_off, h_off.data(), (size_t)nreads * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_len, h_len.data(), (size_t)nreads * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_maxdiff, h_maxdiff.data(), (size_t)nreads * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bid, h_bid.data(), (size_t)nreads * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bopts, bopts.data(), bopts.size() * sizeof(bwtgap_opt_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nstacks, bstacks.data(), bstacks.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(g_meta, &meta, sizeof(bwtgpu_meta_t)));
        auto pool_bytes_avail = [&](size_t reserve_other) {
            size_t fb = 0, tb = 0;
            CUDA_CHECK(cudaMemGetInfo(&fb, &tb));
            size_t keep = ((size_t)1 << 30) + reserve_other;
            size_t avail = fb > keep? fb - keep : 0;
            if (pool_gb > 0) avail = std::min(avail, (size_t)(pool_gb * 1e9));
            return avail;
        };

        /* ================= light tier ================= */
        if (nmine > 0) {
            double ts = now_s();
            int maxl = std::min(nmine, 262144);
            uint64_t wcap = std::min<uint64_t>(total + nreads, (uint64_t)maxl * (max_len + 1)) + 1;
            int acap = 32;
            int *d_idx, *d_heads, *d_naln; uint64_t *d_woff; bwtgap_width_t *d_w, *d_sw; bwtgap_aln_t *d_aln;
            unsigned long long *d_pops; bwtgap_entry_t *d_pool;
            CUDA_CHECK(cudaMalloc(&d_idx, (size_t)maxl * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_woff, (size_t)maxl * sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&d_naln, (size_t)maxl * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_pops, (size_t)maxl * sizeof(unsigned long long)));
            CUDA_CHECK(cudaMalloc(&d_heads, (size_t)maxl * max_stacks * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_aln, (size_t)maxl * acap * sizeof(bwtgap_aln_t)));
            CUDA_CHECK(cudaMalloc(&d_w, wcap * sizeof(bwtgap_width_t)));
            CUDA_CHECK(cudaMalloc(&d_sw, (any_long? wcap : 1) * sizeof(bwtgap_width_t)));
            size_t pentries = pool_bytes_avail(0) / sizeof(bwtgap_entry_t);
            long pl = std::min<long>(maxl, (long)(pentries / (size_t)light_cap));
            if (pl < 1) galn_die("[galn] GPU %d: not enough GPU memory for the light tier\n", device);
            CUDA_CHECK(cudaMalloc(&d_pool, (size_t)pl * light_cap * sizeof(bwtgap_entry_t)));
            std::vector<int> h_idx(pl), r_naln(pl);
            std::vector<uint64_t> h_woff(pl);
            std::vector<unsigned long long> r_pops(pl);
            std::vector<bwtgap_aln_t> r_aln((size_t)pl * acap);
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            for (long c0 = 0; c0 < nmine; c0 += pl) {
                int m = (int)std::min<long>(pl, nmine - c0);
                uint64_t wl = 0;
                for (int j = 0; j < m; ++j) { int r = D.reads[c0 + j]; h_idx[j] = r; h_woff[j] = wl; wl += reads[r].len + 1; }
                CUDA_CHECK(cudaMemcpyAsync(d_idx, h_idx.data(), (size_t)m * sizeof(int), cudaMemcpyHostToDevice, ks));
                CUDA_CHECK(cudaMemcpyAsync(d_woff, h_woff.data(), (size_t)m * sizeof(uint64_t), cudaMemcpyHostToDevice, ks));
                light_args a;
                a.bwt = d_bwt; a.cnt_table = d_cnt; a.query = d_q; a.qoff = d_off; a.qlen = d_len; a.maxdiff = d_maxdiff;
                a.bid = d_bid; a.idx = d_idx; a.woff = d_woff; a.wbuf = d_w; a.swbuf = d_sw; a.pool = d_pool; a.heads = d_heads;
                a.aln = d_aln; a.n_aln = d_naln; a.n_pops = d_pops; a.bopts = d_bopts; a.nstacks = d_nstacks;
                a.heads_stride = max_stacks; a.n = m; a.pool_cap = (int)light_cap; a.aln_cap = acap; a.seed_len = opt.seed_len;
                CUDA_CHECK(cudaEventRecord(e0, ks));
                light_kernel<<<(m + 127) / 128, 128, 0, ks>>>(a);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(e1, ks));
                CUDA_CHECK(cudaMemcpyAsync(r_naln.data(), d_naln, (size_t)m * sizeof(int), cudaMemcpyDeviceToHost, ks));
                CUDA_CHECK(cudaMemcpyAsync(r_pops.data(), d_pops, (size_t)m * sizeof(unsigned long long), cudaMemcpyDeviceToHost, ks));
                CUDA_CHECK(cudaMemcpyAsync(r_aln.data(), d_aln, (size_t)m * acap * sizeof(bwtgap_aln_t), cudaMemcpyDeviceToHost, ks));
                CUDA_CHECK(cudaStreamSynchronize(ks));
                float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); D.k_light += ms / 1e3;
                std::vector<int> ovf, cpu;
                for (int j = 0; j < m; ++j) {
                    int r = h_idx[j], na = r_naln[j];
                    if (na == BWTGAP_ERR_STACK || na == BWTGAP_ERR_ALN) {
                        ovf_pops[r] = r_pops[j];
                        (warp_ok(r)? ovf : cpu).push_back(r);
                    } else if (res.claim(r)) {
                        if (na > 0) res.aln[r].assign(&r_aln[(size_t)j * acap], &r_aln[(size_t)j * acap] + na);
                        res.n[r] = na;
                        ++D.n_light_done;
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(S.m);
                    D.heavy.insert(D.heavy.end(), ovf.begin(), ovf.end());
                    S.cpu_only.insert(S.cpu_only.end(), cpu.begin(), cpu.end());
                    D.n_unsup += (long)cpu.size();
                }
                S.cv.notify_all();
            }
            cudaFree(d_idx); cudaFree(d_woff); cudaFree(d_naln); cudaFree(d_pops); cudaFree(d_heads); cudaFree(d_aln);
            cudaFree(d_w); cudaFree(d_sw); cudaFree(d_pool);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            D.t_light = now_s() - ts;
        }

        /* ================= phases W and G: warp-per-read engine ================= */
        int dev_sm = 0, bps = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&dev_sm, cudaDevAttrMultiProcessorCount, device));
        size_t smem = 4 * sizeof(gapw_smem);
        /* --verify needs the trace hash: a separate instantiation (never use its timings as performance) */
        auto gapw_k = verify? gapw_kernel<true> : gapw_kernel<false>;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, gapw_k, 128, smem));
        if (bps < 1) bps = 1;
        long occ_warps = (long)bps * dev_sm * 4;
        const int aln_cap = 4096;
        const long max_read_chunks = (opt.max_entries + 9) / GAPW_CHUNK + 1 + max_stacks + 64;
        const long min_arena_chunks = 4096;   /* 262,144 entries */
        size_t per_chunk = GAPW_CHUNK * sizeof(gapw_entry) + 2 * sizeof(int);
        /* run one warp-engine phase over list `L` (shared with the CPU through `te`) */
        auto run_phase = [&](TwoEnd &te, bool big, long &n_gpu, double &pops, long &nw_used, double &ksec) {
            int n = (int)te.list.size();
            if (n == 0) return;
            long out_cap = std::max<long>((long)n * 8, 1L << 20);
            int *d_list, *d_back, *d_front, *d_out_n, *d_out_off, *d_chunk_prev, *d_gfree, *d_fail, *d_fail_n, *d_owned;
            unsigned long long *d_out_pops, *d_out_trace;
            bwtgap_aln_t *d_scratch = nullptr, *d_out_rec; gapw_entry *d_chunks = nullptr; gapw_ctl *d_ctl;
            CUDA_CHECK(cudaMalloc(&d_list, (size_t)n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_back, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_front, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_out_n, (size_t)n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_out_off, (size_t)n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_out_pops, (size_t)n * sizeof(unsigned long long)));
            CUDA_CHECK(cudaMalloc(&d_out_trace, (size_t)n * sizeof(unsigned long long)));
            CUDA_CHECK(cudaMalloc(&d_out_rec, (size_t)out_cap * sizeof(bwtgap_aln_t)));
            CUDA_CHECK(cudaMalloc(&d_fail, (size_t)n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_fail_n, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_ctl, sizeof(gapw_ctl)));
            CUDA_CHECK(cudaMalloc(&d_owned, 4 * sizeof(int)));   /* unused in arena mode */
            uint8_t *d_cancel;
            int *d_handoff;
            CUDA_CHECK(cudaMalloc(&d_handoff, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_handoff, 0, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cancel, (size_t)n));
            CUDA_CHECK(cudaMemset(d_cancel, 0, (size_t)n));
            /* size the warp count and the stack memory from what is left. Arenas (a fixed slice
             * per warp, best locality) when every warp can get room for most reads; otherwise
             * one shared pool, so memory goes only to the reads that need it. */
            size_t avail = pool_bytes_avail(0);
            size_t scratch_per_warp = (size_t)aln_cap * sizeof(bwtgap_aln_t);
            long nw = std::min<long>(occ_warps, ((long)n + 3) / 4 * 4);
            if (max_warps > 0) nw = std::min<long>(nw, max_warps);
            const long owned_cap = max_read_chunks + 64;
            bool shared_pool = false;
            if (!big) {
                long arena_if_all = (long)(avail / ((size_t)std::max<long>(nw, 4))) / (long)per_chunk;
                shared_pool = alloc_mode == 2;   /* measured: arenas beat one shared pool even on small GPUs */
                (void)arena_if_all;
            }
            size_t per_warp_extra = scratch_per_warp + (shared_pool? (size_t)owned_cap * sizeof(int) : 0);
            if (!shared_pool) {
                /* phase W: at least ~0.75 M entries per warp (fewer, larger arenas on small GPUs) */
                long need = big? max_read_chunks : std::max<long>(min_arena_chunks, 3 * max_read_chunks / 8);
                nw = std::min<long>(nw, (long)(avail / (need * per_chunk + per_warp_extra)));
            } else {
                nw = std::min<long>(nw, (long)(avail / (min_arena_chunks * per_chunk / 16 + per_warp_extra)));
            }
            nw = nw / 4 * 4;
            long n_chunks = nw > 0? (long)((avail - (size_t)nw * per_warp_extra) / (per_chunk + (shared_pool? sizeof(int) : 0))) : 0;
            if (n_chunks > INT_MAX - 64) n_chunks = INT_MAX - 64;
            if (nw < 4) {   /* GPU too small for this phase: leave it to the CPU */
                if (verbose) fprintf(stderr, "[galn] GPU %d phase %s: not enough GPU memory, %d reads left to the CPU\n", device, big? "G" : "W", n);
                cudaFree(d_list); cudaFree(d_back); cudaFree(d_front); cudaFree(d_out_n); cudaFree(d_out_off);
                cudaFree(d_out_pops); cudaFree(d_out_trace); cudaFree(d_out_rec); cudaFree(d_fail); cudaFree(d_fail_n);
                cudaFree(d_ctl); cudaFree(d_owned); cudaFree(d_cancel);
                std::lock_guard<std::mutex> lk(S.m);
                te.limit.store(n);
                D.cpu_help = true;      /* --gpu-only: the CPU takes what this GPU cannot */
                S.cv.notify_all();
                return;
            }
            nw_used = nw;
            long grid = nw / 4;
            size_t scratch = (size_t)nw * scratch_per_warp;
            CUDA_CHECK(cudaMalloc(&d_scratch, scratch));
            CUDA_CHECK(cudaMalloc(&d_chunks, (size_t)n_chunks * GAPW_CHUNK * sizeof(gapw_entry)));
            CUDA_CHECK(cudaMalloc(&d_chunk_prev, (size_t)n_chunks * sizeof(int)));
            if (shared_pool) {
                cudaFree(d_owned);
                CUDA_CHECK(cudaMalloc(&d_owned, (size_t)nw * owned_cap * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_gfree, (size_t)n_chunks * sizeof(int)));
            } else d_gfree = d_chunk_prev;   /* not used in arena mode */
            CUDA_CHECK(cudaMemcpy(d_list, te.list.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_back, &n, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_front, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_out_n, 0x80, (size_t)n * sizeof(int)));   /* sentinel: not done by the GPU */
            CUDA_CHECK(cudaMemset(d_out_pops, 0, (size_t)n * sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemset(d_fail, 0xff, (size_t)n * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_fail_n, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_ctl, 0, sizeof(gapw_ctl)));
            gapw_args wa;
            memset(&wa, 0, sizeof(wa));
            wa.bwt = d_bwt; wa.query = d_q; wa.qoff = d_off; wa.qlen = d_len; wa.maxdiff = d_maxdiff;
            wa.bopts = d_bopts; wa.nstacks = d_nstacks; wa.bid = d_bid; wa.seed_len = opt.seed_len;
            wa.list = d_list; wa.list_n = n; wa.qctl = nullptr; wa.dback = d_back; wa.dfront = d_front;
            wa.fail = d_fail; wa.fail_n = (unsigned int*)d_fail_n;
            wa.cancel = d_cancel;
            wa.handoff = big || handoff_entries <= 0? nullptr : d_handoff;
            wa.handoff_entries = (int)std::min<long>(handoff_entries, INT_MAX);
            wa.chunks = d_chunks; wa.chunk_prev = d_chunk_prev; wa.gfree = d_gfree; wa.n_chunks = (int)n_chunks;
            wa.owned = d_owned; wa.owned_cap = shared_pool? (int)owned_cap : 0; wa.ctl = d_ctl;
            wa.arena_chunks = shared_pool? 0 : (int)(n_chunks / nw);
            wa.aln_scratch = d_scratch; wa.aln_cap = aln_cap;
            wa.out_n = d_out_n; wa.out_off = d_out_off; wa.out_pops = d_out_pops; wa.out_trace = d_out_trace;
            wa.out_rec = d_out_rec; wa.out_cap = (int)std::min<long>(out_cap, INT_MAX);
            if (verbose) {
                if (shared_pool)
                    fprintf(stderr, "[galn] GPU %d phase %s: %d reads, %ld warps, shared stack pool %.1f GB (%.0f M entries)\n",
                            device, big? "G" : "W", n, nw, n_chunks * (double)per_chunk / 1e9, n_chunks * (double)GAPW_CHUNK / 1e6);
                else
                    fprintf(stderr, "[galn] GPU %d phase %s: %d reads, %ld warps, arena %.1f M entries/warp, stack memory %.1f GB\n",
                            device, big? "G" : "W", n, nw, wa.arena_chunks * (double)GAPW_CHUNK / 1e6, n_chunks * (double)per_chunk / 1e9);
            }
            /* pinned host words for the progress exchange */
            int *h_front, *h_back, *h_failn, *h_handoff;
            CUDA_CHECK(cudaHostAlloc(&h_handoff, sizeof(int), cudaHostAllocDefault));
            *h_handoff = 0;
            CUDA_CHECK(cudaHostAlloc(&h_front, sizeof(int), cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&h_back, sizeof(int), cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&h_failn, sizeof(int), cudaHostAllocDefault));
            int *h_fail;
            CUDA_CHECK(cudaHostAlloc(&h_fail, (size_t)n * sizeof(int), cudaHostAllocDefault));
            uint8_t *h_cancel;
            CUDA_CHECK(cudaHostAlloc(&h_cancel, (size_t)n, cudaHostAllocDefault));
            memset(h_cancel, 0, (size_t)n);
            int fail_seen = 0;
            /* move give-ups the kernel has published to the CPU queues */
            auto drain_fails = [&](int upto) {
                if (upto > fail_seen) {
                    CUDA_CHECK(cudaMemcpyAsync(h_fail + fail_seen, d_fail + fail_seen, (size_t)(upto - fail_seen) * sizeof(int),
                                               cudaMemcpyDeviceToHost, cs));
                    CUDA_CHECK(cudaStreamSynchronize(cs));
                }
                std::vector<int> got;
                while (fail_seen < upto && h_fail[fail_seen] >= 0) got.push_back(h_fail[fail_seen++]);
                if (!got.empty()) {
                    std::lock_guard<std::mutex> lk(S.m);
                    for (int x : got) {
                        int r = x & ~GAPW_FAIL_CPU;
                        if (queued[r] || res.done[r].load()) continue;   /* a CPU worker owns it already, or it is finished */
                        queued[r] = 1;
                        if (big || (x & GAPW_FAIL_CPU)) S.cpu_only.push_back(r);
                        else { D.giant.push_back(r); ++D.n_giant; }
                    }
                }
                return !got.empty();
            };
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
            { std::lock_guard<std::mutex> lk(S.m); D.gpu_running = true; D.running = &te; }
            CUDA_CHECK(cudaEventRecord(e0, ks));
            gapw_k<<<(unsigned)grid, 128, smem, ks>>>(wa);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(e1, ks));
            /* progress exchange while the kernel runs */
            for (long poll = 0;; ++poll) {
                cudaError_t q = cudaEventQuery(e1);
                if (q == cudaSuccess) break;
                if (q != cudaErrorNotReady) CUDA_CHECK(q);
                *h_front = te.front.load();
                CUDA_CHECK(cudaMemcpyAsync(d_front, h_front, sizeof(int), cudaMemcpyHostToDevice, cs));
                CUDA_CHECK(cudaMemcpyAsync(h_back, d_back, sizeof(int), cudaMemcpyDeviceToHost, cs));
                CUDA_CHECK(cudaMemcpyAsync(h_failn, d_fail_n, sizeof(int), cudaMemcpyDeviceToHost, cs));
                CUDA_CHECK(cudaStreamSynchronize(cs));
                if (D.cpu_help) te.limit.store(std::max(*h_back, 0));
                if (!big && handoff_entries > 0) {   /* hand heavy reads to the CPU only while its queues are short */
                    long pending = 0;
                    { std::lock_guard<std::mutex> lk(S.m); for (auto &dp : S.devs) pending += (long)dp->giant.size(); pending += (long)S.cpu_only.size(); }
                    long nd = (long)S.devs.size();
                    int want = D.cpu_help && pending < workers? (int)std::max<long>(1, (workers - pending) / nd) : 0;
                    if (handoff_endgame && te.front.load() < std::max(*h_back, 0)) want = 0;   /* only once the list is used up */
                    if (want != 0 || *h_handoff != 0) {   /* a fresh budget for this poll interval */
                        *h_handoff = want;
                        CUDA_CHECK(cudaMemcpyAsync(d_handoff, h_handoff, sizeof(int), cudaMemcpyHostToDevice, cs));
                        CUDA_CHECK(cudaStreamSynchronize(cs));
                    }
                }
                bool newfail = drain_fails(std::min(*h_failn, n));
                if (poll % 16 == 15) {   /* about every 10 ms: tell the warps which of their reads are finished */
                    int lo = std::min(std::max(*h_back, 0), n);
                    bool any = false;
                    for (int p = lo; p < n; ++p) {
                        uint8_t d = res.done[te.list[p]].load()? 1 : 0;
                        if (d != h_cancel[p]) { h_cancel[p] = d; any = true; }
                    }
                    if (any) CUDA_CHECK(cudaMemcpyAsync(d_cancel + lo, h_cancel + lo, (size_t)(n - lo), cudaMemcpyHostToDevice, cs));
                }
                if (newfail || te.claimable()) S.cv.notify_all();
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); ksec += ms / 1e3;
            /* collect */
            CUDA_CHECK(cudaMemcpy(h_failn, d_fail_n, sizeof(int), cudaMemcpyDeviceToHost));
            drain_fails(std::min(*h_failn, n));
            std::vector<int> o_n(n), o_off(n);
            std::vector<unsigned long long> o_pops(n), o_trace(n);
            CUDA_CHECK(cudaMemcpy(o_n.data(), d_out_n, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(o_off.data(), d_out_off, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(o_pops.data(), d_out_pops, (size_t)n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(o_trace.data(), d_out_trace, (size_t)n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            gapw_ctl hctl;
            CUDA_CHECK(cudaMemcpy(&hctl, d_ctl, sizeof(hctl), cudaMemcpyDeviceToHost));
            long used = (long)std::min<unsigned long long>(hctl.out_top, (unsigned long long)out_cap);
            std::vector<bwtgap_aln_t> o_rec(used > 0? used : 1);
            if (used > 0) CUDA_CHECK(cudaMemcpy(o_rec.data(), d_out_rec, (size_t)used * sizeof(bwtgap_aln_t), cudaMemcpyDeviceToHost));
            const int not_done = (int)0x80808080;
            for (int p = 0; p < n; ++p) {
                int r = te.list[p], na = o_n[p];
                if (na != not_done) D.epops += (double)o_pops[p];   /* every pop the GPU executed, also of given-up/cancelled reads */
                if (na == not_done || na < 0) continue;   /* CPU's, or given up (already queued) */
                pops += (double)o_pops[p];
                if (res.claim(r)) {
                    if (na > 0) res.aln[r].assign(o_rec.begin() + o_off[p], o_rec.begin() + o_off[p] + na);
                    res.n[r] = na;
                    ++n_gpu;
                    if (verify) { by_warp[r] = 1; warp_pops[r] = o_pops[p]; warp_trace[r] = o_trace[p]; }
                }
            }
            {
                std::lock_guard<std::mutex> lk(S.m);
                D.gpu_running = false;
                D.running = nullptr;
                /* every position the CPU did not claim was taken by the GPU: finished, cancelled because
                 * finished elsewhere, or given up and already queued. Retire the list instead of
                 * reopening it, so no worker re-runs a queued give-up; anything else goes to the CPU. */
                for (int p = te.front.load(); p < n; ++p) {
                    int r = te.list[p];
                    if (!res.done[r].load() && !queued[r]) { queued[r] = 1; S.cpu_only.push_back(r); }
                }
                te.front.store(n);
                te.limit.store(n);
            }
            S.cv.notify_all();
            cudaFreeHost(h_handoff); cudaFree(d_handoff);
            cudaFreeHost(h_front); cudaFreeHost(h_back); cudaFreeHost(h_failn); cudaFreeHost(h_fail); cudaFreeHost(h_cancel);
            cudaFree(d_cancel);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            cudaFree(d_list); cudaFree(d_back); cudaFree(d_front); cudaFree(d_out_n); cudaFree(d_out_off);
            cudaFree(d_out_pops); cudaFree(d_out_trace); cudaFree(d_out_rec); cudaFree(d_fail); cudaFree(d_fail_n);
            cudaFree(d_ctl); cudaFree(d_owned); cudaFree(d_scratch); cudaFree(d_chunks); cudaFree(d_chunk_prev);
            if (d_gfree != d_chunk_prev) cudaFree(d_gfree);
        };


        /* phase W: all light-tier overflow; the CPU end gets the reads whose stack grew fastest */
        {
            double ts = now_s();
            std::lock_guard<std::mutex> lk(S.m);
            D.W.list.assign(D.heavy.begin(), D.heavy.end());
            D.heavy.clear();
            std::stable_sort(D.W.list.begin(), D.W.list.end(), [&](int x, int y) { return ovf_pops[x] < ovf_pops[y]; });
            D.W.front.store(0);
            D.W.limit.store(gpu_only? 0 : (int)D.W.list.size());
            D.W.active = true;
            D.t_W = -ts;
        }
        S.cv.notify_all();
        run_phase(D.W, false, D.n_W_gpu, D.pops_W, D.nwW, D.k_W);
        D.t_W += now_s();
        /* phase G: giant reads not yet taken by the CPU */
        {
            double ts = now_s();
            std::lock_guard<std::mutex> lk(S.m);
            D.G.list.assign(D.giant.begin(), D.giant.end());
            D.giant.clear();
            for (int r : D.G.list) queued[r] = 0;   /* owned by the phase-G list now (end-game stealing applies) */
            D.G.front.store(0);
            D.G.limit.store(gpu_only? 0 : (int)D.G.list.size());
            D.G.active = true;
            D.t_G = -ts;
        }
        S.cv.notify_all();
        run_phase(D.G, true, D.n_G_gpu, D.pops_G, D.nwG, D.k_G);
        D.t_G += now_s();
        {
            std::lock_guard<std::mutex> lk(S.m);
            D.gpu_done = true;
            D.cpu_help = true;
            D.W.limit.store((int)D.W.list.size());
            D.G.limit.store((int)D.G.list.size());
            if (++S.ndone == (int)S.devs.size()) S.workers_may_start = true;   /* --gpu-only: the CPU takes what is left */
        }
        S.cv.notify_all();
        cudaFree(d_bwt); cudaFree(d_cnt); cudaFree(d_q); cudaFree(d_off); cudaFree(d_len); cudaFree(d_maxdiff);
        cudaFree(d_bid); cudaFree(d_bopts); cudaFree(d_nstacks);
        cudaStreamDestroy(ks); cudaStreamDestroy(cs);
    };

    if (cpu_only) {
        for (int r = 0; r < nreads; ++r) S.cpu_only.push_back(r);
        for (int w = 0; w < workers; ++w) pool.emplace_back(worker, w);
    } else {
        if (gpu_only) { S.workers_may_start = false; for (auto &d : S.devs) d->cpu_help = false; }
        for (int w = 0; w < workers; ++w) pool.emplace_back(worker, w);
        /* the index is uploaded from pinned host memory (one registration per distinct host copy) */
        std::vector<const uint32_t*> regd;
        for (auto &d : S.devs)
            if (std::find(regd.begin(), regd.end(), d->host_bwt) == regd.end()) {
                CUDA_CHECK(cudaSetDevice(d->dev));
                if (cudaHostRegister((void*)d->host_bwt, bwt_bytes, cudaHostRegisterPortable) == cudaSuccess) regd.push_back(d->host_bwt);
                else cudaGetLastError();
            }
        std::vector<std::thread> drivers;
        for (auto &d : S.devs) drivers.emplace_back(run_device, std::ref(*d));
        for (auto &t : drivers) t.join();
        for (auto p : regd) cudaHostUnregister((void*)p);
    }
    for (auto &th : pool) th.join();
    pool.clear();
    double t_search = now_s();
    if (il_check)
        fprintf(stderr, "[galn il-check] %ld CPU reads re-run with the plain search: %ld differ -> %s\n",
                (long)il_checked, (long)il_bad, il_bad? "FAIL" : "PASS");

    /* ---- optional verification of every warp-engine read against the CPU ---- */
    int verify_fail = 0;
    if (verify && !by_warp.empty()) {
        std::vector<int> vr;
        for (int r = 0; r < nreads; ++r) if (by_warp[r]) vr.push_back(r);
        std::atomic<long> bad(0);
        std::atomic<size_t> next(0);
        std::vector<std::thread> vt;
        for (int w = 0; w < workers; ++w) vt.emplace_back([&] {
            aln_ws_t ws;
            aln_ws_init(&ws);
            aln_ws_prealloc(&ws, opt.max_entries + 16, max_stacks, max_len, opt.seed_len);
            for (size_t j; (j = next++) < vr.size(); ) {
                int r = vr[j], b = h_bid[r];
                int rr = aln_search_host(bwt, &meta, cnt_table, &opt, &bopts[b], bstacks[b], &reads[r], &ws);
                bool ok = rr == res.n[r] && (rr <= 0 || !memcmp(ws.aln, res.aln[r].data(), (size_t)rr * sizeof(bwtgap_aln_t)))
                          && ws.last_pops == warp_pops[r] && ws.last_trace == warp_trace[r];
                if (!ok && bad++ < 10)
                    fprintf(stderr, "[galn verify] read %d (%s) differs: cpu n=%d pops=%llu | gpu n=%d pops=%llu\n", r, reads[r].name,
                            rr, (unsigned long long)ws.last_pops, res.n[r], warp_pops[r]);
            }
            aln_ws_free(&ws);
        });
        for (auto &th : vt) th.join();
        fprintf(stderr, "[galn verify] %zu warp-engine reads re-run on the CPU: %ld differ -> %s\n", vr.size(), (long)bad,
                bad? "FAIL" : "PASS");
        verify_fail = bad? 6 : 0;
    }

    /* ---- write .sai in input order ---- */
    sai_write_header(out, &opt);
    long mapped = 0, n_rec = 0;
    int err = 0;
    for (int r = 0; r < nreads; ++r) {
        int na = res.n[r];
        if (na < 0) {
            fprintf(stderr, "[galn] read %d (%s): %s %d\n", r, reads[r].name, na == INT_MIN? "never searched" : "search error", na);
            err = 4; na = 0;
        }
        sai_write_read(out, na, (const sai_aln1_t*)res.aln[r].data());
        if (na > 0) { ++mapped; n_rec += na; }
    }
    if (fflush(out) != 0 || ferror(out) || (out != stdout && fclose(out) != 0)) { fprintf(stderr, "[galn] error writing the .sai\n"); err = 5; }
    double t_end = now_s();

    long cr = 0, cd = 0; double cb = 0, cp = 0;
    for (int w = 0; w < workers; ++w) { cr += wreads[w]; cd += wdup[w]; cb += wbusy[w]; cp += wpops[w]; }
    fprintf(stderr, "[galn] %d reads (%ld with hits, %ld records) in %.2f s: load %.2f s, search %.2f s, write %.2f s\n",
            nreads, mapped, n_rec, t_end - t0, t_loaded - t0, t_search - t_loaded, t_end - t_search);
    if (verbose) {
        for (auto &dp : S.devs) {
            DevState &D = *dp;
            fprintf(stderr, "[galn] GPU %d (NUMA %d): %zu reads; light tier %ld finished, %.2f s (kernels %.2f s); "
                    "phase W %ld reads on %ld warps, %.2f s (kernel %.2f s, %.3e pops/s; executed %.3e pops/s), %ld giant reported; "
                    "phase G %ld reads on %ld warps, %.2f s (kernel %.2f s)\n",
                    D.dev, D.numa, D.reads.size(), D.n_light_done, D.t_light, D.k_light, D.n_W_gpu, D.nwW, D.t_W, D.k_W,
                    D.k_W > 0? D.pops_W / D.k_W : 0.0, D.k_W + D.k_G > 0? D.epops / (D.k_W + D.k_G) : 0.0, D.n_giant, D.n_G_gpu, D.nwG, D.t_G, D.k_G);
        }
        long n_unsup = 0;
        for (auto &dp : S.devs) n_unsup += dp->n_unsup;
        fprintf(stderr, "[galn] CPU: %d workers%s (%zu index copies), %ld reads (%ld long/unsupported), %ld duplicates discarded, "
                "%.1f thread-s, %.1f ns/pop (interleave %d)\n", workers, pin? " pinned" : "", rep_node.empty()? (size_t)1 : rep_node.size(),
                cr, n_unsup, cd, cb, cp > 0? 1e9 * cb / cp : 0.0, il_n);
    }
    free_reads(reads, nreads);
    if (rep_node.empty()) free(bwt); else for (auto p : rep_ptr) free(p);
    return err? err : verify_fail;
}

extern "C" int galn_index(int argc, char *argv[]);   /* src/index/galn_index.c */
extern "C" int galn_samse(int argc, char *argv[]);   /* src/samse/galn_samse.c */
extern "C" int galn_sampe(int argc, char *argv[]);   /* src/samse/galn_sampe.c */

int main(int argc, char **argv)
{
    if (argc < 2) return usage();
    if (!strcmp(argv[1], "aln")) return galn_aln(argc - 2, argv + 2);
    if (!strcmp(argv[1], "index")) return galn_index(argc - 2, argv + 2);
    if (!strcmp(argv[1], "samse")) return galn_samse(argc - 2, argv + 2);
    if (!strcmp(argv[1], "sampe")) return galn_sampe(argc - 2, argv + 2);
    if (!strcmp(argv[1], "--version") || !strcmp(argv[1], "version")) { printf("G-ALN %s\n", GALN_VERSION); return 0; }
    fprintf(stderr, "[galn] unknown command '%s'\n", argv[1]);
    return usage();
}
