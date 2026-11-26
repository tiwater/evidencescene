/* EvidenceScene C11/SIMD deterministic computational kernels.
 *
 * Strictly adheres to IEEE-754 arithmetic order (-fno-fast-math -ffp-contract=off)
 * so every floating-point result matches the pure-NumPy reference path to 0 ULP.
 * Uses only standard CPython <Python.h> + PEP 3118 buffer protocol (zero NumPy
 * C-ABI build dependency).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <pthread.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define ES_HAVE_X86_RUNTIME_SIMD 1
#else
#define ES_HAVE_X86_RUNTIME_SIMD 0
#endif

/* Runtime-detected SIMD ISA level (initialized once in PyInit__native):
 * 0 = scalar/SSE2, 1 = AVX2, 2 = AVX-512 (F+DQ+VL)
 */
static int g_simd_level = 0;

static PyObject *py_simd_isa(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    (void)self;
    if (g_simd_level >= 2) return PyUnicode_FromString("avx512");
    if (g_simd_level == 1) return PyUnicode_FromString("avx2");
    return PyUnicode_FromString("scalar");
}

/* Cached module references initialized in PyInit__native */
static PyObject *g_np_empty = NULL;
static PyObject *g_np_ndarray = NULL;
static PyObject *g_np_int64 = NULL;
static PyObject *g_np_int32 = NULL;
static PyObject *g_np_uint16 = NULL;
static PyObject *g_np_uint8 = NULL;
static PyObject *g_np_float64 = NULL;
static PyObject *g_np_float32 = NULL;
static PyObject *g_np_intp = NULL;
static PyObject *g_np_bool = NULL;
static PyObject *g_str_size = NULL;
static PyObject *g_str_counts = NULL;
static PyObject *g_str_owner_priority = NULL;
static PyObject *g_str_coarse_shadow = NULL;
static PyObject *g_str_within_budget = NULL;
static PyObject *g_str_insufficient_coarse = NULL;

/* ============================================================================
 * PEP 3118 Recycled Slab Buffer Pool (eliminates mmap/munmap page-fault churn)
 * ============================================================================ */

#define SLAB_POOL_CAP 48
#define SLAB_POOL_MIN_BYTES 4096ULL
#define SLAB_POOL_MAX_BYTES (16ULL * 1024ULL * 1024ULL)
#define SLAB_POOL_TOTAL_MAX (128ULL * 1024ULL * 1024ULL)

typedef struct {
    void *ptr;
    size_t cap;
} SlabEntry;

static SlabEntry g_slab_pool[SLAB_POOL_CAP];
static int g_slab_pool_len = 0;
static size_t g_slab_pool_bytes = 0;
#ifndef _WIN32
static pthread_mutex_t g_slab_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

static void *slab_alloc_aligned(size_t cap) {
#ifndef _WIN32
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, cap) != 0) return NULL;
    return ptr;
#else
    return malloc(cap);
#endif
}

static void *slab_acquire(size_t need_bytes, size_t *out_cap) {
    size_t gran = (need_bytes >= 65536ULL) ? 65536ULL : 4096ULL;
    size_t cap = ((need_bytes + gran - 1ULL) / gran) * gran;
    if (cap == 0) cap = 4096ULL;

#ifndef _WIN32
    pthread_mutex_lock(&g_slab_mutex);
    int best_idx = -1;
    size_t best_cap = SIZE_MAX;
    for (int i = 0; i < g_slab_pool_len; i++) {
        size_t ec = g_slab_pool[i].cap;
        if (ec >= need_bytes && ec <= need_bytes * 2ULL && ec < best_cap) {
            best_cap = ec;
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        void *ptr = g_slab_pool[best_idx].ptr;
        g_slab_pool[best_idx] = g_slab_pool[--g_slab_pool_len];
        g_slab_pool_bytes -= best_cap;
        pthread_mutex_unlock(&g_slab_mutex);
        *out_cap = best_cap;
        return ptr;
    }
    pthread_mutex_unlock(&g_slab_mutex);
#endif
    void *ptr = slab_alloc_aligned(cap);
    if (!ptr) return NULL;
    *out_cap = cap;
    return ptr;
}

static void slab_release(void *ptr, size_t cap) {
    if (!ptr) return;
#ifndef _WIN32
    if (cap >= SLAB_POOL_MIN_BYTES && cap <= SLAB_POOL_MAX_BYTES) {
        pthread_mutex_lock(&g_slab_mutex);
        if (g_slab_pool_len < SLAB_POOL_CAP && g_slab_pool_bytes + cap <= SLAB_POOL_TOTAL_MAX) {
            g_slab_pool[g_slab_pool_len].ptr = ptr;
            g_slab_pool[g_slab_pool_len].cap = cap;
            g_slab_pool_len++;
            g_slab_pool_bytes += cap;
            ptr = NULL;
        }
        pthread_mutex_unlock(&g_slab_mutex);
    }
#else
    (void)cap;
#endif
    if (ptr) free(ptr);
}

typedef struct {
    PyObject_HEAD
    void *data;
    Py_ssize_t size;
    size_t cap;
} PooledBufferObject;

static void pooled_buffer_dealloc(PooledBufferObject *self) {
    slab_release(self->data, self->cap);
    self->data = NULL;
    self->cap = 0;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int pooled_buffer_getbuffer(PyObject *exporter, Py_buffer *view, int flags) {
    PooledBufferObject *self = (PooledBufferObject *)exporter;
    return PyBuffer_FillInfo(view, exporter, self->data, self->size, 0, flags);
}

static PyBufferProcs pooled_buffer_as_buffer = {
    .bf_getbuffer = pooled_buffer_getbuffer,
    .bf_releasebuffer = NULL,
};

static PyTypeObject PooledBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "evidencescene._native._PooledBuffer",
    .tp_basicsize = sizeof(PooledBufferObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)pooled_buffer_dealloc,
    .tp_as_buffer = &pooled_buffer_as_buffer,
};

/* ============================================================================
 * Thread-local scratch memory pool (eliminates mmap/munmap page-fault churn)
 * ============================================================================ */

#define DIRECT_GRID_MAX 262144  /* 1 MiB int32_t table -> hot in L3 cache */

typedef struct {
    int64_t kx;
    int64_t ky;
    int64_t kz;
    int64_t count;
    double sum0;
    double sum1;
    double sum2;
    double sr;
    double sg;
    double sb;
    uint8_t used;
} VoxelHashSlot;

typedef struct {
    int64_t kx;
    int64_t ky;
    int64_t kz;
    int32_t rank;
    uint8_t used;
} KeyRankSlot;

typedef struct {
    int64_t kx, ky, kz;
} Key3;

typedef struct {
    int64_t kx, ky, kz;
    int64_t count;
    double sum0, sum1, sum2;
    double sr, sg, sb;
} VoxelEntry;

typedef struct {
    int64_t kx, ky, kz;
    Py_ssize_t orig_idx;
} KeyWithIdx;

typedef struct {
    int64_t v, c;
    Py_ssize_t lex_pos;
    Py_ssize_t orig_idx;
} PrioItem;

typedef struct {
    int64_t cx, cy, cz;
    uint8_t used;
} CoarseSlot;

typedef struct {
    int32_t start;
    int32_t end;
    int32_t left;
    int32_t right;
    int32_t axis;
    double bmin[3];
    double bmax[3];
} KDNode;

typedef struct {
    int32_t *grid;
    size_t grid_cap;
    int32_t *p_idx;
    size_t p_idx_cap;
    int64_t *counts;
    size_t counts_cap;
    uint8_t *marks;
    size_t marks_cap;
    Key3 *keys_tmp;
    size_t keys_tmp_cap;
    VoxelHashSlot *vht;
    size_t vht_cap;
    KeyRankSlot *kht;
    size_t kht_cap;
    CoarseSlot *cht;
    size_t cht_cap;
    VoxelEntry *ve;
    size_t ve_cap;
    KeyWithIdx *kwi;
    size_t kwi_cap;
    PrioItem *pitems;
    size_t pitems_cap;
    double *dbl_tmp;
    size_t dbl_tmp_cap;
    double *dbl_tmp2;
    size_t dbl_tmp2_cap;
    int64_t *i64_tmp;
    size_t i64_tmp_cap;
    Py_ssize_t *idx_tmp;
    size_t idx_tmp_cap;
    KDNode *kd_nodes;
    size_t kd_nodes_cap;
    void *byte_tmp;
    size_t byte_tmp_cap;
} ThreadScratch;

static void scratch_free(void *ptr) {
    ThreadScratch *s = (ThreadScratch *)ptr;
    if (!s) return;
    free(s->grid);
    free(s->p_idx);
    free(s->counts);
    free(s->marks);
    free(s->keys_tmp);
    free(s->vht);
    free(s->kht);
    free(s->cht);
    free(s->ve);
    free(s->kwi);
    free(s->pitems);
    free(s->dbl_tmp);
    free(s->dbl_tmp2);
    free(s->i64_tmp);
    free(s->idx_tmp);
    free(s->kd_nodes);
    free(s->byte_tmp);
    free(s);
}

#ifndef _WIN32
static pthread_key_t g_scratch_key;
static pthread_once_t g_scratch_once = PTHREAD_ONCE_INIT;

static void make_scratch_key(void) {
    pthread_key_create(&g_scratch_key, scratch_free);
}

static ThreadScratch *get_scratch(void) {
    pthread_once(&g_scratch_once, make_scratch_key);
    ThreadScratch *s = (ThreadScratch *)pthread_getspecific(g_scratch_key);
    if (!s) {
        s = (ThreadScratch *)calloc(1, sizeof(ThreadScratch));
        if (s) {
            pthread_setspecific(g_scratch_key, s);
        }
    }
    return s;
}
#else
static ThreadScratch g_win_scratch = {0};
static ThreadScratch *get_scratch(void) {
    return &g_win_scratch;
}
#endif

static int ensure_grid(ThreadScratch *s, size_t n) {
    if (s->grid_cap >= n) return 0;
    size_t cap = n < 4096 ? 4096 : n;
    int32_t *p = (int32_t *)realloc(s->grid, cap * sizeof(int32_t));
    if (!p) return -1;
    s->grid = p;
    s->grid_cap = cap;
    return 0;
}

static int ensure_p_idx(ThreadScratch *s, size_t n) {
    if (s->p_idx_cap >= n) return 0;
    size_t cap = n < 4096 ? 4096 : n;
    int32_t *p = (int32_t *)realloc(s->p_idx, cap * sizeof(int32_t));
    if (!p) return -1;
    s->p_idx = p;
    s->p_idx_cap = cap;
    return 0;
}

static int ensure_counts(ThreadScratch *s, size_t n) {
    if (s->counts_cap >= n) return 0;
    size_t cap = n < 4096 ? 4096 : n;
    int64_t *p = (int64_t *)realloc(s->counts, cap * sizeof(int64_t));
    if (!p) return -1;
    s->counts = p;
    s->counts_cap = cap;
    return 0;
}

static int ensure_marks(ThreadScratch *s, size_t n) {
    if (s->marks_cap >= n) return 0;
    size_t cap = n < 4096 ? 4096 : n;
    uint8_t *p = (uint8_t *)realloc(s->marks, cap * sizeof(uint8_t));
    if (!p) return -1;
    s->marks = p;
    s->marks_cap = cap;
    return 0;
}

static int ensure_byte_tmp(ThreadScratch *s, size_t nbytes) {
    if (s->byte_tmp_cap >= nbytes) return 0;
    size_t cap = nbytes < 65536 ? 65536 : nbytes;
    void *p = realloc(s->byte_tmp, cap);
    if (!p) return -1;
    s->byte_tmp = p;
    s->byte_tmp_cap = cap;
    return 0;
}

static int ensure_keys_tmp(ThreadScratch *s, size_t n) {
    if (s->keys_tmp_cap >= n) return 0;
    size_t cap = n < 4096 ? 4096 : n;
    Key3 *p = (Key3 *)realloc(s->keys_tmp, cap * sizeof(Key3));
    if (!p) return -1;
    s->keys_tmp = p;
    s->keys_tmp_cap = cap;
    return 0;
}

static int ensure_vht(ThreadScratch *s, size_t n) {
    if (s->vht_cap >= n) return 0;
    VoxelHashSlot *p = (VoxelHashSlot *)realloc(s->vht, n * sizeof(VoxelHashSlot));
    if (!p) return -1;
    s->vht = p;
    s->vht_cap = n;
    return 0;
}

static int ensure_kht(ThreadScratch *s, size_t n) {
    if (s->kht_cap >= n) return 0;
    KeyRankSlot *p = (KeyRankSlot *)realloc(s->kht, n * sizeof(KeyRankSlot));
    if (!p) return -1;
    s->kht = p;
    s->kht_cap = n;
    return 0;
}

static int ensure_cht(ThreadScratch *s, size_t n) {
    if (s->cht_cap >= n) return 0;
    CoarseSlot *p = (CoarseSlot *)realloc(s->cht, n * sizeof(CoarseSlot));
    if (!p) return -1;
    s->cht = p;
    s->cht_cap = n;
    return 0;
}

static int ensure_ve(ThreadScratch *s, size_t n) {
    if (s->ve_cap >= n) return 0;
    size_t cap = n < 1024 ? 1024 : n;
    VoxelEntry *p = (VoxelEntry *)realloc(s->ve, cap * sizeof(VoxelEntry));
    if (!p) return -1;
    s->ve = p;
    s->ve_cap = cap;
    return 0;
}

static int ensure_kwi(ThreadScratch *s, size_t n) {
    if (s->kwi_cap >= n) return 0;
    size_t cap = n < 1024 ? 1024 : n;
    KeyWithIdx *p = (KeyWithIdx *)realloc(s->kwi, cap * sizeof(KeyWithIdx));
    if (!p) return -1;
    s->kwi = p;
    s->kwi_cap = cap;
    return 0;
}

static int ensure_pitems(ThreadScratch *s, size_t n) {
    if (s->pitems_cap >= n) return 0;
    size_t cap = n < 1024 ? 1024 : n;
    PrioItem *p = (PrioItem *)realloc(s->pitems, cap * sizeof(PrioItem));
    if (!p) return -1;
    s->pitems = p;
    s->pitems_cap = cap;
    return 0;
}

static int ensure_dbl_tmp(ThreadScratch *s, size_t n) {
    if (s->dbl_tmp_cap >= n) return 0;
    size_t cap = n < 4096 ? 4096 : n;
    double *p = (double *)realloc(s->dbl_tmp, cap * sizeof(double));
    if (!p) return -1;
    s->dbl_tmp = p;
    s->dbl_tmp_cap = cap;
    return 0;
}

static int ensure_dbl_tmp2(ThreadScratch *s, size_t n) {
    if (s->dbl_tmp2_cap >= n) return 0;
    size_t cap = n < 4096 ? 4096 : n;
    double *p = (double *)realloc(s->dbl_tmp2, cap * sizeof(double));
    if (!p) return -1;
    s->dbl_tmp2 = p;
    s->dbl_tmp2_cap = cap;
    return 0;
}

static int ensure_i64_tmp(ThreadScratch *s, size_t n) {
    if (s->i64_tmp_cap >= n) return 0;
    size_t cap = n < 4096 ? 4096 : n;
    int64_t *p = (int64_t *)realloc(s->i64_tmp, cap * sizeof(int64_t));
    if (!p) return -1;
    s->i64_tmp = p;
    s->i64_tmp_cap = cap;
    return 0;
}

static int ensure_idx_tmp(ThreadScratch *s, size_t n) {
    if (s->idx_tmp_cap >= n) return 0;
    size_t cap = n < 4096 ? 4096 : n;
    Py_ssize_t *p = (Py_ssize_t *)realloc(s->idx_tmp, cap * sizeof(Py_ssize_t));
    if (!p) return -1;
    s->idx_tmp = p;
    s->idx_tmp_cap = cap;
    return 0;
}

static int ensure_kd_nodes(ThreadScratch *s, size_t n) {
    if (s->kd_nodes_cap >= n) return 0;
    size_t cap = n < 512 ? 512 : n;
    KDNode *p = (KDNode *)realloc(s->kd_nodes, cap * sizeof(KDNode));
    if (!p) return -1;
    s->kd_nodes = p;
    s->kd_nodes_cap = cap;
    return 0;
}

/* ============================================================================
 * NumPy array allocation helpers (PEP 3118 writable C-contiguous buffers)
 * ============================================================================ */

static PyObject *alloc_fast_shape(PyObject *shape, Py_ssize_t byte_len, PyObject *dtype_obj, void **out_ptr) {
    if (byte_len >= (Py_ssize_t)SLAB_POOL_MIN_BYTES) {
        size_t cap = 0;
        void *data = slab_acquire((size_t)byte_len, &cap);
        if (!data) {
            Py_DECREF(shape);
            return PyErr_NoMemory();
        }
        PooledBufferObject *owner = PyObject_New(PooledBufferObject, &PooledBufferType);
        if (!owner) {
            slab_release(data, cap);
            Py_DECREF(shape);
            return NULL;
        }
        owner->data = data;
        owner->size = byte_len;
        owner->cap = cap;
        PyObject *arr = PyObject_CallFunctionObjArgs(g_np_ndarray, shape, dtype_obj, (PyObject *)owner, NULL);
        Py_DECREF(owner);
        Py_DECREF(shape);
        if (!arr) return NULL;
        if (out_ptr) *out_ptr = data;
        return arr;
    }
    PyObject *arr = PyObject_CallFunctionObjArgs(g_np_empty, shape, dtype_obj, NULL);
    Py_DECREF(shape);
    if (!arr) return NULL;
    if (out_ptr && byte_len > 0) {
        Py_buffer view;
        if (PyObject_GetBuffer(arr, &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) {
            Py_DECREF(arr);
            return NULL;
        }
        *out_ptr = view.buf;
        PyBuffer_Release(&view);
    } else if (out_ptr) {
        *out_ptr = NULL;
    }
    return arr;
}

static PyObject *alloc_fast_1d(Py_ssize_t len, Py_ssize_t itemsize, PyObject *dtype_obj, void **out_ptr) {
    PyObject *shape = Py_BuildValue("(n)", len);
    if (!shape) return NULL;
    return alloc_fast_shape(shape, len * itemsize, dtype_obj, out_ptr);
}

static PyObject *alloc_fast_2d(Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t itemsize, PyObject *dtype_obj, void **out_ptr) {
    PyObject *shape = Py_BuildValue("(nn)", rows, cols);
    if (!shape) return NULL;
    return alloc_fast_shape(shape, rows * cols * itemsize, dtype_obj, out_ptr);
}

static inline uint64_t hash_key3(int64_t kx, int64_t ky, int64_t kz) {
    uint64_t h = (uint64_t)kx * 0x9e3779b97f4a7c15ULL;
    h ^= (uint64_t)ky * 0xbf58476d1ce4e5b9ULL;
    h ^= (uint64_t)kz * 0x94d049bb133111ebULL;
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

static size_t next_pow2(size_t x) {
    size_t p = 16;
    while (p < x) {
        if (p > ((size_t)1 << 60)) return p;
        p <<= 1;
    }
    return p;
}

static int cmp_key3(int64_t ax, int64_t ay, int64_t az, int64_t bx, int64_t by, int64_t bz) {
    if (ax != bx) return ax < bx ? -1 : 1;
    if (ay != by) return ay < by ? -1 : 1;
    if (az != bz) return az < bz ? -1 : 1;
    return 0;
}

/* ============================================================================
 * 1. observation_voxel_centroids
 * ============================================================================ */

static int cmp_voxel_entry(const void *a, const void *b) {
    const VoxelEntry *ea = (const VoxelEntry *)a;
    const VoxelEntry *eb = (const VoxelEntry *)b;
    return cmp_key3(ea->kx, ea->ky, ea->kz, eb->kx, eb->ky, eb->kz);
}

#if ES_HAVE_X86_RUNTIME_SIMD
__attribute__((target("avx512f,avx512dq,avx512vl")))
static Py_ssize_t quantize_voxel_keys_avx512(
    const double *pts, int64_t *k_flat, Py_ssize_t total_f,
    double voxel_size_m, double max_key_f)
{
    Py_ssize_t m = 0;
    __m512d v_vs = _mm512_set1_pd(voxel_size_m);
    __m512d v_bound = _mm512_set1_pd(max_key_f);
    __m512d v_inf = _mm512_set1_pd(HUGE_VAL);
    __m512d v_abs = _mm512_castsi512_pd(_mm512_set1_epi64(0x7fffffffffffffffLL));
    for (; m + 8 <= total_f; m += 8) {
        __m512d vp = _mm512_loadu_pd(pts + m);
        __m512d vp_abs = _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vp), _mm512_castpd_si512(v_abs)));
        __mmask8 ok_p = _mm512_cmp_pd_mask(vp_abs, v_inf, _CMP_LT_OQ);
        if (ok_p != 0xFF) break;
        __m512d vs = _mm512_floor_pd(_mm512_div_pd(vp, v_vs));
        __m512d vs_abs = _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vs), _mm512_castpd_si512(v_abs)));
        __mmask8 ok_s = _mm512_cmp_pd_mask(vs_abs, v_bound, _CMP_LT_OQ);
        if (ok_s != 0xFF) break;
        __m512i vk = _mm512_cvttpd_epi64(vs);
        _mm512_storeu_si512((__m512i *)(k_flat + m), vk);
    }
    return m;
}

__attribute__((target("avx2")))
static Py_ssize_t quantize_voxel_keys_avx2(
    const double *pts, int64_t *k_flat, Py_ssize_t total_f,
    double voxel_size_m, double max_key_f)
{
    Py_ssize_t m = 0;
    __m256d v_vs = _mm256_set1_pd(voxel_size_m);
    __m256d v_bound = _mm256_set1_pd(max_key_f);
    __m256d v_inf = _mm256_set1_pd(HUGE_VAL);
    __m256d v_abs = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
    double tmp_s[4];
    for (; m + 4 <= total_f; m += 4) {
        __m256d vp = _mm256_loadu_pd(pts + m);
        __m256d vp_abs = _mm256_and_pd(vp, v_abs);
        int ok_p = _mm256_movemask_pd(_mm256_cmp_pd(vp_abs, v_inf, _CMP_LT_OQ));
        if (ok_p != 0xF) break;
        __m256d vs = _mm256_floor_pd(_mm256_div_pd(vp, v_vs));
        __m256d vs_abs = _mm256_and_pd(vs, v_abs);
        int ok_s = _mm256_movemask_pd(_mm256_cmp_pd(vs_abs, v_bound, _CMP_LT_OQ));
        if (ok_s != 0xF) break;
        _mm256_storeu_pd(tmp_s, vs);
        k_flat[m + 0] = (int64_t)tmp_s[0];
        k_flat[m + 1] = (int64_t)tmp_s[1];
        k_flat[m + 2] = (int64_t)tmp_s[2];
        k_flat[m + 3] = (int64_t)tmp_s[3];
    }
    return m;
}
#endif

static PyObject *py_voxel_centroids(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2 && nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "voxel_centroids expects (points, voxel_size_m[, colors_rgb_u8])");
        return NULL;
    }
    PyObject *pts_obj = args[0];
    double voxel_size_m = PyFloat_AsDouble(args[1]);
    if (voxel_size_m == -1.0 && PyErr_Occurred()) return NULL;
    int with_rgb = (nargs == 3 && args[2] != Py_None);

    Py_buffer pts_buf = {0}, rgb_buf = {0};
    int got_rgb = 0;
    if (PyObject_GetBuffer(pts_obj, &pts_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (pts_buf.itemsize != (Py_ssize_t)sizeof(double) || pts_buf.len % (3 * (Py_ssize_t)sizeof(double)) != 0) {
        PyBuffer_Release(&pts_buf);
        PyErr_SetString(PyExc_ValueError, "points must be contiguous float64 Nx3");
        return NULL;
    }
    Py_ssize_t n = pts_buf.len / (3 * (Py_ssize_t)sizeof(double));
    const double *pts = (const double *)pts_buf.buf;
    const uint8_t *rgb = NULL;
    if (with_rgb) {
        if (PyObject_GetBuffer(args[2], &rgb_buf, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) {
            PyBuffer_Release(&pts_buf);
            return NULL;
        }
        got_rgb = 1;
        if (rgb_buf.ndim != 2 || rgb_buf.shape[0] != n || rgb_buf.shape[1] != 3 || rgb_buf.itemsize != 1) {
            PyBuffer_Release(&rgb_buf);
            PyBuffer_Release(&pts_buf);
            PyErr_SetString(PyExc_ValueError, "colors_rgb_u8 must be contiguous uint8 Nx3 matching points");
            return NULL;
        }
        rgb = (const uint8_t *)rgb_buf.buf;
    }

    if (n == 0) {
        if (got_rgb) PyBuffer_Release(&rgb_buf);
        PyBuffer_Release(&pts_buf);
        PyObject *k_arr = alloc_fast_2d(0, 3, (Py_ssize_t)sizeof(int64_t), g_np_int64, NULL);
        PyObject *c_arr = alloc_fast_2d(0, 3, (Py_ssize_t)sizeof(double), g_np_float64, NULL);
        PyObject *r_arr = with_rgb ? alloc_fast_2d(0, 3, (Py_ssize_t)sizeof(float), g_np_float32, NULL) : NULL;
        if (!k_arr || !c_arr || (with_rgb && !r_arr)) {
            Py_XDECREF(k_arr); Py_XDECREF(c_arr); Py_XDECREF(r_arr);
            return NULL;
        }
        PyObject *ret = with_rgb ? PyTuple_Pack(3, k_arr, c_arr, r_arr) : PyTuple_Pack(2, k_arr, c_arr);
        Py_DECREF(k_arr); Py_DECREF(c_arr); Py_XDECREF(r_arr);
        return ret;
    }

    int err = 0;
    int sorted_strict = 1;
    int64_t min0 = 0, min1 = 0, min2 = 0;
    int64_t max0 = 0, max1 = 0, max2 = 0;
    const double max_key_f = 9223372036854775808.0; /* float(INT64_MAX) */

    ThreadScratch *sc = get_scratch();
    if (!sc || ensure_keys_tmp(sc, (size_t)n) < 0) {
        if (got_rgb) PyBuffer_Release(&rgb_buf);
        PyBuffer_Release(&pts_buf);
        return PyErr_NoMemory();
    }
    Key3 *ktmp = sc->keys_tmp;

    Py_BEGIN_ALLOW_THREADS
    int64_t *k_flat = &ktmp[0].kx;
    Py_ssize_t total_f = 3 * n;
    Py_ssize_t m = 0;
#if ES_HAVE_X86_RUNTIME_SIMD
    if (g_simd_level >= 2) {
        m = quantize_voxel_keys_avx512(pts, k_flat, total_f, voxel_size_m, max_key_f);
    } else if (g_simd_level == 1) {
        m = quantize_voxel_keys_avx2(pts, k_flat, total_f, voxel_size_m, max_key_f);
    }
#endif
    if (m < total_f) {
        Py_ssize_t i_start = m / 3;
        for (Py_ssize_t i = i_start; i < n; i++) {
            double p0 = pts[3 * i + 0], p1 = pts[3 * i + 1], p2 = pts[3 * i + 2];
            if (!isfinite(p0) || !isfinite(p1) || !isfinite(p2)) { err = 1; break; }
            double s0 = floor(p0 / voxel_size_m);
            double s1 = floor(p1 / voxel_size_m);
            double s2 = floor(p2 / voxel_size_m);
            if (!isfinite(s0) || !isfinite(s1) || !isfinite(s2) ||
                fabs(s0) >= max_key_f || fabs(s1) >= max_key_f || fabs(s2) >= max_key_f) {
                err = 2; break;
            }
            ktmp[i].kx = (int64_t)s0;
            ktmp[i].ky = (int64_t)s1;
            ktmp[i].kz = (int64_t)s2;
        }
    }

    if (err == 0) {
        min0 = max0 = ktmp[0].kx;
        min1 = max1 = ktmp[0].ky;
        min2 = max2 = ktmp[0].kz;
        int64_t pk0 = min0, pk1 = min1, pk2 = min2;
        for (Py_ssize_t i = 1; i < n; i++) {
            int64_t k0 = ktmp[i].kx, k1 = ktmp[i].ky, k2 = ktmp[i].kz;
            if (k0 < min0) min0 = k0; else if (k0 > max0) max0 = k0;
            if (k1 < min1) min1 = k1; else if (k1 > max1) max1 = k1;
            if (k2 < min2) min2 = k2; else if (k2 > max2) max2 = k2;
            if (sorted_strict) {
                if (!(k0 > pk0 || (k0 == pk0 && (k1 > pk1 || (k1 == pk1 && k2 > pk2))))) {
                    sorted_strict = 0;
                }
                pk0 = k0; pk1 = k1; pk2 = k2;
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (err == 1) {
        if (got_rgb) PyBuffer_Release(&rgb_buf);
        PyBuffer_Release(&pts_buf);
        PyErr_SetString(PyExc_ValueError, "each surface observation must be finite Nx3");
        return NULL;
    }
    if (err == 2) {
        if (got_rgb) PyBuffer_Release(&rgb_buf);
        PyBuffer_Release(&pts_buf);
        PyErr_SetString(PyExc_ValueError, "surface voxel index overflow");
        return NULL;
    }

    /* Fast Path A: already strictly lexicographically sorted */
    if (sorted_strict) {
        int64_t *out_k = NULL;
        double *out_c = NULL;
        float *out_r = NULL;
        PyObject *k_arr = alloc_fast_2d(n, 3, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_k);
        PyObject *c_arr = k_arr ? alloc_fast_2d(n, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out_c) : NULL;
        PyObject *r_arr = (c_arr && with_rgb) ? alloc_fast_2d(n, 3, (Py_ssize_t)sizeof(float), g_np_float32, (void **)&out_r) : NULL;
        if (!k_arr || !c_arr || (with_rgb && !r_arr)) {
            Py_XDECREF(k_arr); Py_XDECREF(c_arr); Py_XDECREF(r_arr);
            if (got_rgb) PyBuffer_Release(&rgb_buf);
            PyBuffer_Release(&pts_buf);
            return NULL;
        }
        Py_BEGIN_ALLOW_THREADS
        memcpy(out_k, ktmp, (size_t)n * sizeof(Key3));
        memcpy(out_c, pts, (size_t)n * 3 * sizeof(double));
        if (out_r) {
            for (Py_ssize_t i = 0; i < 3 * n; i++) out_r[i] = (float)rgb[i];
        }
        Py_END_ALLOW_THREADS
        if (got_rgb) PyBuffer_Release(&rgb_buf);
        PyBuffer_Release(&pts_buf);
        PyObject *ret = with_rgb ? PyTuple_Pack(3, k_arr, c_arr, r_arr) : PyTuple_Pack(2, k_arr, c_arr);
        Py_DECREF(k_arr); Py_DECREF(c_arr); Py_XDECREF(r_arr);
        return ret;
    }

    /* Compute bounded grid span safely using __int128 */
    __int128 d0 = (__int128)max0 - (__int128)min0 + 1;
    __int128 d1 = (__int128)max1 - (__int128)min1 + 1;
    __int128 d2 = (__int128)max2 - (__int128)min2 + 1;
    int use_grid = 0;
    size_t total_span = 0;
    if (d0 <= DIRECT_GRID_MAX && d1 <= DIRECT_GRID_MAX && d2 <= DIRECT_GRID_MAX) {
        __int128 prod = d0 * d1 * d2;
        if (prod > 0 && prod <= DIRECT_GRID_MAX && prod <= (__int128)16 * (__int128)n && n <= 100000000) {
            use_grid = 1;
            total_span = (size_t)prod;
        }
    }

    if (use_grid) {
        if (ensure_grid(sc, total_span) < 0 || ensure_p_idx(sc, (size_t)n) < 0) {
            if (got_rgb) PyBuffer_Release(&rgb_buf);
            PyBuffer_Release(&pts_buf);
            return PyErr_NoMemory();
        }
        int32_t *grid = sc->grid;
        int32_t *p_idx = sc->p_idx;
        int64_t d0_i = (int64_t)d0;
        int64_t s1 = (int64_t)d1;
        int64_t s2 = (int64_t)d2;
        int64_t s12 = s1 * s2;
        Py_ssize_t u_count = 0;

        Py_BEGIN_ALLOW_THREADS
        memset(grid, 0, total_span * sizeof(int32_t));
        for (Py_ssize_t i = 0; i < n; i++) {
            int32_t p = (int32_t)((ktmp[i].kx - min0) * s12 + (ktmp[i].ky - min1) * s2 + (ktmp[i].kz - min2));
            p_idx[i] = p;
            grid[p]++;
        }
        for (size_t p = 0; p < total_span; p++) {
            if (grid[p] > 0) u_count++;
        }
        Py_END_ALLOW_THREADS

        if (ensure_counts(sc, (size_t)u_count) < 0 || (with_rgb && ensure_dbl_tmp(sc, (size_t)u_count * 3) < 0)) {
            if (got_rgb) PyBuffer_Release(&rgb_buf);
            PyBuffer_Release(&pts_buf);
            return PyErr_NoMemory();
        }
        int64_t *out_k = NULL;
        double *out_c = NULL;
        float *out_r = NULL;
        PyObject *k_arr = alloc_fast_2d(u_count, 3, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_k);
        PyObject *c_arr = k_arr ? alloc_fast_2d(u_count, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out_c) : NULL;
        PyObject *r_arr = (c_arr && with_rgb) ? alloc_fast_2d(u_count, 3, (Py_ssize_t)sizeof(float), g_np_float32, (void **)&out_r) : NULL;
        if (!k_arr || !c_arr || (with_rgb && !r_arr)) {
            Py_XDECREF(k_arr); Py_XDECREF(c_arr); Py_XDECREF(r_arr);
            if (got_rgb) PyBuffer_Release(&rgb_buf);
            PyBuffer_Release(&pts_buf);
            return NULL;
        }
        int64_t *counts = sc->counts;
        double *rgb_acc = with_rgb ? sc->dbl_tmp : NULL;

        Py_BEGIN_ALLOW_THREADS
        size_t p_last = 0;
        for (size_t q = total_span; q > 0; q--) {
            if (grid[q - 1] > 0) { p_last = q - 1; break; }
        }
        Py_ssize_t u = 0;
        size_t p = 0;
        for (int64_t dx = 0; dx < d0_i && p < p_last; dx++) {
            int64_t kx_val = min0 + dx;
            for (int64_t dy = 0; dy < s1 && p < p_last; dy++) {
                int64_t ky_val = min1 + dy;
                for (int64_t dz = 0; dz < s2 && p < p_last; dz++, p++) {
                    int32_t cnt = grid[p];
                    int32_t ok = (cnt > 0);
                    out_k[3 * u + 0] = kx_val;
                    out_k[3 * u + 1] = ky_val;
                    out_k[3 * u + 2] = min2 + dz;
                    counts[u] = (int64_t)cnt;
                    grid[p] = (int32_t)u;
                    u += ok;
                }
            }
        }
        {
            int32_t cnt = grid[p_last];
            int64_t q = (int64_t)p_last / s2;
            int64_t dz = (int64_t)p_last - q * s2;
            int64_t dx = q / s1;
            int64_t dy = q - dx * s1;
            out_k[3 * u + 0] = min0 + dx;
            out_k[3 * u + 1] = min1 + dy;
            out_k[3 * u + 2] = min2 + dz;
            counts[u] = (int64_t)cnt;
            grid[p_last] = (int32_t)u;
        }
        memset(out_c, 0, (size_t)u_count * 3 * sizeof(double));
        if (rgb_acc) {
            memset(rgb_acc, 0, (size_t)u_count * 3 * sizeof(double));
            for (Py_ssize_t i = 0; i < n; i++) {
                int32_t uid = grid[p_idx[i]];
                out_c[3 * uid + 0] += pts[3 * i + 0];
                out_c[3 * uid + 1] += pts[3 * i + 1];
                out_c[3 * uid + 2] += pts[3 * i + 2];
                rgb_acc[3 * uid + 0] += (double)rgb[3 * i + 0];
                rgb_acc[3 * uid + 1] += (double)rgb[3 * i + 1];
                rgb_acc[3 * uid + 2] += (double)rgb[3 * i + 2];
            }
            for (Py_ssize_t j = 0; j < u_count; j++) {
                int64_t cnt = counts[j];
                if (cnt > 1) {
                    double c_d = (double)cnt;
                    out_c[3 * j + 0] /= c_d;
                    out_c[3 * j + 1] /= c_d;
                    out_c[3 * j + 2] /= c_d;
                    out_r[3 * j + 0] = (float)(rgb_acc[3 * j + 0] / c_d);
                    out_r[3 * j + 1] = (float)(rgb_acc[3 * j + 1] / c_d);
                    out_r[3 * j + 2] = (float)(rgb_acc[3 * j + 2] / c_d);
                } else {
                    out_r[3 * j + 0] = (float)rgb_acc[3 * j + 0];
                    out_r[3 * j + 1] = (float)rgb_acc[3 * j + 1];
                    out_r[3 * j + 2] = (float)rgb_acc[3 * j + 2];
                }
            }
        } else {
            for (Py_ssize_t i = 0; i < n; i++) {
                int32_t uid = grid[p_idx[i]];
                out_c[3 * uid + 0] += pts[3 * i + 0];
                out_c[3 * uid + 1] += pts[3 * i + 1];
                out_c[3 * uid + 2] += pts[3 * i + 2];
            }
            for (Py_ssize_t j = 0; j < u_count; j++) {
                int64_t cnt = counts[j];
                if (cnt > 1) {
                    double c_d = (double)cnt;
                    out_c[3 * j + 0] /= c_d;
                    out_c[3 * j + 1] /= c_d;
                    out_c[3 * j + 2] /= c_d;
                }
            }
        }
        Py_END_ALLOW_THREADS

        if (got_rgb) PyBuffer_Release(&rgb_buf);
        PyBuffer_Release(&pts_buf);
        PyObject *ret = with_rgb ? PyTuple_Pack(3, k_arr, c_arr, r_arr) : PyTuple_Pack(2, k_arr, c_arr);
        Py_DECREF(k_arr); Py_DECREF(c_arr); Py_XDECREF(r_arr);
        return ret;
    }

    /* General Path C: Open-addressed hash table + lexicographic qsort */
    size_t cap = next_pow2((size_t)n * 2 + 16);
    if (ensure_vht(sc, cap) < 0) {
        if (got_rgb) PyBuffer_Release(&rgb_buf);
        PyBuffer_Release(&pts_buf);
        return PyErr_NoMemory();
    }
    VoxelHashSlot *ht = sc->vht;
    size_t mask = cap - 1;
    Py_ssize_t u_count = 0;

    Py_BEGIN_ALLOW_THREADS
    for (size_t i = 0; i < cap; i++) ht[i].used = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        int64_t k0 = ktmp[i].kx;
        int64_t k1 = ktmp[i].ky;
        int64_t k2 = ktmp[i].kz;
        size_t slot = (size_t)hash_key3(k0, k1, k2) & mask;
        while (ht[slot].used) {
            if (ht[slot].kx == k0 && ht[slot].ky == k1 && ht[slot].kz == k2) {
                ht[slot].sum0 += pts[3 * i + 0];
                ht[slot].sum1 += pts[3 * i + 1];
                ht[slot].sum2 += pts[3 * i + 2];
                if (with_rgb) {
                    ht[slot].sr += (double)rgb[3 * i + 0];
                    ht[slot].sg += (double)rgb[3 * i + 1];
                    ht[slot].sb += (double)rgb[3 * i + 2];
                }
                ht[slot].count++;
                goto found_vht;
            }
            slot = (slot + 1) & mask;
        }
        ht[slot].used = 1;
        ht[slot].kx = k0;
        ht[slot].ky = k1;
        ht[slot].kz = k2;
        ht[slot].sum0 = pts[3 * i + 0];
        ht[slot].sum1 = pts[3 * i + 1];
        ht[slot].sum2 = pts[3 * i + 2];
        if (with_rgb) {
            ht[slot].sr = (double)rgb[3 * i + 0];
            ht[slot].sg = (double)rgb[3 * i + 1];
            ht[slot].sb = (double)rgb[3 * i + 2];
        }
        ht[slot].count = 1;
        u_count++;
        found_vht:;
    }
    Py_END_ALLOW_THREADS

    if (ensure_ve(sc, (size_t)(u_count > 0 ? u_count : 1)) < 0) {
        if (got_rgb) PyBuffer_Release(&rgb_buf);
        PyBuffer_Release(&pts_buf);
        return PyErr_NoMemory();
    }
    VoxelEntry *entries = sc->ve;

    int64_t *out_k = NULL;
    double *out_c = NULL;
    float *out_r = NULL;
    PyObject *k_arr = alloc_fast_2d(u_count, 3, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_k);
    PyObject *c_arr = k_arr ? alloc_fast_2d(u_count, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out_c) : NULL;
    PyObject *r_arr = (c_arr && with_rgb) ? alloc_fast_2d(u_count, 3, (Py_ssize_t)sizeof(float), g_np_float32, (void **)&out_r) : NULL;
    if (!k_arr || !c_arr || (with_rgb && !r_arr)) {
        Py_XDECREF(k_arr); Py_XDECREF(c_arr); Py_XDECREF(r_arr);
        if (got_rgb) PyBuffer_Release(&rgb_buf);
        PyBuffer_Release(&pts_buf);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t idx = 0;
    for (size_t i = 0; i < cap; i++) {
        if (ht[i].used) {
            entries[idx].kx = ht[i].kx;
            entries[idx].ky = ht[i].ky;
            entries[idx].kz = ht[i].kz;
            entries[idx].count = ht[i].count;
            entries[idx].sum0 = ht[i].sum0;
            entries[idx].sum1 = ht[i].sum1;
            entries[idx].sum2 = ht[i].sum2;
            if (with_rgb) {
                entries[idx].sr = ht[i].sr;
                entries[idx].sg = ht[i].sg;
                entries[idx].sb = ht[i].sb;
            }
            idx++;
        }
    }
    qsort(entries, (size_t)u_count, sizeof(VoxelEntry), cmp_voxel_entry);
    for (Py_ssize_t j = 0; j < u_count; j++) {
        out_k[3 * j + 0] = entries[j].kx;
        out_k[3 * j + 1] = entries[j].ky;
        out_k[3 * j + 2] = entries[j].kz;
        double c_d = (double)entries[j].count;
        out_c[3 * j + 0] = entries[j].sum0 / c_d;
        out_c[3 * j + 1] = entries[j].sum1 / c_d;
        out_c[3 * j + 2] = entries[j].sum2 / c_d;
        if (out_r) {
            out_r[3 * j + 0] = (float)(entries[j].sr / c_d);
            out_r[3 * j + 1] = (float)(entries[j].sg / c_d);
            out_r[3 * j + 2] = (float)(entries[j].sb / c_d);
        }
    }
    Py_END_ALLOW_THREADS

    if (got_rgb) PyBuffer_Release(&rgb_buf);
    PyBuffer_Release(&pts_buf);
    PyObject *ret = with_rgb ? PyTuple_Pack(3, k_arr, c_arr, r_arr) : PyTuple_Pack(2, k_arr, c_arr);
    Py_DECREF(k_arr); Py_DECREF(c_arr); Py_XDECREF(r_arr);
    return ret;
}

/* ============================================================================
 * 2. surfel_general_update
 * ============================================================================ */

static int cmp_key3_struct(const void *a, const void *b) {
    const Key3 *ka = (const Key3 *)a;
    const Key3 *kb = (const Key3 *)b;
    return cmp_key3(ka->kx, ka->ky, ka->kz, kb->kx, kb->ky, kb->kz);
}

typedef struct {
    int cam_slot;
    int sign;
    Py_ssize_t n;
    Py_buffer k_buf;
    Py_buffer c_buf;
    Py_buffer r_buf;
    int has_rgb;
    const int64_t *keys;
    const double *centroids;
    const float *rgb;
    int32_t *ids;
} StagedEventC;

static PyObject *py_surfel_general_update(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *old_k_obj, *old_s_obj, *old_sup_obj, *old_cams_seq, *staged_seq;
    PyObject *old_rgb_obj = NULL;
    Py_ssize_t num_cams = 0, max_voxels = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOnn|O",
            &old_k_obj, &old_s_obj, &old_sup_obj, &old_cams_seq, &staged_seq,
            &num_cams, &max_voxels, &old_rgb_obj)) {
        return NULL;
    }
    int with_rgb = (old_rgb_obj != NULL && old_rgb_obj != Py_None);

    Py_buffer old_k_buf = {0}, old_s_buf = {0}, old_sup_buf = {0}, old_rgb_buf = {0};
    int got_old_k = 0, got_old_s = 0, got_old_sup = 0, got_old_rgb = 0;
    Py_buffer stack_old_cam_bufs[16];
    const int64_t *stack_old_cam_ptrs[16];
    Py_buffer *old_cam_bufs = NULL;
    const int64_t **old_cam_ptrs = NULL;
    Py_ssize_t num_old_cams = 0;
    StagedEventC stack_events[32];
    StagedEventC *events = NULL;
    Py_ssize_t n_staged = 0;
    int32_t *old_ids = NULL;
    Key3 *u_keys_buf = NULL;
    double *rgb_w = NULL;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(old_k_obj, &old_k_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_old_k = 1;
    if (PyObject_GetBuffer(old_s_obj, &old_s_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_old_s = 1;
    if (PyObject_GetBuffer(old_sup_obj, &old_sup_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_old_sup = 1;

    Py_ssize_t v_old = old_sup_buf.len / (Py_ssize_t)sizeof(int64_t);
    const int64_t *old_keys = (const int64_t *)old_k_buf.buf;
    const double *old_sums = (const double *)old_s_buf.buf;
    const int64_t *old_sup = (const int64_t *)old_sup_buf.buf;
    const float *old_rgb = NULL;
    Py_ssize_t old_rgb_rows = 0;
    if (with_rgb) {
        if (PyObject_GetBuffer(old_rgb_obj, &old_rgb_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        got_old_rgb = 1;
        old_rgb = (const float *)old_rgb_buf.buf;
        old_rgb_rows = old_rgb_buf.len / (3 * (Py_ssize_t)sizeof(float));
    }

    num_old_cams = PySequence_Size(old_cams_seq);
    if (num_old_cams < 0) goto cleanup;
    if (num_old_cams > 0) {
        if (num_old_cams <= 16) {
            memset(stack_old_cam_bufs, 0, (size_t)num_old_cams * sizeof(Py_buffer));
            old_cam_bufs = stack_old_cam_bufs;
            old_cam_ptrs = stack_old_cam_ptrs;
        } else {
            old_cam_bufs = (Py_buffer *)calloc((size_t)num_old_cams, sizeof(Py_buffer));
            old_cam_ptrs = (const int64_t **)calloc((size_t)num_old_cams, sizeof(int64_t *));
            if (!old_cam_bufs || !old_cam_ptrs) { PyErr_NoMemory(); goto cleanup; }
        }
        for (Py_ssize_t c = 0; c < num_old_cams; c++) {
            PyObject *item = PySequence_ITEM(old_cams_seq, c);
            if (!item) goto cleanup;
            int rc = PyObject_GetBuffer(item, &old_cam_bufs[c], PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS);
            Py_DECREF(item);
            if (rc < 0) goto cleanup;
            old_cam_ptrs[c] = (const int64_t *)old_cam_bufs[c].buf;
        }
    }

    n_staged = PySequence_Size(staged_seq);
    if (n_staged < 0) goto cleanup;
    if (n_staged <= 32) {
        memset(stack_events, 0, (size_t)(n_staged > 0 ? n_staged : 1) * sizeof(StagedEventC));
        events = stack_events;
    } else {
        events = (StagedEventC *)calloc((size_t)n_staged, sizeof(StagedEventC));
        if (!events) { PyErr_NoMemory(); goto cleanup; }
    }

    Py_ssize_t total_keys = v_old;
    for (Py_ssize_t s = 0; s < n_staged; s++) {
        PyObject *tup = PySequence_ITEM(staged_seq, s);
        if (!tup) goto cleanup;
        PyObject *k_obj = NULL, *c_obj = NULL, *r_obj = NULL;
        long cam_slot = 0, sign = 0;
        if (!PyArg_ParseTuple(tup, "lOOl|O", &cam_slot, &k_obj, &c_obj, &sign, &r_obj)) {
            Py_DECREF(tup);
            goto cleanup;
        }
        events[s].cam_slot = (int)cam_slot;
        events[s].sign = (int)sign;
        int rc1 = PyObject_GetBuffer(k_obj, &events[s].k_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS);
        int rc2 = (rc1 == 0) ? PyObject_GetBuffer(c_obj, &events[s].c_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) : -1;
        int rc3 = 0;
        if (rc1 == 0 && rc2 == 0 && with_rgb && r_obj != NULL && r_obj != Py_None) {
            rc3 = PyObject_GetBuffer(r_obj, &events[s].r_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS);
            if (rc3 == 0) {
                events[s].has_rgb = 1;
                events[s].rgb = (const float *)events[s].r_buf.buf;
            }
        }
        Py_DECREF(tup);
        if (rc1 < 0 || rc2 < 0 || rc3 < 0) goto cleanup;
        events[s].keys = (const int64_t *)events[s].k_buf.buf;
        events[s].centroids = (const double *)events[s].c_buf.buf;
        events[s].n = events[s].k_buf.len / (3 * (Py_ssize_t)sizeof(int64_t));
        total_keys += events[s].n;
    }

    ThreadScratch *sc = get_scratch();
    if (!sc || ensure_p_idx(sc, (size_t)(total_keys > 0 ? total_keys : 1)) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }
    int32_t *all_ids = sc->p_idx;
    old_ids = all_ids;
    Py_ssize_t id_off = v_old;
    for (Py_ssize_t s = 0; s < n_staged; s++) {
        events[s].ids = all_ids + id_off;
        id_off += events[s].n;
    }

    /* Compute global key bounds */
    int64_t min0 = 0, min1 = 0, min2 = 0, max0 = 0, max1 = 0, max2 = 0;
    int have_any = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < v_old; i++) {
        int64_t k0 = old_keys[3 * i + 0], k1 = old_keys[3 * i + 1], k2 = old_keys[3 * i + 2];
        if (!have_any) {
            min0 = max0 = k0; min1 = max1 = k1; min2 = max2 = k2; have_any = 1;
        } else {
            if (k0 < min0) min0 = k0; else if (k0 > max0) max0 = k0;
            if (k1 < min1) min1 = k1; else if (k1 > max1) max1 = k1;
            if (k2 < min2) min2 = k2; else if (k2 > max2) max2 = k2;
        }
    }
    for (Py_ssize_t s = 0; s < n_staged; s++) {
        const int64_t *sk = events[s].keys;
        Py_ssize_t sn = events[s].n;
        for (Py_ssize_t i = 0; i < sn; i++) {
            int64_t k0 = sk[3 * i + 0], k1 = sk[3 * i + 1], k2 = sk[3 * i + 2];
            if (!have_any) {
                min0 = max0 = k0; min1 = max1 = k1; min2 = max2 = k2; have_any = 1;
            } else {
                if (k0 < min0) min0 = k0; else if (k0 > max0) max0 = k0;
                if (k1 < min1) min1 = k1; else if (k1 > max1) max1 = k1;
                if (k2 < min2) min2 = k2; else if (k2 > max2) max2 = k2;
            }
        }
    }
    Py_END_ALLOW_THREADS

    __int128 d0 = (__int128)max0 - (__int128)min0 + 1;
    __int128 d1 = (__int128)max1 - (__int128)min1 + 1;
    __int128 d2 = (__int128)max2 - (__int128)min2 + 1;
    int use_grid = 0;
    size_t total_span = 0;
    if (have_any && d0 <= DIRECT_GRID_MAX && d1 <= DIRECT_GRID_MAX && d2 <= DIRECT_GRID_MAX) {
        __int128 prod = d0 * d1 * d2;
        if (prod > 0 && prod <= DIRECT_GRID_MAX && prod <= (__int128)16 * (__int128)(total_keys > 0 ? total_keys : 1)) {
            use_grid = 1;
            total_span = (size_t)prod;
        }
    }

    Py_ssize_t u_count = 0;

    if (use_grid) {
        if (ensure_grid(sc, total_span) < 0) { PyErr_NoMemory(); goto cleanup; }
        int32_t *grid = sc->grid;
        int64_t d0_i = (int64_t)d0, s1 = (int64_t)d1, s2 = (int64_t)d2, s12 = s1 * s2;

        Py_BEGIN_ALLOW_THREADS
        memset(grid, 0, total_span * sizeof(int32_t));
        for (Py_ssize_t i = 0; i < v_old; i++) {
            int32_t p = (int32_t)((old_keys[3 * i] - min0) * s12 + (old_keys[3 * i + 1] - min1) * s2 + (old_keys[3 * i + 2] - min2));
            old_ids[i] = p;
            grid[p] = 1;
        }
        for (Py_ssize_t s = 0; s < n_staged; s++) {
            const int64_t *sk = events[s].keys;
            int32_t *sids = events[s].ids;
            Py_ssize_t sn = events[s].n;
            for (Py_ssize_t i = 0; i < sn; i++) {
                int32_t p = (int32_t)((sk[3 * i] - min0) * s12 + (sk[3 * i + 1] - min1) * s2 + (sk[3 * i + 2] - min2));
                sids[i] = p;
                grid[p] = 1;
            }
        }
        for (size_t p = 0; p < total_span; p++) {
            if (grid[p]) u_count++;
        }
        Py_END_ALLOW_THREADS

        if (ensure_keys_tmp(sc, (size_t)(u_count > 0 ? u_count : 1)) < 0) { PyErr_NoMemory(); goto cleanup; }
        u_keys_buf = sc->keys_tmp;

        Py_BEGIN_ALLOW_THREADS
        Py_ssize_t u = 0;
        size_t p = 0;
        for (int64_t dx = 0; dx < d0_i; dx++) {
            int64_t kx_v = min0 + dx;
            for (int64_t dy = 0; dy < s1; dy++) {
                int64_t ky_v = min1 + dy;
                for (int64_t dz = 0; dz < s2; dz++, p++) {
                    if (grid[p]) {
                        u_keys_buf[u].kx = kx_v;
                        u_keys_buf[u].ky = ky_v;
                        u_keys_buf[u].kz = min2 + dz;
                        grid[p] = (int32_t)u++;
                    }
                }
            }
        }
        for (Py_ssize_t i = 0; i < v_old; i++) old_ids[i] = grid[old_ids[i]];
        for (Py_ssize_t s = 0; s < n_staged; s++) {
            int32_t *sids = events[s].ids;
            Py_ssize_t sn = events[s].n;
            for (Py_ssize_t i = 0; i < sn; i++) sids[i] = grid[sids[i]];
        }
        Py_END_ALLOW_THREADS
    } else {
        size_t cap = next_pow2((size_t)(total_keys > 0 ? total_keys : 1) * 2 + 16);
        if (ensure_kht(sc, cap) < 0) { PyErr_NoMemory(); goto cleanup; }
        KeyRankSlot *ht = sc->kht;
        size_t mask = cap - 1;

        Py_BEGIN_ALLOW_THREADS
        for (size_t i = 0; i < cap; i++) ht[i].used = 0;
        for (Py_ssize_t i = 0; i < v_old; i++) {
            int64_t k0 = old_keys[3 * i], k1 = old_keys[3 * i + 1], k2 = old_keys[3 * i + 2];
            size_t slot = (size_t)hash_key3(k0, k1, k2) & mask;
            while (ht[slot].used) {
                if (ht[slot].kx == k0 && ht[slot].ky == k1 && ht[slot].kz == k2) goto next_old_k;
                slot = (slot + 1) & mask;
            }
            ht[slot].used = 1; ht[slot].kx = k0; ht[slot].ky = k1; ht[slot].kz = k2;
            u_count++;
            next_old_k:;
        }
        for (Py_ssize_t s = 0; s < n_staged; s++) {
            const int64_t *sk = events[s].keys;
            Py_ssize_t sn = events[s].n;
            for (Py_ssize_t i = 0; i < sn; i++) {
                int64_t k0 = sk[3 * i], k1 = sk[3 * i + 1], k2 = sk[3 * i + 2];
                size_t slot = (size_t)hash_key3(k0, k1, k2) & mask;
                while (ht[slot].used) {
                    if (ht[slot].kx == k0 && ht[slot].ky == k1 && ht[slot].kz == k2) goto next_st_k;
                    slot = (slot + 1) & mask;
                }
                ht[slot].used = 1; ht[slot].kx = k0; ht[slot].ky = k1; ht[slot].kz = k2;
                u_count++;
                next_st_k:;
            }
        }
        Py_END_ALLOW_THREADS

        if (ensure_keys_tmp(sc, (size_t)(u_count > 0 ? u_count : 1)) < 0) { PyErr_NoMemory(); goto cleanup; }
        u_keys_buf = sc->keys_tmp;

        Py_BEGIN_ALLOW_THREADS
        Py_ssize_t u = 0;
        for (size_t i = 0; i < cap; i++) {
            if (ht[i].used) {
                u_keys_buf[u].kx = ht[i].kx;
                u_keys_buf[u].ky = ht[i].ky;
                u_keys_buf[u].kz = ht[i].kz;
                u++;
            }
        }
        qsort(u_keys_buf, (size_t)u_count, sizeof(Key3), cmp_key3_struct);
        for (Py_ssize_t j = 0; j < u_count; j++) {
            int64_t k0 = u_keys_buf[j].kx, k1 = u_keys_buf[j].ky, k2 = u_keys_buf[j].kz;
            size_t slot = (size_t)hash_key3(k0, k1, k2) & mask;
            while (!(ht[slot].kx == k0 && ht[slot].ky == k1 && ht[slot].kz == k2)) slot = (slot + 1) & mask;
            ht[slot].rank = (int32_t)j;
        }
        for (Py_ssize_t i = 0; i < v_old; i++) {
            int64_t k0 = old_keys[3 * i], k1 = old_keys[3 * i + 1], k2 = old_keys[3 * i + 2];
            size_t slot = (size_t)hash_key3(k0, k1, k2) & mask;
            while (!(ht[slot].kx == k0 && ht[slot].ky == k1 && ht[slot].kz == k2)) slot = (slot + 1) & mask;
            old_ids[i] = ht[slot].rank;
        }
        for (Py_ssize_t s = 0; s < n_staged; s++) {
            const int64_t *sk = events[s].keys;
            int32_t *sids = events[s].ids;
            Py_ssize_t sn = events[s].n;
            for (Py_ssize_t i = 0; i < sn; i++) {
                int64_t k0 = sk[3 * i], k1 = sk[3 * i + 1], k2 = sk[3 * i + 2];
                size_t slot = (size_t)hash_key3(k0, k1, k2) & mask;
                while (!(ht[slot].kx == k0 && ht[slot].ky == k1 && ht[slot].kz == k2)) slot = (slot + 1) & mask;
                sids[i] = ht[slot].rank;
            }
        }
        Py_END_ALLOW_THREADS
    }

    size_t u_pos = (size_t)(u_count > 0 ? u_count : 1);
    size_t cam_cells = (num_cams > 0 && u_count > 0) ? ((size_t)num_cams * (size_t)u_count) : 1;
    if (ensure_dbl_tmp(sc, u_pos * 3) < 0 ||
        ensure_counts(sc, u_pos) < 0 ||
        ensure_i64_tmp(sc, cam_cells) < 0 ||
        ensure_marks(sc, (size_t)(num_cams > 0 ? num_cams : 1)) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }
    if (with_rgb) {
        rgb_w = (double *)calloc(u_pos * 3, sizeof(double));
        if (!rgb_w) { PyErr_NoMemory(); goto cleanup; }
    }
    double *sums_w = sc->dbl_tmp;
    int64_t *sup_w = sc->counts;
    int64_t *cam_w = sc->i64_tmp;
    uint8_t *cam_active = sc->marks;

    Py_ssize_t resulting_count = 0;
    Py_ssize_t active_cam_count = 0;
    Py_BEGIN_ALLOW_THREADS
    memset(sums_w, 0, u_pos * 3 * sizeof(double));
    memset(sup_w, 0, u_pos * sizeof(int64_t));
    memset(cam_w, 0, cam_cells * sizeof(int64_t));
    if (num_cams > 0) memset(cam_active, 0, (size_t)num_cams);

    for (Py_ssize_t i = 0; i < v_old; i++) {
        int32_t u = old_ids[i];
        sums_w[3 * u + 0] = old_sums[3 * i + 0];
        sums_w[3 * u + 1] = old_sums[3 * i + 1];
        sums_w[3 * u + 2] = old_sums[3 * i + 2];
        if (rgb_w && i < old_rgb_rows) {
            rgb_w[3 * u + 0] = (double)old_rgb[3 * i + 0];
            rgb_w[3 * u + 1] = (double)old_rgb[3 * i + 1];
            rgb_w[3 * u + 2] = (double)old_rgb[3 * i + 2];
        }
        sup_w[u] = old_sup[i];
        for (Py_ssize_t c = 0; c < num_old_cams; c++) {
            cam_w[c * u_count + u] = old_cam_ptrs[c][i];
        }
    }
    for (Py_ssize_t s = 0; s < n_staged; s++) {
        int c = events[s].cam_slot;
        double dsgn = (double)events[s].sign;
        int64_t isgn = (int64_t)events[s].sign;
        const int32_t *sids = events[s].ids;
        const double *scent = events[s].centroids;
        const float *srgb = events[s].has_rgb ? events[s].rgb : NULL;
        Py_ssize_t sn = events[s].n;
        int64_t *cam_row = cam_w + (Py_ssize_t)c * u_count;
        for (Py_ssize_t k = 0; k < sn; k++) {
            int32_t u = sids[k];
            sums_w[3 * u + 0] += dsgn * scent[3 * k + 0];
            sums_w[3 * u + 1] += dsgn * scent[3 * k + 1];
            sums_w[3 * u + 2] += dsgn * scent[3 * k + 2];
            if (rgb_w && srgb) {
                rgb_w[3 * u + 0] += dsgn * (double)srgb[3 * k + 0];
                rgb_w[3 * u + 1] += dsgn * (double)srgb[3 * k + 1];
                rgb_w[3 * u + 2] += dsgn * (double)srgb[3 * k + 2];
            }
            sup_w[u] += isgn;
            cam_row[u] += isgn;
            if (sup_w[u] == 0) {
                sums_w[3 * u + 0] = 0.0;
                sums_w[3 * u + 1] = 0.0;
                sums_w[3 * u + 2] = 0.0;
                if (rgb_w) {
                    rgb_w[3 * u + 0] = 0.0;
                    rgb_w[3 * u + 1] = 0.0;
                    rgb_w[3 * u + 2] = 0.0;
                }
            }
        }
    }
    for (Py_ssize_t u = 0; u < u_count; u++) {
        if (sup_w[u] > 0) resulting_count++;
    }
    for (Py_ssize_t c = 0; c < num_cams; c++) {
        const int64_t *cam_row = cam_w + c * u_count;
        for (Py_ssize_t u = 0; u < u_count; u++) {
            if (sup_w[u] > 0 && cam_row[u] != 0) {
                cam_active[c] = 1;
                active_cam_count++;
                break;
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (resulting_count > max_voxels) {
        PyErr_SetString(PyExc_ValueError, "surface voxel budget exceeded");
        goto cleanup;
    }

    int64_t *out_k = NULL;
    double *out_s = NULL;
    int64_t *out_sup = NULL;
    int64_t *out_cam = NULL;
    float *out_rgb = NULL;
    PyObject *nk_arr = alloc_fast_2d(resulting_count, 3, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_k);
    PyObject *ns_arr = nk_arr ? alloc_fast_2d(resulting_count, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out_s) : NULL;
    PyObject *nsup_arr = ns_arr ? alloc_fast_1d(resulting_count, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_sup) : NULL;
    PyObject *ncam_arr = nsup_arr ? alloc_fast_1d(resulting_count, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_cam) : NULL;
    PyObject *nrgb_arr = (ncam_arr && with_rgb) ? alloc_fast_2d(resulting_count, 3, (Py_ssize_t)sizeof(float), g_np_float32, (void **)&out_rgb) : NULL;
    PyObject *kept_slots = PyTuple_New(active_cam_count);
    PyObject *kept_arrs = PyTuple_New(active_cam_count);

    int64_t *stack_out_cam_ptrs[16];
    Py_ssize_t stack_active_slots[16];
    int64_t **out_cam_ptrs = NULL;
    Py_ssize_t *active_slots = NULL;
    if (active_cam_count > 0) {
        if (active_cam_count <= 16) {
            out_cam_ptrs = stack_out_cam_ptrs;
            active_slots = stack_active_slots;
        } else {
            out_cam_ptrs = (int64_t **)malloc((size_t)active_cam_count * sizeof(int64_t *));
            active_slots = (Py_ssize_t *)malloc((size_t)active_cam_count * sizeof(Py_ssize_t));
            if (!out_cam_ptrs || !active_slots) {
                Py_XDECREF(nk_arr); Py_XDECREF(ns_arr); Py_XDECREF(nsup_arr); Py_XDECREF(ncam_arr); Py_XDECREF(nrgb_arr);
                Py_XDECREF(kept_slots); Py_XDECREF(kept_arrs);
                free(out_cam_ptrs); free(active_slots);
                PyErr_NoMemory();
                goto cleanup;
            }
        }
    }

    if (!nk_arr || !ns_arr || !nsup_arr || !ncam_arr || (with_rgb && !nrgb_arr) || !kept_slots || !kept_arrs) {
        Py_XDECREF(nk_arr); Py_XDECREF(ns_arr); Py_XDECREF(nsup_arr); Py_XDECREF(ncam_arr); Py_XDECREF(nrgb_arr);
        Py_XDECREF(kept_slots); Py_XDECREF(kept_arrs);
        if (active_cam_count > 16) { free(out_cam_ptrs); free(active_slots); }
        goto cleanup;
    }

    int cam_alloc_ok = 1;
    Py_ssize_t a_idx = 0;
    for (Py_ssize_t c = 0; c < num_cams; c++) {
        if (!cam_active[c]) continue;
        PyObject *slot_int = PyLong_FromSsize_t(c);
        int64_t *cptr = NULL;
        PyObject *carr = alloc_fast_1d(resulting_count, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&cptr);
        if (!slot_int || !carr) {
            Py_XDECREF(slot_int); Py_XDECREF(carr);
            cam_alloc_ok = 0;
            break;
        }
        PyTuple_SET_ITEM(kept_slots, a_idx, slot_int);
        PyTuple_SET_ITEM(kept_arrs, a_idx, carr);
        out_cam_ptrs[a_idx] = cptr;
        active_slots[a_idx] = c;
        a_idx++;
    }
    if (!cam_alloc_ok) {
        Py_DECREF(nk_arr); Py_DECREF(ns_arr); Py_DECREF(nsup_arr); Py_DECREF(ncam_arr); Py_XDECREF(nrgb_arr);
        Py_DECREF(kept_slots); Py_DECREF(kept_arrs);
        if (active_cam_count > 16) { free(out_cam_ptrs); free(active_slots); }
        goto cleanup;
    }

    int64_t s_min = 0, s_max = 0, c_min = 0, c_max = 0;

    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t j = 0;
    for (Py_ssize_t u = 0; u < u_count; u++) {
        int64_t sup = sup_w[u];
        if (sup <= 0) continue;
        out_k[3 * j + 0] = u_keys_buf[u].kx;
        out_k[3 * j + 1] = u_keys_buf[u].ky;
        out_k[3 * j + 2] = u_keys_buf[u].kz;
        out_s[3 * j + 0] = sums_w[3 * u + 0];
        out_s[3 * j + 1] = sums_w[3 * u + 1];
        out_s[3 * j + 2] = sums_w[3 * u + 2];
        if (out_rgb) {
            out_rgb[3 * j + 0] = (float)rgb_w[3 * u + 0];
            out_rgb[3 * j + 1] = (float)rgb_w[3 * u + 1];
            out_rgb[3 * j + 2] = (float)rgb_w[3 * u + 2];
        }
        out_sup[j] = sup;
        int64_t cam_cnt = 0;
        for (Py_ssize_t k = 0; k < active_cam_count; k++) {
            int64_t v = cam_w[active_slots[k] * u_count + u];
            out_cam_ptrs[k][j] = v;
            if (v > 0) cam_cnt++;
        }
        out_cam[j] = cam_cnt;
        if (j == 0) {
            s_min = s_max = sup;
            c_min = c_max = cam_cnt;
        } else {
            if (sup < s_min) s_min = sup; else if (sup > s_max) s_max = sup;
            if (cam_cnt < c_min) c_min = cam_cnt; else if (cam_cnt > c_max) c_max = cam_cnt;
        }
        j++;
    }
    Py_END_ALLOW_THREADS

    if (active_cam_count > 16) { free(out_cam_ptrs); free(active_slots); }

    if (with_rgb) {
        ret_obj = Py_BuildValue("(OOOOOOLLLLO)", nk_arr, ns_arr, nsup_arr, ncam_arr, kept_slots, kept_arrs,
                                (long long)s_min, (long long)s_max, (long long)c_min, (long long)c_max, nrgb_arr);
    } else {
        ret_obj = Py_BuildValue("(OOOOOOLLLL)", nk_arr, ns_arr, nsup_arr, ncam_arr, kept_slots, kept_arrs,
                                (long long)s_min, (long long)s_max, (long long)c_min, (long long)c_max);
    }
    Py_DECREF(nk_arr); Py_DECREF(ns_arr); Py_DECREF(nsup_arr); Py_DECREF(ncam_arr); Py_XDECREF(nrgb_arr);
    Py_DECREF(kept_slots); Py_DECREF(kept_arrs);

cleanup:
    free(rgb_w);
    if (events) {
        for (Py_ssize_t s = 0; s < n_staged; s++) {
            if (events[s].k_buf.buf) PyBuffer_Release(&events[s].k_buf);
            if (events[s].c_buf.buf) PyBuffer_Release(&events[s].c_buf);
            if (events[s].has_rgb) PyBuffer_Release(&events[s].r_buf);
        }
        if (n_staged > 32) free(events);
    }
    if (old_cam_bufs) {
        for (Py_ssize_t c = 0; c < num_old_cams; c++) {
            if (old_cam_bufs[c].buf) PyBuffer_Release(&old_cam_bufs[c]);
        }
        if (num_old_cams > 16) {
            free(old_cam_bufs);
            free(old_cam_ptrs);
        }
    }
    if (got_old_rgb) PyBuffer_Release(&old_rgb_buf);
    if (got_old_k) PyBuffer_Release(&old_k_buf);
    if (got_old_s) PyBuffer_Release(&old_s_buf);
    if (got_old_sup) PyBuffer_Release(&old_sup_buf);
    return ret_obj;
}

/* ============================================================================
 * 3. snapshot_select & snapshot_uniform_fast
 * ============================================================================ */

typedef struct {
    Py_ssize_t idx;
    int64_t cam;
    int64_t sup;
} SnapSortItem;

static int cmp_snap_sort_desc(const void *a, const void *b) {
    const SnapSortItem *sa = (const SnapSortItem *)a;
    const SnapSortItem *sb = (const SnapSortItem *)b;
    if (sa->cam != sb->cam) return sa->cam > sb->cam ? -1 : 1;
    if (sa->sup != sb->sup) return sa->sup > sb->sup ? -1 : 1;
    return sa->idx < sb->idx ? -1 : (sa->idx > sb->idx ? 1 : 0);
}

static PyObject *py_snapshot_uniform_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 6 && nargs != 7) {
        PyErr_SetString(PyExc_TypeError, "snapshot_uniform_fast expects (sums, support, cameras, keys, k, include_keys[, rgb_sums])");
        return NULL;
    }
    PyObject *s_obj = args[0], *sup_obj = args[1], *cam_obj = args[2], *k_obj = args[3];
    Py_ssize_t k = PyLong_AsSsize_t(args[4]);
    if (k == -1 && PyErr_Occurred()) return NULL;
    int include_keys = PyObject_IsTrue(args[5]);
    if (include_keys < 0) return NULL;
    if (k < 0) k = 0;
    int with_rgb = (nargs == 7 && args[6] != Py_None);

    Py_buffer s_buf = {0}, sup_buf = {0}, cam_buf = {0}, k_buf = {0}, rgb_buf = {0};
    int got_s = 0, got_sup = 0, got_cam = 0, got_k = 0, got_rgb = 0;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(s_obj, &s_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_s = 1;
    if (PyObject_GetBuffer(sup_obj, &sup_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_sup = 1;
    if (PyObject_GetBuffer(cam_obj, &cam_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_cam = 1;
    if (include_keys) {
        if (PyObject_GetBuffer(k_obj, &k_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        got_k = 1;
    }
    if (with_rgb) {
        if (PyObject_GetBuffer(args[6], &rgb_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        got_rgb = 1;
    }

    Py_ssize_t n_avail = sup_buf.len / (Py_ssize_t)sizeof(int64_t);
    if (k > n_avail) k = n_avail;

    const double *sums = (const double *)s_buf.buf;
    const int64_t *sup = (const int64_t *)sup_buf.buf;
    const int64_t *cams = (const int64_t *)cam_buf.buf;
    const int64_t *keys = include_keys ? (const int64_t *)k_buf.buf : NULL;
    const float *rgb_sums = with_rgb ? (const float *)rgb_buf.buf : NULL;

    double *out_p = NULL;
    int64_t *out_sup = NULL;
    int64_t *out_cam = NULL;
    int64_t *out_k = NULL;
    uint8_t *out_rgb = NULL;
    PyObject *p_arr = alloc_fast_2d(k, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out_p);
    PyObject *sup_arr = p_arr ? alloc_fast_1d(k, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_sup) : NULL;
    PyObject *cam_arr = sup_arr ? alloc_fast_1d(k, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_cam) : NULL;
    PyObject *key_arr = (cam_arr && include_keys) ? alloc_fast_2d(k, 3, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_k) : NULL;
    PyObject *rgb_arr = (cam_arr && (!include_keys || key_arr) && with_rgb) ? alloc_fast_2d(k, 3, 1, g_np_uint8, (void **)&out_rgb) : NULL;
    if (!p_arr || !sup_arr || !cam_arr || (include_keys && !key_arr) || (with_rgb && !rgb_arr)) {
        Py_XDECREF(p_arr); Py_XDECREF(sup_arr); Py_XDECREF(cam_arr); Py_XDECREF(key_arr); Py_XDECREF(rgb_arr);
        goto cleanup;
    }

    if (k > 0) {
        Py_BEGIN_ALLOW_THREADS
        memcpy(out_sup, sup, (size_t)k * sizeof(int64_t));
        memcpy(out_cam, cams, (size_t)k * sizeof(int64_t));
        if (out_k) memcpy(out_k, keys, (size_t)k * 3 * sizeof(int64_t));
        for (Py_ssize_t i = 0; i < k; i++) {
            double sv_d = (double)sup[i];
            out_p[3 * i + 0] = sums[3 * i + 0] / sv_d;
            out_p[3 * i + 1] = sums[3 * i + 1] / sv_d;
            out_p[3 * i + 2] = sums[3 * i + 2] / sv_d;
            if (out_rgb) {
                double r0 = rint((double)rgb_sums[3 * i + 0] / sv_d);
                double r1 = rint((double)rgb_sums[3 * i + 1] / sv_d);
                double r2 = rint((double)rgb_sums[3 * i + 2] / sv_d);
                out_rgb[3 * i + 0] = (uint8_t)(r0 < 0.0 ? 0.0 : (r0 > 255.0 ? 255.0 : r0));
                out_rgb[3 * i + 1] = (uint8_t)(r1 < 0.0 ? 0.0 : (r1 > 255.0 ? 255.0 : r1));
                out_rgb[3 * i + 2] = (uint8_t)(r2 < 0.0 ? 0.0 : (r2 > 255.0 ? 255.0 : r2));
            }
        }
        Py_END_ALLOW_THREADS
    }

    if (include_keys && with_rgb) {
        ret_obj = PyTuple_Pack(5, p_arr, sup_arr, cam_arr, key_arr, rgb_arr);
    } else if (include_keys) {
        ret_obj = PyTuple_Pack(4, p_arr, sup_arr, cam_arr, key_arr);
    } else if (with_rgb) {
        ret_obj = PyTuple_Pack(4, p_arr, sup_arr, cam_arr, rgb_arr);
    } else {
        ret_obj = PyTuple_Pack(3, p_arr, sup_arr, cam_arr);
    }
    Py_DECREF(p_arr); Py_DECREF(sup_arr); Py_DECREF(cam_arr); Py_XDECREF(key_arr); Py_XDECREF(rgb_arr);

cleanup:
    if (got_rgb) PyBuffer_Release(&rgb_buf);
    if (got_s) PyBuffer_Release(&s_buf);
    if (got_sup) PyBuffer_Release(&sup_buf);
    if (got_cam) PyBuffer_Release(&cam_buf);
    if (got_k) PyBuffer_Release(&k_buf);
    return ret_obj;
}

static PyObject *py_snapshot_select(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *k_obj, *s_obj, *sup_obj, *cam_obj, *rgb_obj = NULL;
    int64_t min_support = 1, max_points = 1;
    int include_keys = 0;
    if (!PyArg_ParseTuple(args, "OOOOLLp|O", &k_obj, &s_obj, &sup_obj, &cam_obj,
                          &min_support, &max_points, &include_keys, &rgb_obj)) {
        return NULL;
    }
    int with_rgb = (rgb_obj != NULL && rgb_obj != Py_None);

    Py_buffer k_buf = {0}, s_buf = {0}, sup_buf = {0}, cam_buf = {0}, rgb_buf = {0};
    int got_k = 0, got_s = 0, got_sup = 0, got_cam = 0, got_rgb = 0;
    SnapSortItem *sort_items = NULL;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(k_obj, &k_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_k = 1;
    if (PyObject_GetBuffer(s_obj, &s_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_s = 1;
    if (PyObject_GetBuffer(sup_obj, &sup_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_sup = 1;
    if (PyObject_GetBuffer(cam_obj, &cam_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_cam = 1;
    if (with_rgb) {
        if (PyObject_GetBuffer(rgb_obj, &rgb_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        got_rgb = 1;
    }

    Py_ssize_t n = sup_buf.len / (Py_ssize_t)sizeof(int64_t);
    const int64_t *keys = (const int64_t *)k_buf.buf;
    const double *sums = (const double *)s_buf.buf;
    const int64_t *sup = (const int64_t *)sup_buf.buf;
    const int64_t *cams = (const int64_t *)cam_buf.buf;
    const float *rgb_sums = with_rgb ? (const float *)rgb_buf.buf : NULL;

    Py_ssize_t kept_count = 0;
    int64_t c_min = 0, c_max = 0, s_min = 0, s_max = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        int64_t sv = sup[i];
        if (sv >= min_support) {
            int64_t cv = cams[i];
            if (kept_count == 0) {
                c_min = c_max = cv;
                s_min = s_max = sv;
            } else {
                if (cv < c_min) c_min = cv; else if (cv > c_max) c_max = cv;
                if (sv < s_min) s_min = sv; else if (sv > s_max) s_max = sv;
            }
            kept_count++;
        }
    }
    Py_END_ALLOW_THREADS

    if (kept_count == 0) {
        PyObject *p_arr = alloc_fast_2d(0, 3, (Py_ssize_t)sizeof(double), g_np_float64, NULL);
        PyObject *sup_arr = alloc_fast_1d(0, (Py_ssize_t)sizeof(int64_t), g_np_int64, NULL);
        PyObject *cam_arr = alloc_fast_1d(0, (Py_ssize_t)sizeof(int64_t), g_np_int64, NULL);
        PyObject *key_arr = include_keys ? alloc_fast_2d(0, 3, (Py_ssize_t)sizeof(int64_t), g_np_int64, NULL) : NULL;
        PyObject *rgb_arr = with_rgb ? alloc_fast_2d(0, 3, 1, g_np_uint8, NULL) : NULL;
        if (!p_arr || !sup_arr || !cam_arr || (include_keys && !key_arr) || (with_rgb && !rgb_arr)) {
            Py_XDECREF(p_arr); Py_XDECREF(sup_arr); Py_XDECREF(cam_arr); Py_XDECREF(key_arr); Py_XDECREF(rgb_arr);
            goto cleanup;
        }
        if (include_keys && with_rgb) ret_obj = PyTuple_Pack(5, p_arr, sup_arr, cam_arr, key_arr, rgb_arr);
        else if (include_keys) ret_obj = PyTuple_Pack(4, p_arr, sup_arr, cam_arr, key_arr);
        else if (with_rgb) ret_obj = PyTuple_Pack(4, p_arr, sup_arr, cam_arr, rgb_arr);
        else ret_obj = PyTuple_Pack(3, p_arr, sup_arr, cam_arr);
        Py_DECREF(p_arr); Py_DECREF(sup_arr); Py_DECREF(cam_arr); Py_XDECREF(key_arr); Py_XDECREF(rgb_arr);
        goto cleanup;
    }

    Py_ssize_t out_n = kept_count > (Py_ssize_t)max_points ? (Py_ssize_t)max_points : kept_count;
    ThreadScratch *sc = get_scratch();
    if (!sc || ensure_idx_tmp(sc, (size_t)out_n) < 0) { PyErr_NoMemory(); goto cleanup; }
    Py_ssize_t *sel_indices = sc->idx_tmp;

    int oom = 0;
    Py_BEGIN_ALLOW_THREADS
    if (kept_count <= (Py_ssize_t)max_points || (c_min == c_max && s_min == s_max)) {
        Py_ssize_t j = 0;
        for (Py_ssize_t i = 0; i < n && j < out_n; i++) {
            if (sup[i] >= min_support) sel_indices[j++] = i;
        }
    } else {
        uint64_t span_c = (uint64_t)(c_max - c_min) + 1ULL;
        uint64_t span_s = (uint64_t)(s_max - s_min) + 1ULL;
        if (span_c <= 4096ULL && span_s <= 4096ULL && span_c * span_s <= 4096ULL) {
            size_t nb = (size_t)(span_c * span_s);
            Py_ssize_t hist[4096];
            Py_ssize_t offsets[4096];
            memset(hist, 0, nb * sizeof(Py_ssize_t));
            for (Py_ssize_t i = 0; i < n; i++) {
                if (sup[i] >= min_support) {
                    size_t b = (size_t)((uint64_t)(c_max - cams[i]) * span_s + (uint64_t)(s_max - sup[i]));
                    hist[b]++;
                }
            }
            Py_ssize_t cum = 0;
            size_t b_cut = nb - 1;
            Py_ssize_t rem_in_cut = 0;
            for (size_t b = 0; b < nb; b++) {
                offsets[b] = cum;
                if (cum + hist[b] >= out_n) {
                    b_cut = b;
                    rem_in_cut = out_n - cum;
                    break;
                }
                cum += hist[b];
            }
            for (Py_ssize_t i = 0; i < n; i++) {
                if (sup[i] >= min_support) {
                    size_t b = (size_t)((uint64_t)(c_max - cams[i]) * span_s + (uint64_t)(s_max - sup[i]));
                    if (b < b_cut) {
                        sel_indices[offsets[b]++] = i;
                    } else if (b == b_cut && rem_in_cut > 0) {
                        sel_indices[offsets[b]++] = i;
                        rem_in_cut--;
                    }
                }
            }
        } else {
            sort_items = (SnapSortItem *)malloc((size_t)kept_count * sizeof(SnapSortItem));
            if (!sort_items) {
                oom = 1;
            } else {
                Py_ssize_t j = 0;
                for (Py_ssize_t i = 0; i < n; i++) {
                    if (sup[i] >= min_support) {
                        sort_items[j].idx = i;
                        sort_items[j].cam = cams[i];
                        sort_items[j].sup = sup[i];
                        j++;
                    }
                }
                qsort(sort_items, (size_t)kept_count, sizeof(SnapSortItem), cmp_snap_sort_desc);
                for (Py_ssize_t k = 0; k < out_n; k++) sel_indices[k] = sort_items[k].idx;
                free(sort_items);
                sort_items = NULL;
            }
        }
    }
    Py_END_ALLOW_THREADS
    if (oom) { PyErr_NoMemory(); goto cleanup; }

    double *out_p = NULL;
    int64_t *out_sup = NULL;
    int64_t *out_cam = NULL;
    int64_t *out_k = NULL;
    uint8_t *out_rgb = NULL;
    PyObject *p_arr = alloc_fast_2d(out_n, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out_p);
    PyObject *sup_arr = p_arr ? alloc_fast_1d(out_n, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_sup) : NULL;
    PyObject *cam_arr = sup_arr ? alloc_fast_1d(out_n, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_cam) : NULL;
    PyObject *key_arr = (cam_arr && include_keys) ? alloc_fast_2d(out_n, 3, (Py_ssize_t)sizeof(int64_t), g_np_int64, (void **)&out_k) : NULL;
    PyObject *rgb_arr = (cam_arr && (!include_keys || key_arr) && with_rgb) ? alloc_fast_2d(out_n, 3, 1, g_np_uint8, (void **)&out_rgb) : NULL;
    if (!p_arr || !sup_arr || !cam_arr || (include_keys && !key_arr) || (with_rgb && !rgb_arr)) {
        Py_XDECREF(p_arr); Py_XDECREF(sup_arr); Py_XDECREF(cam_arr); Py_XDECREF(key_arr); Py_XDECREF(rgb_arr);
        goto cleanup;
    }

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t j = 0; j < out_n; j++) {
        Py_ssize_t idx = sel_indices[j];
        int64_t sv = sup[idx];
        double sv_d = (double)sv;
        out_p[3 * j + 0] = sums[3 * idx + 0] / sv_d;
        out_p[3 * j + 1] = sums[3 * idx + 1] / sv_d;
        out_p[3 * j + 2] = sums[3 * idx + 2] / sv_d;
        out_sup[j] = sv;
        out_cam[j] = cams[idx];
        if (out_k) {
            out_k[3 * j + 0] = keys[3 * idx + 0];
            out_k[3 * j + 1] = keys[3 * idx + 1];
            out_k[3 * j + 2] = keys[3 * idx + 2];
        }
        if (out_rgb) {
            double r0 = rint((double)rgb_sums[3 * idx + 0] / sv_d);
            double r1 = rint((double)rgb_sums[3 * idx + 1] / sv_d);
            double r2 = rint((double)rgb_sums[3 * idx + 2] / sv_d);
            out_rgb[3 * j + 0] = (uint8_t)(r0 < 0.0 ? 0.0 : (r0 > 255.0 ? 255.0 : r0));
            out_rgb[3 * j + 1] = (uint8_t)(r1 < 0.0 ? 0.0 : (r1 > 255.0 ? 255.0 : r1));
            out_rgb[3 * j + 2] = (uint8_t)(r2 < 0.0 ? 0.0 : (r2 > 255.0 ? 255.0 : r2));
        }
    }
    Py_END_ALLOW_THREADS

    if (include_keys && with_rgb) ret_obj = PyTuple_Pack(5, p_arr, sup_arr, cam_arr, key_arr, rgb_arr);
    else if (include_keys) ret_obj = PyTuple_Pack(4, p_arr, sup_arr, cam_arr, key_arr);
    else if (with_rgb) ret_obj = PyTuple_Pack(4, p_arr, sup_arr, cam_arr, rgb_arr);
    else ret_obj = PyTuple_Pack(3, p_arr, sup_arr, cam_arr);
    Py_DECREF(p_arr); Py_DECREF(sup_arr); Py_DECREF(cam_arr); Py_XDECREF(key_arr); Py_XDECREF(rgb_arr);

cleanup:
    free(sort_items);
    if (got_rgb) PyBuffer_Release(&rgb_buf);
    if (got_k) PyBuffer_Release(&k_buf);
    if (got_s) PyBuffer_Release(&s_buf);
    if (got_sup) PyBuffer_Release(&sup_buf);
    if (got_cam) PyBuffer_Release(&cam_buf);
    return ret_obj;
}

/* ============================================================================
 * 4. select_existing_points_shadow_fast
 * ============================================================================ */

static int cmp_key_with_idx(const void *a, const void *b) {
    const KeyWithIdx *ka = (const KeyWithIdx *)a;
    const KeyWithIdx *kb = (const KeyWithIdx *)b;
    int c = cmp_key3(ka->kx, ka->ky, ka->kz, kb->kx, kb->ky, kb->kz);
    if (c != 0) return c;
    return ka->orig_idx < kb->orig_idx ? -1 : (ka->orig_idx > kb->orig_idx ? 1 : 0);
}

static int cmp_prio_item(const void *a, const void *b) {
    const PrioItem *pa = (const PrioItem *)a;
    const PrioItem *pb = (const PrioItem *)b;
    if (pa->v != pb->v) return pa->v > pb->v ? -1 : 1;
    if (pa->c != pb->c) return pa->c > pb->c ? -1 : 1;
    return pa->lex_pos < pb->lex_pos ? -1 : (pa->lex_pos > pb->lex_pos ? 1 : 0);
}

#if ES_HAVE_X86_RUNTIME_SIMD
__attribute__((target("avx512f,avx512dq,avx512vl")))
static Py_ssize_t verify_shadow_keys_avx512(
    const double *xyz, const int64_t *keys, Py_ssize_t total_f,
    double voxel_size_m, double bound_2p62, int *coord_ok)
{
    Py_ssize_t m = 0;
    __m512d v_vs = _mm512_set1_pd(voxel_size_m);
    __m512d v_bound = _mm512_set1_pd(bound_2p62);
    __m512d v_inf = _mm512_set1_pd(HUGE_VAL);
    __m512i v_abs_mask = _mm512_set1_epi64(0x7fffffffffffffffLL);
    for (; m + 8 <= total_f; m += 8) {
        __m512d vp = _mm512_loadu_pd(xyz + m);
        __m512d vp_abs = _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vp), v_abs_mask));
        if (_mm512_cmp_pd_mask(vp_abs, v_inf, _CMP_LT_OQ) != 0xFF) { *coord_ok = 0; break; }
        __m512d vs = _mm512_floor_pd(_mm512_div_pd(vp, v_vs));
        __m512d vs_abs = _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vs), v_abs_mask));
        if (_mm512_cmp_pd_mask(vs_abs, v_bound, _CMP_LT_OQ) != 0xFF) { *coord_ok = 0; break; }
        __m512i vk_calc = _mm512_cvttpd_epi64(vs);
        __m512i vk_in = _mm512_loadu_si512((const __m512i *)(keys + m));
        if (_mm512_cmpeq_epi64_mask(vk_calc, vk_in) != 0xFF) { *coord_ok = 0; break; }
    }
    return m;
}

__attribute__((target("avx2")))
static Py_ssize_t verify_shadow_keys_avx2(
    const double *xyz, const int64_t *keys, Py_ssize_t total_f,
    double voxel_size_m, double bound_2p62, int *coord_ok)
{
    Py_ssize_t m = 0;
    __m256d v_vs = _mm256_set1_pd(voxel_size_m);
    __m256d v_bound = _mm256_set1_pd(bound_2p62);
    __m256d v_inf = _mm256_set1_pd(HUGE_VAL);
    __m256d v_abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
    double tmp_s[4];
    for (; m + 4 <= total_f; m += 4) {
        __m256d vp = _mm256_loadu_pd(xyz + m);
        __m256d vp_abs = _mm256_and_pd(vp, v_abs_mask);
        if (_mm256_movemask_pd(_mm256_cmp_pd(vp_abs, v_inf, _CMP_LT_OQ)) != 0xF) { *coord_ok = 0; break; }
        __m256d vs = _mm256_floor_pd(_mm256_div_pd(vp, v_vs));
        __m256d vs_abs = _mm256_and_pd(vs, v_abs_mask);
        if (_mm256_movemask_pd(_mm256_cmp_pd(vs_abs, v_bound, _CMP_LT_OQ)) != 0xF) { *coord_ok = 0; break; }
        _mm256_storeu_pd(tmp_s, vs);
        if ((int64_t)tmp_s[0] != keys[m + 0] || (int64_t)tmp_s[1] != keys[m + 1] ||
            (int64_t)tmp_s[2] != keys[m + 2] || (int64_t)tmp_s[3] != keys[m + 3]) {
            *coord_ok = 0; break;
        }
    }
    return m;
}
#endif

static PyObject *py_select_existing_points_shadow(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 8) {
        PyErr_SetString(PyExc_TypeError, "select_existing_points_shadow_fast expects 8 arguments");
        return NULL;
    }
    PyObject *xyz_obj = args[0], *cnt_obj = args[1], *view_obj = args[2], *keys_obj = args[3];
    Py_ssize_t budget = PyLong_AsSsize_t(args[4]);
    if (budget == -1 && PyErr_Occurred()) return NULL;
    int64_t min_coarse_support = PyLong_AsLongLong(args[5]);
    if (min_coarse_support == -1 && PyErr_Occurred()) return NULL;
    double voxel_size_m = PyFloat_AsDouble(args[6]);
    if (voxel_size_m == -1.0 && PyErr_Occurred()) return NULL;
    double coarse_cell_m = PyFloat_AsDouble(args[7]);
    if (coarse_cell_m == -1.0 && PyErr_Occurred()) return NULL;

    Py_buffer xyz_b = {0}, cnt_b = {0}, view_b = {0}, keys_b = {0};
    int got_xyz = 0, got_cnt = 0, got_view = 0, got_keys = 0;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(xyz_obj, &xyz_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_xyz = 1;
    if (PyObject_GetBuffer(cnt_obj, &cnt_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_cnt = 1;
    if (PyObject_GetBuffer(view_obj, &view_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_view = 1;
    if (PyObject_GetBuffer(keys_obj, &keys_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_keys = 1;

    Py_ssize_t n = cnt_b.len / (Py_ssize_t)sizeof(int64_t);
    const double *xyz = (const double *)xyz_b.buf;
    const int64_t *counts = (const int64_t *)cnt_b.buf;
    const int64_t *views = (const int64_t *)view_b.buf;
    const int64_t *keys = (const int64_t *)keys_b.buf;

    const double bound_2p62 = 4611686018427387904.0; /* 2.0^62 */
    int err = 0;
    int strictly_lex_sorted = 1;
    Py_ssize_t eligible_count = 0;
    int64_t c_min = 0, c_max = 0, v_min = 0, v_max = 0;

    Py_BEGIN_ALLOW_THREADS
    int coord_ok = 1;
    int attr_ok = 1;
    Py_ssize_t total_f = 3 * n;
    Py_ssize_t m = 0;

#if ES_HAVE_X86_RUNTIME_SIMD
    if (g_simd_level >= 2) {
        m = verify_shadow_keys_avx512(xyz, keys, total_f, voxel_size_m, bound_2p62, &coord_ok);
    } else if (g_simd_level == 1) {
        m = verify_shadow_keys_avx2(xyz, keys, total_f, voxel_size_m, bound_2p62, &coord_ok);
    }
#endif

    if (coord_ok) {
        for (; m < total_f; m++) {
            double x = xyz[m];
            if (!isfinite(x)) { coord_ok = 0; break; }
            double sx = floor(x / voxel_size_m);
            if (!isfinite(sx) || fabs(sx) >= bound_2p62 || (int64_t)sx != keys[m]) {
                coord_ok = 0; break;
            }
        }
    }

    if (coord_ok && n > 0) {
        c_min = c_max = counts[0];
        v_min = v_max = views[0];
        if (c_min < 2 || v_min < 1) {
            attr_ok = 0;
        } else {
            if (c_min >= min_coarse_support) eligible_count = 1;
            int64_t pk0 = keys[0], pk1 = keys[1], pk2 = keys[2];
            for (Py_ssize_t i = 1; i < n; i++) {
                int64_t cnt = counts[i], vw = views[i];
                if (cnt < 2 || vw < 1) { attr_ok = 0; break; }
                if (cnt >= min_coarse_support) eligible_count++;
                if (cnt < c_min) c_min = cnt; else if (cnt > c_max) c_max = cnt;
                if (vw < v_min) v_min = vw; else if (vw > v_max) v_max = vw;
                if (strictly_lex_sorted) {
                    int64_t k0 = keys[3 * i + 0], k1 = keys[3 * i + 1], k2 = keys[3 * i + 2];
                    if (!(k0 > pk0 || (k0 == pk0 && (k1 > pk1 || (k1 == pk1 && k2 > pk2))))) {
                        strictly_lex_sorted = 0;
                    }
                    pk0 = k0; pk1 = k1; pk2 = k2;
                }
            }
        }
    }

    if (!coord_ok || !attr_ok) {
        for (Py_ssize_t i = 0; i < n; i++) {
            if (!isfinite(xyz[3 * i + 0]) || !isfinite(xyz[3 * i + 1]) || !isfinite(xyz[3 * i + 2]) ||
                counts[i] < 2 || views[i] < 1) {
                err = 1; break;
            }
        }
        if (err == 0) err = 2;
    }
    Py_END_ALLOW_THREADS

    if (err == 1) { PyErr_SetString(PyExc_ValueError, "invalid supported snapshot arrays"); goto cleanup; }
    if (err == 2) { PyErr_SetString(PyExc_ValueError, "ambiguous voxel keys"); goto cleanup; }

    ThreadScratch *sc = get_scratch();
    if (!sc || ensure_idx_tmp(sc, (size_t)n * 2 + (size_t)(eligible_count > 0 ? eligible_count : 1)) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }
    Py_ssize_t *lex = sc->idx_tmp;
    Py_ssize_t *priority = sc->idx_tmp + n;
    Py_ssize_t *eligible = sc->idx_tmp + 2 * n;

    /* Build lex if not strictly sorted and verify no duplicate keys */
    if (!strictly_lex_sorted) {
        if (ensure_kwi(sc, (size_t)n) < 0) { PyErr_NoMemory(); goto cleanup; }
        KeyWithIdx *kwi = sc->kwi;
        int dup_err = 0;
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t i = 0; i < n; i++) {
            kwi[i].kx = keys[3 * i + 0];
            kwi[i].ky = keys[3 * i + 1];
            kwi[i].kz = keys[3 * i + 2];
            kwi[i].orig_idx = i;
        }
        qsort(kwi, (size_t)n, sizeof(KeyWithIdx), cmp_key_with_idx);
        for (Py_ssize_t i = 0; i < n; i++) {
            if (i > 0 && kwi[i].kx == kwi[i - 1].kx && kwi[i].ky == kwi[i - 1].ky && kwi[i].kz == kwi[i - 1].kz) {
                dup_err = 1; break;
            }
            lex[i] = kwi[i].orig_idx;
        }
        Py_END_ALLOW_THREADS
        if (dup_err) { PyErr_SetString(PyExc_ValueError, "ambiguous voxel keys"); goto cleanup; }
    }

    /* Early return for within_budget: returns lex[:n] */
    if (n <= budget) {
        Py_ssize_t *out_i = NULL;
        PyObject *idx_arr = alloc_fast_1d(n, (Py_ssize_t)sizeof(Py_ssize_t), g_np_intp, (void **)&out_i);
        if (!idx_arr) goto cleanup;
        Py_BEGIN_ALLOW_THREADS
        if (strictly_lex_sorted) {
            for (Py_ssize_t i = 0; i < n; i++) out_i[i] = i;
        } else {
            memcpy(out_i, lex, (size_t)n * sizeof(Py_ssize_t));
        }
        Py_END_ALLOW_THREADS
        PyObject *elig_obj = PyLong_FromSsize_t(eligible_count);
        if (!elig_obj) { Py_DECREF(idx_arr); goto cleanup; }
        ret_obj = PyTuple_Pack(4, g_str_owner_priority, elig_obj,
                               idx_arr, g_str_within_budget);
        Py_DECREF(elig_obj);
        Py_DECREF(idx_arr);
        goto cleanup;
    }

    /* Build priority array only when needed (homogeneous or bucketed or qsort) */
    int homogeneous = (c_min == c_max && v_min == v_max);
    if (!homogeneous) {
        uint64_t span_v = (uint64_t)(v_max - v_min) + 1ULL;
        uint64_t span_c = (uint64_t)(c_max - c_min) + 1ULL;
        if (span_v <= 4096ULL && span_c <= 4096ULL && span_v * span_c <= 4096ULL) {
            Py_BEGIN_ALLOW_THREADS
            size_t nb = (size_t)(span_v * span_c);
            Py_ssize_t hist[4096];
            Py_ssize_t offsets[4096];
            memset(hist, 0, nb * sizeof(Py_ssize_t));
            for (Py_ssize_t p = 0; p < n; p++) {
                Py_ssize_t idx = strictly_lex_sorted ? p : lex[p];
                size_t b = (size_t)((uint64_t)(v_max - views[idx]) * span_c + (uint64_t)(c_max - counts[idx]));
                hist[b]++;
            }
            Py_ssize_t cum = 0;
            for (size_t b = 0; b < nb; b++) { offsets[b] = cum; cum += hist[b]; }
            for (Py_ssize_t p = 0; p < n; p++) {
                Py_ssize_t idx = strictly_lex_sorted ? p : lex[p];
                size_t b = (size_t)((uint64_t)(v_max - views[idx]) * span_c + (uint64_t)(c_max - counts[idx]));
                priority[offsets[b]++] = idx;
            }
            Py_END_ALLOW_THREADS
        } else {
            if (ensure_pitems(sc, (size_t)n) < 0) { PyErr_NoMemory(); goto cleanup; }
            PrioItem *pitems = sc->pitems;
            Py_BEGIN_ALLOW_THREADS
            for (Py_ssize_t p = 0; p < n; p++) {
                Py_ssize_t idx = strictly_lex_sorted ? p : lex[p];
                pitems[p].v = views[idx];
                pitems[p].c = counts[idx];
                pitems[p].lex_pos = p;
                pitems[p].orig_idx = idx;
            }
            qsort(pitems, (size_t)n, sizeof(PrioItem), cmp_prio_item);
            for (Py_ssize_t p = 0; p < n; p++) priority[p] = pitems[p].orig_idx;
            Py_END_ALLOW_THREADS
        }
    }

    if (eligible_count < budget) {
        Py_ssize_t *out_i = NULL;
        PyObject *idx_arr = alloc_fast_1d(budget, (Py_ssize_t)sizeof(Py_ssize_t), g_np_intp, (void **)&out_i);
        if (!idx_arr) goto cleanup;
        Py_BEGIN_ALLOW_THREADS
        if (homogeneous) {
            if (strictly_lex_sorted) {
                for (Py_ssize_t i = 0; i < budget; i++) out_i[i] = i;
            } else {
                memcpy(out_i, lex, (size_t)budget * sizeof(Py_ssize_t));
            }
        } else {
            memcpy(out_i, priority, (size_t)budget * sizeof(Py_ssize_t));
        }
        Py_END_ALLOW_THREADS
        PyObject *elig_obj = PyLong_FromSsize_t(eligible_count);
        if (!elig_obj) { Py_DECREF(idx_arr); goto cleanup; }
        ret_obj = PyTuple_Pack(4, g_str_owner_priority, elig_obj, idx_arr, g_str_insufficient_coarse);
        Py_DECREF(elig_obj);
        Py_DECREF(idx_arr);
        goto cleanup;
    }

    /* coarse_shadow path */
    if (ensure_keys_tmp(sc, (size_t)eligible_count) < 0) { PyErr_NoMemory(); goto cleanup; }
    Key3 *ckeys = sc->keys_tmp;
    int coarse_err = 0;
    int64_t cmin0 = 0, cmin1 = 0, cmin2 = 0, cmax0 = 0, cmax1 = 0, cmax2 = 0;
    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t e_idx = 0;
    for (Py_ssize_t p = 0; p < n; p++) {
        Py_ssize_t idx = homogeneous ? (strictly_lex_sorted ? p : lex[p]) : priority[p];
        if (counts[idx] >= min_coarse_support) {
            double sc0 = floor(xyz[3 * idx + 0] / coarse_cell_m);
            double sc1 = floor(xyz[3 * idx + 1] / coarse_cell_m);
            double sc2 = floor(xyz[3 * idx + 2] / coarse_cell_m);
            if (!isfinite(sc0) || !isfinite(sc1) || !isfinite(sc2) ||
                fabs(sc0) >= bound_2p62 || fabs(sc1) >= bound_2p62 || fabs(sc2) >= bound_2p62) {
                coarse_err = 1; break;
            }
            int64_t cx0 = (int64_t)sc0, cx1 = (int64_t)sc1, cx2 = (int64_t)sc2;
            if (e_idx == 0) {
                cmin0 = cmax0 = cx0; cmin1 = cmax1 = cx1; cmin2 = cmax2 = cx2;
            } else {
                if (cx0 < cmin0) cmin0 = cx0; else if (cx0 > cmax0) cmax0 = cx0;
                if (cx1 < cmin1) cmin1 = cx1; else if (cx1 > cmax1) cmax1 = cx1;
                if (cx2 < cmin2) cmin2 = cx2; else if (cx2 > cmax2) cmax2 = cx2;
            }
            ckeys[e_idx].kx = cx0;
            ckeys[e_idx].ky = cx1;
            ckeys[e_idx].kz = cx2;
            eligible[e_idx++] = idx;
        }
    }
    Py_END_ALLOW_THREADS
    if (coarse_err) { PyErr_SetString(PyExc_ValueError, "invalid coarse keys"); goto cleanup; }

    __int128 cd0 = (__int128)cmax0 - (__int128)cmin0 + 1;
    __int128 cd1 = (__int128)cmax1 - (__int128)cmin1 + 1;
    __int128 cd2 = (__int128)cmax2 - (__int128)cmin2 + 1;
    int use_coarse_grid = 0;
    size_t cspan = 0;
    if (cd0 <= DIRECT_GRID_MAX && cd1 <= DIRECT_GRID_MAX && cd2 <= DIRECT_GRID_MAX) {
        __int128 prod = cd0 * cd1 * cd2;
        if (prod > 0 && prod <= DIRECT_GRID_MAX) {
            use_coarse_grid = 1;
            cspan = (size_t)prod;
        }
    }

    if (ensure_marks(sc, (size_t)eligible_count + (use_coarse_grid ? cspan : 0)) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }
    uint8_t *is_rep = sc->marks;
    uint8_t *seen = sc->marks + eligible_count;

    Py_ssize_t *out_i = NULL;
    PyObject *idx_arr = alloc_fast_1d(budget, (Py_ssize_t)sizeof(Py_ssize_t), g_np_intp, (void **)&out_i);
    if (!idx_arr) goto cleanup;

    if (use_coarse_grid) {
        int64_t s1 = (int64_t)cd1, s2 = (int64_t)cd2, s12 = s1 * s2;
        Py_BEGIN_ALLOW_THREADS
        memset(is_rep, 0, (size_t)eligible_count);
        memset(seen, 0, cspan);
        Py_ssize_t rep_cnt = 0;
        for (Py_ssize_t e = 0; e < eligible_count && rep_cnt < budget; e++) {
            int64_t cx0 = ckeys[e].kx, cx1 = ckeys[e].ky, cx2 = ckeys[e].kz;
            size_t p = (size_t)((cx0 - cmin0) * s12 + (cx1 - cmin1) * s2 + (cx2 - cmin2));
            if (!seen[p]) {
                seen[p] = 1;
                is_rep[e] = 1;
                out_i[rep_cnt++] = eligible[e];
            }
        }
        for (Py_ssize_t e = 0; e < eligible_count && rep_cnt < budget; e++) {
            if (!is_rep[e]) out_i[rep_cnt++] = eligible[e];
        }
        Py_END_ALLOW_THREADS
    } else {
        size_t cap = next_pow2((size_t)eligible_count * 2 + 16);
        if (ensure_cht(sc, cap) < 0) {
            Py_DECREF(idx_arr);
            PyErr_NoMemory();
            goto cleanup;
        }
        CoarseSlot *cht = sc->cht;
        size_t mask = cap - 1;
        Py_BEGIN_ALLOW_THREADS
        memset(is_rep, 0, (size_t)eligible_count);
        for (size_t s_idx = 0; s_idx < cap; s_idx++) cht[s_idx].used = 0;
        Py_ssize_t rep_cnt = 0;
        for (Py_ssize_t e = 0; e < eligible_count && rep_cnt < budget; e++) {
            int64_t cx0 = ckeys[e].kx, cx1 = ckeys[e].ky, cx2 = ckeys[e].kz;
            size_t slot = (size_t)hash_key3(cx0, cx1, cx2) & mask;
            int found = 0;
            while (cht[slot].used) {
                if (cht[slot].cx == cx0 && cht[slot].cy == cx1 && cht[slot].cz == cx2) {
                    found = 1; break;
                }
                slot = (slot + 1) & mask;
            }
            if (!found) {
                cht[slot].used = 1;
                cht[slot].cx = cx0; cht[slot].cy = cx1; cht[slot].cz = cx2;
                is_rep[e] = 1;
                out_i[rep_cnt++] = eligible[e];
            }
        }
        for (Py_ssize_t e = 0; e < eligible_count && rep_cnt < budget; e++) {
            if (!is_rep[e]) out_i[rep_cnt++] = eligible[e];
        }
        Py_END_ALLOW_THREADS
    }

    PyObject *elig_obj = PyLong_FromSsize_t(eligible_count);
    if (!elig_obj) { Py_DECREF(idx_arr); goto cleanup; }
    ret_obj = PyTuple_Pack(4, g_str_coarse_shadow, elig_obj, idx_arr, Py_None);
    Py_DECREF(elig_obj);
    Py_DECREF(idx_arr);

cleanup:
    if (got_xyz) PyBuffer_Release(&xyz_b);
    if (got_cnt) PyBuffer_Release(&cnt_b);
    if (got_view) PyBuffer_Release(&view_b);
    if (got_keys) PyBuffer_Release(&keys_b);
    return ret_obj;
}

/* ============================================================================
 * 5. RLE Interval Algebra & Codec Kernels
 * ============================================================================ */

typedef struct {
    int64_t start;
    int64_t end;
} RleRun;

/* Parse and validate payload matching _validated_runs() exactly.
 * Returns 0 on success, -1 on Python exception (ValueError raised),
 * or 1 if fallback to Python is needed (e.g., gigantic non-int64 integers).
 */
static int parse_validated_runs(PyObject *payload, int64_t *h_out, int64_t *w_out,
                                RleRun **runs_out, Py_ssize_t *n_runs_out,
                                RleRun *stack_buf, Py_ssize_t stack_cap) {
    if (!PyDict_Check(payload)) {
        PyErr_SetString(PyExc_ValueError, "RLE payload must be a dict");
        return -1;
    }
    PyObject *size = PyDict_GetItem(payload, g_str_size);
    PyObject *counts = PyDict_GetItem(payload, g_str_counts);
    if (!size || !PyList_Check(size) || PyList_GET_SIZE(size) != 2 ||
        !counts || !PyList_Check(counts) || (PyList_GET_SIZE(counts) & 1) != 0) {
        PyErr_SetString(PyExc_ValueError, "RLE payload must contain valid 'size' [H, W] and even 'counts'");
        return -1;
    }
    PyObject *h_obj = PyList_GET_ITEM(size, 0);
    PyObject *w_obj = PyList_GET_ITEM(size, 1);
    if (!PyLong_CheckExact(h_obj) || !PyLong_CheckExact(w_obj)) {
        PyErr_SetString(PyExc_ValueError, "RLE payload must contain valid 'size' [H, W] and even 'counts'");
        return -1;
    }
    int overflow = 0;
    long long h = PyLong_AsLongLongAndOverflow(h_obj, &overflow);
    if (overflow) { PyErr_Clear(); return 1; }
    long long w = PyLong_AsLongLongAndOverflow(w_obj, &overflow);
    if (overflow) { PyErr_Clear(); return 1; }
    if (h < 0 || w < 0) {
        PyErr_SetString(PyExc_ValueError, "RLE payload must contain valid 'size' [H, W] and even 'counts'");
        return -1;
    }
    __int128 limit128 = (__int128)h * (__int128)w;
    if (limit128 > INT64_MAX) return 1;
    int64_t limit = (int64_t)limit128;

    Py_ssize_t c_len = PyList_GET_SIZE(counts);
    Py_ssize_t max_runs = c_len >> 1;
    RleRun *runs = max_runs <= stack_cap ? stack_buf : (RleRun *)malloc((size_t)(max_runs > 0 ? max_runs : 1) * sizeof(RleRun));
    if (!runs) { PyErr_NoMemory(); return -1; }

    Py_ssize_t n_runs = 0;
    int64_t prev_end = -1;
    for (Py_ssize_t i = 0; i < c_len; i += 2) {
        PyObject *s_obj = PyList_GET_ITEM(counts, i);
        PyObject *l_obj = PyList_GET_ITEM(counts, i + 1);
        if (!PyLong_CheckExact(s_obj) || !PyLong_CheckExact(l_obj)) {
            if (runs != stack_buf) free(runs);
            PyErr_SetString(PyExc_ValueError, "RLE run start and length must be positive integers");
            return -1;
        }
        long long start = PyLong_AsLongLongAndOverflow(s_obj, &overflow);
        if (overflow) { PyErr_Clear(); if (runs != stack_buf) free(runs); return 1; }
        long long length = PyLong_AsLongLongAndOverflow(l_obj, &overflow);
        if (overflow) { PyErr_Clear(); if (runs != stack_buf) free(runs); return 1; }
        if (length <= 0) {
            if (runs != stack_buf) free(runs);
            PyErr_SetString(PyExc_ValueError, "RLE run start and length must be positive integers");
            return -1;
        }
        if (start < 0 || (__int128)start + (__int128)length > (__int128)limit) {
            if (runs != stack_buf) free(runs);
            PyErr_SetString(PyExc_ValueError, "RLE run exceeds image bounds");
            return -1;
        }
        int64_t end = (int64_t)start + (int64_t)length;
        if ((int64_t)start < prev_end) {
            if (runs != stack_buf) free(runs);
            PyErr_SetString(PyExc_ValueError, "RLE runs must be sorted and non-overlapping");
            return -1;
        }
        if ((int64_t)start == prev_end) {
            runs[n_runs - 1].end = end;
        } else {
            runs[n_runs].start = (int64_t)start;
            runs[n_runs].end = end;
            n_runs++;
        }
        prev_end = end;
    }

    *h_out = (int64_t)h;
    *w_out = (int64_t)w;
    *runs_out = runs;
    *n_runs_out = n_runs;
    return 0;
}

static PyObject *runs_to_rle_dict(int64_t height, int64_t width, const RleRun *runs, Py_ssize_t n_runs) {
    PyObject *dict = PyDict_New();
    PyObject *size_list = PyList_New(2);
    PyObject *counts_list = PyList_New(n_runs * 2);
    if (!dict || !size_list || !counts_list) {
        Py_XDECREF(dict); Py_XDECREF(size_list); Py_XDECREF(counts_list);
        return NULL;
    }
    PyObject *h_obj = PyLong_FromLongLong(height);
    PyObject *w_obj = PyLong_FromLongLong(width);
    if (!h_obj || !w_obj) {
        Py_XDECREF(h_obj); Py_XDECREF(w_obj);
        Py_DECREF(dict); Py_DECREF(size_list); Py_DECREF(counts_list);
        return NULL;
    }
    PyList_SET_ITEM(size_list, 0, h_obj);
    PyList_SET_ITEM(size_list, 1, w_obj);

    for (Py_ssize_t i = 0; i < n_runs; i++) {
        PyObject *s_obj = PyLong_FromLongLong(runs[i].start);
        PyObject *l_obj = PyLong_FromLongLong(runs[i].end - runs[i].start);
        if (!s_obj || !l_obj) {
            Py_XDECREF(s_obj); Py_XDECREF(l_obj);
            Py_DECREF(dict); Py_DECREF(size_list); Py_DECREF(counts_list);
            return NULL;
        }
        PyList_SET_ITEM(counts_list, 2 * i, s_obj);
        PyList_SET_ITEM(counts_list, 2 * i + 1, l_obj);
    }
    int rc1 = PyDict_SetItem(dict, g_str_size, size_list);
    int rc2 = (rc1 == 0) ? PyDict_SetItem(dict, g_str_counts, counts_list) : -1;
    Py_DECREF(size_list);
    Py_DECREF(counts_list);
    if (rc1 < 0 || rc2 < 0) {
        Py_DECREF(dict);
        return NULL;
    }
    return dict;
}

static PyObject *py_rle_area(PyObject *self, PyObject *arg) {
    (void)self;
    RleRun stack_a[1024];
    RleRun *runs_a = NULL;
    Py_ssize_t n_a = 0;
    int64_t h = 0, w = 0;
    int rc = parse_validated_runs(arg, &h, &w, &runs_a, &n_a, stack_a, 1024);
    if (rc < 0) return NULL;
    if (rc > 0) Py_RETURN_NOTIMPLEMENTED;
    int64_t area = 0;
    for (Py_ssize_t i = 0; i < n_a; i++) area += runs_a[i].end - runs_a[i].start;
    if (runs_a != stack_a) free(runs_a);
    return PyLong_FromLongLong(area);
}

static PyObject *py_rle_intersection_area(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "rle_intersection_area expects 2 arguments");
        return NULL;
    }
    PyObject *a_obj = args[0], *b_obj = args[1];
    RleRun stack_a[1024], stack_b[1024];
    RleRun *runs_a = NULL, *runs_b = NULL;
    Py_ssize_t n_a = 0, n_b = 0;
    int64_t ha = 0, wa = 0, hb = 0, wb = 0;
    int rca = parse_validated_runs(a_obj, &ha, &wa, &runs_a, &n_a, stack_a, 1024);
    if (rca < 0) return NULL;
    if (rca > 0) Py_RETURN_NOTIMPLEMENTED;
    int rcb = parse_validated_runs(b_obj, &hb, &wb, &runs_b, &n_b, stack_b, 1024);
    if (rcb < 0) { if (runs_a != stack_a) free(runs_a); return NULL; }
    if (rcb > 0) { if (runs_a != stack_a) free(runs_a); Py_RETURN_NOTIMPLEMENTED; }
    if (ha != hb || wa != wb) {
        if (runs_a != stack_a) free(runs_a);
        if (runs_b != stack_b) free(runs_b);
        PyErr_SetString(PyExc_ValueError, "RLE masks must share the same [height, width] dimensions");
        return NULL;
    }
    int64_t inter = 0;
    Py_ssize_t i = 0, j = 0;
    while (i < n_a && j < n_b) {
        int64_t sa = runs_a[i].start, ea = runs_a[i].end;
        int64_t sb = runs_b[j].start, eb = runs_b[j].end;
        int64_t s = sa > sb ? sa : sb;
        int64_t e = ea < eb ? ea : eb;
        if (s < e) inter += e - s;
        if (ea <= eb) i++; else j++;
    }
    if (runs_a != stack_a) free(runs_a);
    if (runs_b != stack_b) free(runs_b);
    return PyLong_FromLongLong(inter);
}

static PyObject *py_rle_iou(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "rle_iou expects 2 arguments");
        return NULL;
    }
    PyObject *a_obj = args[0], *b_obj = args[1];
    RleRun stack_a[1024], stack_b[1024];
    RleRun *runs_a = NULL, *runs_b = NULL;
    Py_ssize_t n_a = 0, n_b = 0;
    int64_t ha = 0, wa = 0, hb = 0, wb = 0;
    int rca = parse_validated_runs(a_obj, &ha, &wa, &runs_a, &n_a, stack_a, 1024);
    if (rca < 0) return NULL;
    if (rca > 0) Py_RETURN_NOTIMPLEMENTED;
    int rcb = parse_validated_runs(b_obj, &hb, &wb, &runs_b, &n_b, stack_b, 1024);
    if (rcb < 0) { if (runs_a != stack_a) free(runs_a); return NULL; }
    if (rcb > 0) { if (runs_a != stack_a) free(runs_a); Py_RETURN_NOTIMPLEMENTED; }
    if (ha != hb || wa != wb) {
        if (runs_a != stack_a) free(runs_a);
        if (runs_b != stack_b) free(runs_b);
        PyErr_SetString(PyExc_ValueError, "RLE masks must share the same [height, width] dimensions");
        return NULL;
    }
    int64_t area_a = 0, area_b = 0;
    for (Py_ssize_t i = 0; i < n_a; i++) area_a += runs_a[i].end - runs_a[i].start;
    for (Py_ssize_t j = 0; j < n_b; j++) area_b += runs_b[j].end - runs_b[j].start;
    double res = 0.0;
    if (area_a == 0 && area_b == 0) {
        res = 1.0;
    } else if (area_a > 0 && area_b > 0) {
        int64_t inter = 0;
        Py_ssize_t i = 0, j = 0;
        while (i < n_a && j < n_b) {
            int64_t sa = runs_a[i].start, ea = runs_a[i].end;
            int64_t sb = runs_b[j].start, eb = runs_b[j].end;
            int64_t s = sa > sb ? sa : sb;
            int64_t e = ea < eb ? ea : eb;
            if (s < e) inter += e - s;
            if (ea <= eb) i++; else j++;
        }
        int64_t uni = area_a + area_b - inter;
        res = uni > 0 ? ((double)inter / (double)uni) : 0.0;
    }
    if (runs_a != stack_a) free(runs_a);
    if (runs_b != stack_b) free(runs_b);
    return PyFloat_FromDouble(res);
}

static PyObject *py_rle_set_op(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "rle_set_op expects 3 arguments");
        return NULL;
    }
    PyObject *a_obj = args[0], *b_obj = args[1];
    long op = PyLong_AsLong(args[2]);
    if (op == -1 && PyErr_Occurred()) return NULL;
    RleRun stack_a[1024], stack_b[1024], stack_out[2048];
    RleRun *runs_a = NULL, *runs_b = NULL, *out = stack_out;
    Py_ssize_t n_a = 0, n_b = 0;
    int64_t ha = 0, wa = 0, hb = 0, wb = 0;
    int rca = parse_validated_runs(a_obj, &ha, &wa, &runs_a, &n_a, stack_a, 1024);
    if (rca < 0) return NULL;
    if (rca > 0) Py_RETURN_NOTIMPLEMENTED;
    int rcb = parse_validated_runs(b_obj, &hb, &wb, &runs_b, &n_b, stack_b, 1024);
    if (rcb < 0) { if (runs_a != stack_a) free(runs_a); return NULL; }
    if (rcb > 0) { if (runs_a != stack_a) free(runs_a); Py_RETURN_NOTIMPLEMENTED; }
    if (ha != hb || wa != wb) {
        if (runs_a != stack_a) free(runs_a);
        if (runs_b != stack_b) free(runs_b);
        PyErr_SetString(PyExc_ValueError, "RLE masks must share the same [height, width] dimensions");
        return NULL;
    }

    Py_ssize_t max_out = n_a + n_b + 1;
    if (max_out > 2048) {
        out = (RleRun *)malloc((size_t)max_out * sizeof(RleRun));
        if (!out) {
            if (runs_a != stack_a) free(runs_a);
            if (runs_b != stack_b) free(runs_b);
            return PyErr_NoMemory();
        }
    }

    Py_ssize_t n_out = 0;
    if (op == 0) { /* intersection */
        Py_ssize_t i = 0, j = 0;
        while (i < n_a && j < n_b) {
            int64_t sa = runs_a[i].start, ea = runs_a[i].end;
            int64_t sb = runs_b[j].start, eb = runs_b[j].end;
            int64_t s = sa > sb ? sa : sb;
            int64_t e = ea < eb ? ea : eb;
            if (s < e) { out[n_out].start = s; out[n_out].end = e; n_out++; }
            if (ea <= eb) i++; else j++;
        }
    } else if (op == 1) { /* union */
        Py_ssize_t i = 0, j = 0;
        while (i < n_a || j < n_b) {
            int64_t cur_s, cur_e;
            if (j >= n_b || (i < n_a && runs_a[i].start <= runs_b[j].start)) {
                cur_s = runs_a[i].start; cur_e = runs_a[i].end; i++;
            } else {
                cur_s = runs_b[j].start; cur_e = runs_b[j].end; j++;
            }
            if (n_out > 0 && cur_s <= out[n_out - 1].end) {
                if (cur_e > out[n_out - 1].end) out[n_out - 1].end = cur_e;
            } else {
                out[n_out].start = cur_s; out[n_out].end = cur_e; n_out++;
            }
        }
    } else { /* difference A \ B */
        Py_ssize_t j = 0;
        for (Py_ssize_t i = 0; i < n_a; i++) {
            int64_t cur = runs_a[i].start, ea = runs_a[i].end;
            while (j < n_b && runs_b[j].end <= cur) j++;
            Py_ssize_t k = j;
            while (k < n_b && runs_b[k].start < ea) {
                int64_t sb = runs_b[k].start, eb = runs_b[k].end;
                if (sb > cur) {
                    out[n_out].start = cur;
                    out[n_out].end = sb < ea ? sb : ea;
                    n_out++;
                }
                if (eb > cur) cur = eb;
                if (cur >= ea) break;
                k++;
            }
            if (cur < ea) {
                out[n_out].start = cur; out[n_out].end = ea; n_out++;
            }
        }
    }

    PyObject *res = runs_to_rle_dict(ha, wa, out, n_out);
    if (out != stack_out) free(out);
    if (runs_a != stack_a) free(runs_a);
    if (runs_b != stack_b) free(runs_b);
    return res;
}

static PyObject *py_rle_mask_box_and_centroid(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "rle_mask_box_and_centroid expects (mask_rle, width, height)");
        return NULL;
    }
    PyObject *mask_rle = args[0];
    int ov = 0;
    long long width = PyLong_AsLongLongAndOverflow(args[1], &ov);
    if (ov || (width == -1 && PyErr_Occurred())) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
    long long height = PyLong_AsLongLongAndOverflow(args[2], &ov);
    if (ov || (height == -1 && PyErr_Occurred())) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }

    if (!PyDict_Check(mask_rle)) Py_RETURN_NONE;
    PyObject *size = PyDict_GetItem(mask_rle, g_str_size);
    PyObject *counts = PyDict_GetItem(mask_rle, g_str_counts);
    if (!size || !PyList_Check(size) || PyList_GET_SIZE(size) != 2 ||
        !counts || !PyList_Check(counts) || (PyList_GET_SIZE(counts) & 1) != 0) {
        Py_RETURN_NONE;
    }
    PyObject *s0_obj = PyList_GET_ITEM(size, 0);
    PyObject *s1_obj = PyList_GET_ITEM(size, 1);
    if (!PyLong_Check(s0_obj) || !PyLong_Check(s1_obj)) Py_RETURN_NOTIMPLEMENTED;
    long long sh = PyLong_AsLongLongAndOverflow(s0_obj, &ov);
    if (ov) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
    long long sw = PyLong_AsLongLongAndOverflow(s1_obj, &ov);
    if (ov) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
    if (sh != height || sw != width || width <= 0 || height <= 0) Py_RETURN_NONE;

    __int128 limit128 = (__int128)width * (__int128)height;
    if (limit128 > INT64_MAX) Py_RETURN_NOTIMPLEMENTED;

    Py_ssize_t c_len = PyList_GET_SIZE(counts);

#ifdef __SIZEOF_INT128__
    if (width <= 65535 && limit128 <= 0x3fffffff) {
        int64_t lim = (int64_t)limit128;
        int64_t w64 = (int64_t)width;
        uint64_t inv_w = (width > 1) ? (((1ULL << 48) / (uint64_t)width) + 1ULL) : 0ULL;
        int64_t total_pixels = 0;
        int64_t sum_y = 0;
        int64_t sum_index = 0;
        int64_t min_x = w64, min_y = (int64_t)height, max_x = -1, max_y = -1;

        for (Py_ssize_t i = 0; i < c_len; i += 2) {
            PyObject *st_obj = PyList_GET_ITEM(counts, i);
            PyObject *ln_obj = PyList_GET_ITEM(counts, i + 1);
            if (!PyLong_Check(st_obj) || !PyLong_Check(ln_obj)) Py_RETURN_NONE;
            long start_l = PyLong_AsLong(st_obj);
            if (start_l == -1 && PyErr_Occurred()) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
            long len_l = PyLong_AsLong(ln_obj);
            if (len_l == -1 && PyErr_Occurred()) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
            if (len_l <= 0 || start_l < 0 || start_l > (long)lim || len_l > (long)lim - start_l) Py_RETURN_NONE;
            int64_t start = (int64_t)start_l;
            int64_t length = (int64_t)len_l;
            int64_t end = start + length - 1;
            int64_t y0 = (width == 1) ? start : (int64_t)(((unsigned __int128)(uint32_t)start * inv_w) >> 48);
            int64_t x0 = start - y0 * w64;
            int64_t y1 = (width == 1) ? end : (int64_t)(((unsigned __int128)(uint32_t)end * inv_w) >> 48);
            int64_t x1 = end - y1 * w64;
            if (y0 < min_y) min_y = y0;
            if (y1 > max_y) max_y = y1;
            if (y0 == y1) {
                if (x0 < min_x) min_x = x0;
                if (x1 > max_x) max_x = x1;
                sum_y += length * y0;
            } else {
                min_x = 0;
                max_x = w64 - 1;
                int64_t n0 = w64 - x0;
                int64_t n1 = x1 + 1;
                int64_t m = y1 - y0 - 1;
                sum_y += n0 * y0 + ((w64 * m * (y0 + y1)) >> 1) + n1 * y1;
            }
            sum_index += length * start + ((length * (length - 1)) >> 1);
            total_pixels += length;
        }
        if (total_pixels == 0) Py_RETURN_NONE;
        int64_t sum_x = sum_index - w64 * sum_y;
        double mean_x = (double)sum_x / (double)total_pixels;
        double mean_y = (double)sum_y / (double)total_pixels;
        return Py_BuildValue("((LLLL)(dd))",
                             (long long)min_x, (long long)min_y,
                             (long long)max_x, (long long)max_y,
                             mean_x, mean_y);
    }
#endif

    int64_t total_pixels = 0;
    __int128 sum_y = 0;
    __int128 sum_index = 0;
    int64_t min_x = width, min_y = height, max_x = -1, max_y = -1;

    for (Py_ssize_t i = 0; i < c_len; i += 2) {
        PyObject *st_obj = PyList_GET_ITEM(counts, i);
        PyObject *ln_obj = PyList_GET_ITEM(counts, i + 1);
        if (!PyLong_Check(st_obj) || !PyLong_Check(ln_obj)) Py_RETURN_NONE;
        long long start = PyLong_AsLongLongAndOverflow(st_obj, &ov);
        if (ov) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
        long long length = PyLong_AsLongLongAndOverflow(ln_obj, &ov);
        if (ov) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
        if (length <= 0 || start < 0 || (__int128)start + (__int128)length - 1 >= limit128) {
            Py_RETURN_NONE;
        }
        int64_t end = (int64_t)start + (int64_t)length - 1;
        int64_t y0 = (int64_t)start / width;
        int64_t x0 = (int64_t)start - y0 * width;
        int64_t y1 = end / width;
        int64_t x1 = end - y1 * width;
        if (y0 < min_y) min_y = y0;
        if (y1 > max_y) max_y = y1;
        if (y0 == y1) {
            if (x0 < min_x) min_x = x0;
            if (x1 > max_x) max_x = x1;
            sum_y += (__int128)length * (__int128)y0;
        } else {
            min_x = 0;
            max_x = width - 1;
            int64_t n0 = width - x0;
            int64_t n1 = x1 + 1;
            int64_t m = y1 - y0 - 1;
            sum_y += (__int128)n0 * (__int128)y0
                   + ((__int128)width * (__int128)m * (__int128)(y0 + y1)) / 2
                   + (__int128)n1 * (__int128)y1;
        }
        sum_index += (__int128)length * (__int128)start
                   + ((__int128)length * (__int128)(length - 1)) / 2;
        total_pixels += (int64_t)length;
    }

    if (total_pixels == 0) Py_RETURN_NONE;
    __int128 sum_x = sum_index - (__int128)width * sum_y;
    const __int128 max_exact_d = (__int128)1 << 53;
    if (sum_x > max_exact_d || sum_x < -max_exact_d || sum_y > max_exact_d || total_pixels > (1LL << 53)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    double mean_x = (double)sum_x / (double)total_pixels;
    double mean_y = (double)sum_y / (double)total_pixels;
    return Py_BuildValue("((LLLL)(dd))",
                         (long long)min_x, (long long)min_y,
                         (long long)max_x, (long long)max_y,
                         mean_x, mean_y);
}

static PyObject *py_encode_mask_rle_bytes(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *mask_obj = NULL;
    long long height = 0, width = 0;
    if (!PyArg_ParseTuple(args, "OLL", &mask_obj, &height, &width)) return NULL;

    Py_buffer m_buf;
    if (PyObject_GetBuffer(mask_obj, &m_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) return NULL;
    const uint8_t *p = (const uint8_t *)m_buf.buf;
    Py_ssize_t total = m_buf.len;

    Py_ssize_t cap = 64;
    int64_t *pairs = (int64_t *)malloc((size_t)cap * 2 * sizeof(int64_t));
    if (!pairs) { PyBuffer_Release(&m_buf); return PyErr_NoMemory(); }
    Py_ssize_t n_runs = 0;
    int oom = 0;

    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t i = 0;
    while (i < total) {
        /* Skip false pixels using 8-byte word scans */
        while (i + 8 <= total) {
            uint64_t w64;
            memcpy(&w64, p + i, 8);
            if (w64 != 0ULL) break;
            i += 8;
        }
        while (i < total && p[i] == 0) i++;
        if (i >= total) break;
        Py_ssize_t start = i;
        while (i + 8 <= total) {
            uint64_t w64;
            memcpy(&w64, p + i, 8);
            if (w64 != 0x0101010101010101ULL) break;
            i += 8;
        }
        while (i < total && p[i] != 0) i++;
        if (n_runs >= cap) {
            Py_ssize_t new_cap = cap * 2;
            int64_t *np_pairs = (int64_t *)realloc(pairs, (size_t)new_cap * 2 * sizeof(int64_t));
            if (!np_pairs) { oom = 1; break; }
            pairs = np_pairs;
            cap = new_cap;
        }
        pairs[2 * n_runs + 0] = (int64_t)start;
        pairs[2 * n_runs + 1] = (int64_t)(i - start);
        n_runs++;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&m_buf);

    if (oom) { free(pairs); return PyErr_NoMemory(); }

    PyObject *dict = PyDict_New();
    PyObject *size_list = PyList_New(2);
    PyObject *counts_list = PyList_New(n_runs * 2);
    if (!dict || !size_list || !counts_list) {
        free(pairs);
        Py_XDECREF(dict); Py_XDECREF(size_list); Py_XDECREF(counts_list);
        return NULL;
    }
    PyObject *h_obj = PyLong_FromLongLong(height);
    PyObject *w_obj = PyLong_FromLongLong(width);
    if (!h_obj || !w_obj) {
        free(pairs); Py_XDECREF(h_obj); Py_XDECREF(w_obj);
        Py_DECREF(dict); Py_DECREF(size_list); Py_DECREF(counts_list);
        return NULL;
    }
    PyList_SET_ITEM(size_list, 0, h_obj);
    PyList_SET_ITEM(size_list, 1, w_obj);
    for (Py_ssize_t j = 0; j < n_runs * 2; j++) {
        PyObject *v = PyLong_FromLongLong(pairs[j]);
        if (!v) {
            free(pairs); Py_DECREF(dict); Py_DECREF(size_list); Py_DECREF(counts_list);
            return NULL;
        }
        PyList_SET_ITEM(counts_list, j, v);
    }
    free(pairs);
    int rc1 = PyDict_SetItem(dict, g_str_size, size_list);
    int rc2 = (rc1 == 0) ? PyDict_SetItem(dict, g_str_counts, counts_list) : -1;
    Py_DECREF(size_list);
    Py_DECREF(counts_list);
    if (rc1 < 0 || rc2 < 0) { Py_DECREF(dict); return NULL; }
    return dict;
}

static PyObject *py_decode_mask_rle_into(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "decode_mask_rle_into expects (payload, width, height)");
        return NULL;
    }
    PyObject *payload = args[0];
    int ov = 0;
    long long width = PyLong_AsLongLongAndOverflow(args[1], &ov);
    if (ov || (width == -1 && PyErr_Occurred())) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
    long long height = PyLong_AsLongLongAndOverflow(args[2], &ov);
    if (ov || (height == -1 && PyErr_Occurred())) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }

    if (!PyDict_Check(payload)) Py_RETURN_NONE;
    PyObject *size = PyDict_GetItem(payload, g_str_size);
    PyObject *counts = PyDict_GetItem(payload, g_str_counts);
    if (!size || !PyList_Check(size) || PyList_GET_SIZE(size) != 2 ||
        !counts || !PyList_Check(counts) || (PyList_GET_SIZE(counts) & 1) != 0) {
        Py_RETURN_NONE;
    }
    PyObject *s0 = PyList_GET_ITEM(size, 0);
    PyObject *s1 = PyList_GET_ITEM(size, 1);
    if (!PyLong_Check(s0) || !PyLong_Check(s1)) Py_RETURN_NOTIMPLEMENTED;
    long long sh = PyLong_AsLongLongAndOverflow(s0, &ov);
    if (ov) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
    long long sw = PyLong_AsLongLongAndOverflow(s1, &ov);
    if (ov) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
    if (sh != height || sw != width || width < 0 || height < 0) Py_RETURN_NONE;

    __int128 limit128 = (__int128)width * (__int128)height;
    if (limit128 > INT64_MAX) Py_RETURN_NOTIMPLEMENTED;
    int64_t total_limit = (int64_t)limit128;

    Py_ssize_t c_len = PyList_GET_SIZE(counts);
    RleRun stack_buf[1024];
    RleRun *runs = (c_len >> 1) <= 1024 ? stack_buf
                   : (RleRun *)malloc((size_t)((c_len >> 1) > 0 ? (c_len >> 1) : 1) * sizeof(RleRun));
    if (!runs) return PyErr_NoMemory();

    Py_ssize_t n_runs = 0;
    for (Py_ssize_t i = 0; i < c_len; i += 2) {
        PyObject *st_obj = PyList_GET_ITEM(counts, i);
        PyObject *ln_obj = PyList_GET_ITEM(counts, i + 1);
        if (!PyLong_Check(st_obj) || !PyLong_Check(ln_obj)) {
            if (runs != stack_buf) free(runs);
            Py_RETURN_NONE;
        }
        long long start = PyLong_AsLongLongAndOverflow(st_obj, &ov);
        if (ov) { PyErr_Clear(); if (runs != stack_buf) free(runs); Py_RETURN_NOTIMPLEMENTED; }
        long long length = PyLong_AsLongLongAndOverflow(ln_obj, &ov);
        if (ov) { PyErr_Clear(); if (runs != stack_buf) free(runs); Py_RETURN_NOTIMPLEMENTED; }
        if (length <= 0 || start < 0 || (__int128)start + (__int128)length > limit128) {
            if (runs != stack_buf) free(runs);
            Py_RETURN_NONE;
        }
        runs[n_runs].start = (int64_t)start;
        runs[n_runs].end = (int64_t)length; /* length */
        n_runs++;
    }

    uint8_t *dst = NULL;
    PyObject *arr = alloc_fast_2d((Py_ssize_t)height, (Py_ssize_t)width, 1, g_np_bool, (void **)&dst);
    if (!arr) {
        if (runs != stack_buf) free(runs);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    if (total_limit > 0) memset(dst, 0, (size_t)total_limit);
    for (Py_ssize_t i = 0; i < n_runs; i++) {
        memset(dst + runs[i].start, 1, (size_t)runs[i].end);
    }
    Py_END_ALLOW_THREADS

    if (runs != stack_buf) free(runs);
    return arr;
}

/* ============================================================================
 * 6 & 7. circle_consensus_2d & fit_circular_aperture_fast (AVX-512/AVX2 + C11)
 * ============================================================================ */

static double pairwise_sum_f64(const double *a, Py_ssize_t n) {
    if (n == 0) return 0.0;
    if (n < 8) {
        double s = 0.0;
        for (Py_ssize_t i = 0; i < n; i++) s += a[i];
        return s;
    }
    if (n <= 128) {
        double r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
        double r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        Py_ssize_t i = 8;
        for (; i <= n - 8; i += 8) {
            r0 += a[i + 0]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3];
            r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7];
        }
        double s = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; i++) s += a[i];
        return s;
    }
    Py_ssize_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum_f64(a, n2) + pairwise_sum_f64(a + n2, n - n2);
}

static inline int popcnt_mask8(unsigned m) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(m);
#else
    int c = 0;
    while (m) { c += (int)(m & 1u); m >>= 1; }
    return c;
#endif
}

#if ES_HAVE_X86_RUNTIME_SIMD
__attribute__((target("avx512f,avx512dq,avx512vl")))
static void circle_consensus_2d_core_avx512(
    const double *px, const double *py, Py_ssize_t n,
    const int64_t *tri, Py_ssize_t h,
    double fc_x, double fc_y, double fitted_radius,
    double inlier_thresh, double max_c_disc, double max_r_disc,
    double *row,
    int *out_agreed, double *out_ux, double *out_uy, double *out_r)
{
    Py_ssize_t best_inliers = -1;
    double best_rmse = HUGE_VAL;
    double best_ux = fc_x, best_uy = fc_y, best_r = fitted_radius;
    int found_any = 0;
    __m512d v_abs = _mm512_castsi512_pd(_mm512_set1_epi64(0x7fffffffffffffffLL));
    __m512d v_thresh = _mm512_set1_pd(inlier_thresh);

    for (Py_ssize_t j = 0; j < h; j++) {
        int64_t i0 = tri[3 * j + 0];
        int64_t i1 = tri[3 * j + 1];
        int64_t i2 = tri[3 * j + 2];
        double ax = px[i0], ay = py[i0];
        double bx = px[i1], by = py[i1];
        double cx = px[i2], cy = py[i2];
        double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if (fabs(d) < 1e-10) continue;
        double a2 = ax * ax + ay * ay;
        double b2 = bx * bx + by * by;
        double c2 = cx * cx + cy * cy;
        double ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
        double uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
        double r = sqrt((ax - ux) * (ax - ux) + (ay - uy) * (ay - uy));
        if (!(r >= 0.01 && r <= 0.35)) continue;

        Py_ssize_t cnt = 0;
        Py_ssize_t misses = 0;
        Py_ssize_t max_misses = (best_inliers > 0) ? (n - best_inliers) : n;
        Py_ssize_t k = 0;
        __m512d v_ux = _mm512_set1_pd(ux);
        __m512d v_uy = _mm512_set1_pd(uy);
        __m512d v_r = _mm512_set1_pd(r);
        for (; k + 32 <= n; k += 32) {
            for (int blk = 0; blk < 4; blk++) {
                Py_ssize_t off = k + (Py_ssize_t)blk * 8;
                __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(px + off), v_ux);
                __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(py + off), v_uy);
                __m512d d2 = _mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy));
                __m512d dist = _mm512_sqrt_pd(d2);
                __m512d diff = _mm512_sub_pd(dist, v_r);
                __m512d resid = _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(diff), _mm512_castpd_si512(v_abs)));
                __mmask8 m = _mm512_cmp_pd_mask(resid, v_thresh, _CMP_LE_OQ);
                if (m == 0xFF) {
                    _mm512_storeu_pd(row + cnt, resid);
                    cnt += 8;
                } else if (m != 0) {
                    _mm512_mask_compressstoreu_pd(row + cnt, m, resid);
                    int pc = __builtin_popcount((unsigned)m);
                    cnt += pc;
                    misses += (8 - pc);
                } else {
                    misses += 8;
                }
            }
            if (misses > max_misses) break;
        }
        if (misses <= max_misses) {
            for (; k + 8 <= n; k += 8) {
                __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(px + k), v_ux);
                __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(py + k), v_uy);
                __m512d d2 = _mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy));
                __m512d dist = _mm512_sqrt_pd(d2);
                __m512d diff = _mm512_sub_pd(dist, v_r);
                __m512d resid = _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(diff), _mm512_castpd_si512(v_abs)));
                __mmask8 m = _mm512_cmp_pd_mask(resid, v_thresh, _CMP_LE_OQ);
                if (m == 0xFF) {
                    _mm512_storeu_pd(row + cnt, resid);
                    cnt += 8;
                } else if (m != 0) {
                    _mm512_mask_compressstoreu_pd(row + cnt, m, resid);
                    int pc = __builtin_popcount((unsigned)m);
                    cnt += pc;
                    misses += (8 - pc);
                } else {
                    misses += 8;
                }
            }
        }
        if (misses <= max_misses) {
            for (; k < n; k++) {
                double dx = px[k] - ux;
                double dy = py[k] - uy;
                double resid = fabs(sqrt(dx * dx + dy * dy) - r);
                if (resid <= inlier_thresh) {
                    row[cnt++] = resid;
                } else {
                    misses++;
                    if (misses > max_misses) break;
                }
            }
        }
        if (cnt < best_inliers) continue;
        double rmse = cnt > 0 ? (pairwise_sum_f64(row, cnt) / (double)cnt) : HUGE_VAL;
        if (cnt > best_inliers || (cnt == best_inliers && rmse < best_rmse)) {
            best_inliers = cnt;
            best_rmse = rmse;
            best_ux = ux;
            best_uy = uy;
            best_r = r;
            found_any = 1;
        }
    }

    double out_ux_v = found_any ? best_ux : fc_x;
    double out_uy_v = found_any ? best_uy : fc_y;
    double out_r_v = found_any ? best_r : fitted_radius;
    double dcx = out_ux_v - fc_x, dcy = out_uy_v - fc_y;
    double center_diff = sqrt(dcx * dcx + dcy * dcy);
    double radius_diff = fabs(out_r_v - fitted_radius);
    *out_agreed = (center_diff <= max_c_disc && radius_diff <= max_r_disc) ? 1 : 0;
    *out_ux = out_ux_v;
    *out_uy = out_uy_v;
    *out_r = out_r_v;
}

__attribute__((target("avx2")))
static void circle_consensus_2d_core_avx2(
    const double *px, const double *py, Py_ssize_t n,
    const int64_t *tri, Py_ssize_t h,
    double fc_x, double fc_y, double fitted_radius,
    double inlier_thresh, double max_c_disc, double max_r_disc,
    double *row,
    int *out_agreed, double *out_ux, double *out_uy, double *out_r)
{
    Py_ssize_t best_inliers = -1;
    double best_rmse = HUGE_VAL;
    double best_ux = fc_x, best_uy = fc_y, best_r = fitted_radius;
    int found_any = 0;
    __m256d v_abs = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
    __m256d v_thresh = _mm256_set1_pd(inlier_thresh);

    for (Py_ssize_t j = 0; j < h; j++) {
        int64_t i0 = tri[3 * j + 0];
        int64_t i1 = tri[3 * j + 1];
        int64_t i2 = tri[3 * j + 2];
        double ax = px[i0], ay = py[i0];
        double bx = px[i1], by = py[i1];
        double cx = px[i2], cy = py[i2];
        double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if (fabs(d) < 1e-10) continue;
        double a2 = ax * ax + ay * ay;
        double b2 = bx * bx + by * by;
        double c2 = cx * cx + cy * cy;
        double ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
        double uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
        double r = sqrt((ax - ux) * (ax - ux) + (ay - uy) * (ay - uy));
        if (!(r >= 0.01 && r <= 0.35)) continue;

        Py_ssize_t cnt = 0;
        Py_ssize_t misses = 0;
        Py_ssize_t max_misses = (best_inliers > 0) ? (n - best_inliers) : n;
        Py_ssize_t k = 0;
        __m256d v_ux = _mm256_set1_pd(ux);
        __m256d v_uy = _mm256_set1_pd(uy);
        __m256d v_r = _mm256_set1_pd(r);
        double tmp_r[4];
        for (; k + 16 <= n; k += 16) {
            for (int blk = 0; blk < 4; blk++) {
                Py_ssize_t off = k + (Py_ssize_t)blk * 4;
                __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(px + off), v_ux);
                __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(py + off), v_uy);
                __m256d d2 = _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy));
                __m256d dist = _mm256_sqrt_pd(d2);
                __m256d resid = _mm256_and_pd(_mm256_sub_pd(dist, v_r), v_abs);
                int m = _mm256_movemask_pd(_mm256_cmp_pd(resid, v_thresh, _CMP_LE_OQ));
                if (m == 0) {
                    misses += 4;
                } else if (m == 0xF) {
                    _mm256_storeu_pd(row + cnt, resid);
                    cnt += 4;
                } else {
                    _mm256_storeu_pd(tmp_r, resid);
                    for (int l = 0; l < 4; l++) {
                        if (m & (1 << l)) row[cnt++] = tmp_r[l];
                        else misses++;
                    }
                }
            }
            if (misses > max_misses) break;
        }
        if (misses <= max_misses) {
            for (; k < n; k++) {
                double dx = px[k] - ux;
                double dy = py[k] - uy;
                double resid = fabs(sqrt(dx * dx + dy * dy) - r);
                if (resid <= inlier_thresh) {
                    row[cnt++] = resid;
                } else {
                    misses++;
                    if (misses > max_misses) break;
                }
            }
        }
        if (cnt < best_inliers) continue;
        double rmse = cnt > 0 ? (pairwise_sum_f64(row, cnt) / (double)cnt) : HUGE_VAL;
        if (cnt > best_inliers || (cnt == best_inliers && rmse < best_rmse)) {
            best_inliers = cnt;
            best_rmse = rmse;
            best_ux = ux;
            best_uy = uy;
            best_r = r;
            found_any = 1;
        }
    }

    double out_ux_v = found_any ? best_ux : fc_x;
    double out_uy_v = found_any ? best_uy : fc_y;
    double out_r_v = found_any ? best_r : fitted_radius;
    double dcx = out_ux_v - fc_x, dcy = out_uy_v - fc_y;
    double center_diff = sqrt(dcx * dcx + dcy * dcy);
    double radius_diff = fabs(out_r_v - fitted_radius);
    *out_agreed = (center_diff <= max_c_disc && radius_diff <= max_r_disc) ? 1 : 0;
    *out_ux = out_ux_v;
    *out_uy = out_uy_v;
    *out_r = out_r_v;
}
#endif

static void circle_consensus_2d_core(
    const double *px, const double *py, Py_ssize_t n,
    const int64_t *tri, Py_ssize_t h,
    double fc_x, double fc_y, double fitted_radius,
    double inlier_thresh, double max_c_disc, double max_r_disc,
    double *row,
    int *out_agreed, double *out_ux, double *out_uy, double *out_r)
{
#if ES_HAVE_X86_RUNTIME_SIMD
    if (g_simd_level >= 2) {
        circle_consensus_2d_core_avx512(
            px, py, n, tri, h, fc_x, fc_y, fitted_radius,
            inlier_thresh, max_c_disc, max_r_disc, row,
            out_agreed, out_ux, out_uy, out_r);
        return;
    }
    if (g_simd_level == 1) {
        circle_consensus_2d_core_avx2(
            px, py, n, tri, h, fc_x, fc_y, fitted_radius,
            inlier_thresh, max_c_disc, max_r_disc, row,
            out_agreed, out_ux, out_uy, out_r);
        return;
    }
#endif
    Py_ssize_t best_inliers = -1;
    double best_rmse = HUGE_VAL;
    double best_ux = fc_x, best_uy = fc_y, best_r = fitted_radius;
    int found_any = 0;

    for (Py_ssize_t j = 0; j < h; j++) {
        int64_t i0 = tri[3 * j + 0];
        int64_t i1 = tri[3 * j + 1];
        int64_t i2 = tri[3 * j + 2];
        double ax = px[i0], ay = py[i0];
        double bx = px[i1], by = py[i1];
        double cx = px[i2], cy = py[i2];
        double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if (fabs(d) < 1e-10) continue;
        double a2 = ax * ax + ay * ay;
        double b2 = bx * bx + by * by;
        double c2 = cx * cx + cy * cy;
        double ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
        double uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
        double r = sqrt((ax - ux) * (ax - ux) + (ay - uy) * (ay - uy));
        if (!(r >= 0.01 && r <= 0.35)) continue;

        Py_ssize_t cnt = 0;
        Py_ssize_t misses = 0;
        Py_ssize_t max_misses = (best_inliers > 0) ? (n - best_inliers) : n;
        for (Py_ssize_t k = 0; k < n; k++) {
            double dx = px[k] - ux;
            double dy = py[k] - uy;
            double resid = fabs(sqrt(dx * dx + dy * dy) - r);
            if (resid <= inlier_thresh) {
                row[cnt++] = resid;
            } else {
                misses++;
                if (misses > max_misses) break;
            }
        }
        if (cnt < best_inliers) continue;
        double rmse = cnt > 0 ? (pairwise_sum_f64(row, cnt) / (double)cnt) : HUGE_VAL;
        if (cnt > best_inliers || (cnt == best_inliers && rmse < best_rmse)) {
            best_inliers = cnt;
            best_rmse = rmse;
            best_ux = ux;
            best_uy = uy;
            best_r = r;
            found_any = 1;
        }
    }

    double out_ux_v = found_any ? best_ux : fc_x;
    double out_uy_v = found_any ? best_uy : fc_y;
    double out_r_v = found_any ? best_r : fitted_radius;
    double dcx = out_ux_v - fc_x, dcy = out_uy_v - fc_y;
    double center_diff = sqrt(dcx * dcx + dcy * dcy);
    double radius_diff = fabs(out_r_v - fitted_radius);
    *out_agreed = (center_diff <= max_c_disc && radius_diff <= max_r_disc) ? 1 : 0;
    *out_ux = out_ux_v;
    *out_uy = out_uy_v;
    *out_r = out_r_v;
}

static PyObject *py_circle_consensus_2d(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 7) {
        PyErr_SetString(PyExc_TypeError, "circle_consensus_2d expects 7 arguments");
        return NULL;
    }
    PyObject *pts_obj = args[0], *tri_obj = args[1], *fc_obj = args[2];
    double fitted_radius = PyFloat_AsDouble(args[3]);
    if (fitted_radius == -1.0 && PyErr_Occurred()) return NULL;
    double inlier_thresh = PyFloat_AsDouble(args[4]);
    if (inlier_thresh == -1.0 && PyErr_Occurred()) return NULL;
    double max_c_disc = PyFloat_AsDouble(args[5]);
    if (max_c_disc == -1.0 && PyErr_Occurred()) return NULL;
    double max_r_disc = PyFloat_AsDouble(args[6]);
    if (max_r_disc == -1.0 && PyErr_Occurred()) return NULL;

    Py_buffer pts_b = {0}, tri_b = {0}, fc_b = {0};
    int got_p = 0, got_t = 0, got_fc = 0;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(pts_obj, &pts_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_p = 1;
    if (PyObject_GetBuffer(tri_obj, &tri_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_t = 1;
    if (PyObject_GetBuffer(fc_obj, &fc_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_fc = 1;

    Py_ssize_t n = pts_b.len / (2 * (Py_ssize_t)sizeof(double));
    Py_ssize_t h = tri_b.len / (3 * (Py_ssize_t)sizeof(int64_t));
    const double *pts = (const double *)pts_b.buf;
    const int64_t *tri = (const int64_t *)tri_b.buf;
    const double *fc = (const double *)fc_b.buf;

    ThreadScratch *sc = get_scratch();
    if (!sc || ensure_dbl_tmp(sc, (size_t)n * 3 + 64) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }
    double *px = sc->dbl_tmp;
    double *py = px + n;
    double *row = py + n;

    int agreed = 0;
    double out_ux = fc[0], out_uy = fc[1], out_r = fitted_radius;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t k = 0; k < n; k++) {
        px[k] = pts[2 * k + 0];
        py[k] = pts[2 * k + 1];
    }
    circle_consensus_2d_core(
        px, py, n, tri, h, fc[0], fc[1], fitted_radius,
        inlier_thresh, max_c_disc, max_r_disc, row,
        &agreed, &out_ux, &out_uy, &out_r
    );
    Py_END_ALLOW_THREADS

    double *c_out = NULL;
    PyObject *c_arr = alloc_fast_1d(2, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&c_out);
    if (!c_arr) goto cleanup;
    c_out[0] = out_ux;
    c_out[1] = out_uy;

    ret_obj = Py_BuildValue("(OOd)", agreed ? Py_True : Py_False, c_arr, out_r);
    Py_DECREF(c_arr);

cleanup:
    if (got_p) PyBuffer_Release(&pts_b);
    if (got_t) PyBuffer_Release(&tri_b);
    if (got_fc) PyBuffer_Release(&fc_b);
    return ret_obj;
}

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return da < db ? -1 : (da > db ? 1 : 0);
}

static inline void sort_doubles_asc(double *a, Py_ssize_t n) {
    if (n <= 64) {
        for (Py_ssize_t i = 1; i < n; i++) {
            double key = a[i];
            Py_ssize_t j = i - 1;
            while (j >= 0 && a[j] > key) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = key;
        }
    } else {
        qsort(a, (size_t)n, sizeof(double), cmp_double_asc);
    }
}

static void jacobi_eigh_3x3(const double a_in[3][3], double evals[3], double evecs[3][3]) {
    double a[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            a[i][j] = a_in[i][j];
            evecs[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    static const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int sweep = 0; sweep < 15; sweep++) {
        double off = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
        double diag = fabs(a[0][0]) + fabs(a[1][1]) + fabs(a[2][2]) + 1.0;
        if (off <= 1e-32 * diag) break;
        for (int idx = 0; idx < 3; idx++) {
            int p = pairs[idx][0], q = pairs[idx][1];
            double apq = a[p][q];
            if (fabs(apq) <= 1e-32) continue;
            double tau = (a[q][q] - a[p][p]) / (2.0 * apq);
            double t = (tau >= 0.0)
                ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
            double c = 1.0 / sqrt(1.0 + t * t);
            double s = t * c;
            a[p][p] -= t * apq;
            a[q][q] += t * apq;
            a[p][q] = a[q][p] = 0.0;
            for (int r = 0; r < 3; r++) {
                if (r == p || r == q) continue;
                double arp = a[r][p], arq = a[r][q];
                a[r][p] = a[p][r] = c * arp - s * arq;
                a[r][q] = a[q][r] = s * arp + c * arq;
            }
            for (int r = 0; r < 3; r++) {
                double vp = evecs[r][p], vq = evecs[r][q];
                evecs[r][p] = c * vp - s * vq;
                evecs[r][q] = s * vp + c * vq;
            }
        }
    }
    evals[0] = a[0][0];
    evals[1] = a[1][1];
    evals[2] = a[2][2];
}

static PyObject *py_fit_circular_aperture_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 7) {
        PyErr_SetString(PyExc_TypeError, "fit_circular_aperture_fast expects 7 arguments");
        return NULL;
    }
    PyObject *pts_obj = args[0], *up_obj = args[1];
    long num_sectors_l = PyLong_AsLong(args[2]);
    if (num_sectors_l == -1 && PyErr_Occurred()) return NULL;
    double min_cov = PyFloat_AsDouble(args[3]);
    if (min_cov == -1.0 && PyErr_Occurred()) return NULL;
    double min_rad = PyFloat_AsDouble(args[4]);
    if (min_rad == -1.0 && PyErr_Occurred()) return NULL;
    double max_rad = PyFloat_AsDouble(args[5]);
    if (max_rad == -1.0 && PyErr_Occurred()) return NULL;
    PyObject *triplets_fn = args[6];

    Py_buffer pts_b = {0}, up_b = {0};
    int got_p = 0, got_u = 0;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(pts_obj, &pts_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_p = 1;
    if (PyObject_GetBuffer(up_obj, &up_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_u = 1;

    Py_ssize_t n = pts_b.len / (3 * (Py_ssize_t)sizeof(double));
    int num_sectors = (int)num_sectors_l;
    if (n < 6 || num_sectors < 1 || up_b.len != 3 * (Py_ssize_t)sizeof(double)) {
        ret_obj = Py_NotImplemented; Py_INCREF(ret_obj); goto cleanup;
    }
    const double *pts = (const double *)pts_b.buf;
    const double *up_raw = (const double *)up_b.buf;

    size_t n_pos = (size_t)n;
    size_t ns_pos = (size_t)num_sectors;
    ThreadScratch *sc = get_scratch();
    if (!sc ||
        ensure_dbl_tmp(sc, 14 * n_pos + ns_pos + 1) < 0 ||
        ensure_dbl_tmp2(sc, n_pos + 64) < 0 ||
        ensure_marks(sc, n_pos) < 0 ||
        ensure_p_idx(sc, n_pos) < 0 ||
        ensure_idx_tmp(sc, 2 * ns_pos + 1 + n_pos) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }

    double *px = sc->dbl_tmp;
    double *py = px + n_pos;
    double *pz = py + n_pos;
    double *in_px = pz + n_pos;
    double *in_py = in_px + n_pos;
    double *in_pz = in_py + n_pos;
    double *weights = in_pz + n_pos;
    double *residuals = weights + n_pos;
    double *xs = residuals + n_pos;
    double *ys = xs + n_pos;
    double *rads = ys + n_pos;
    double *rim_px = rads + n_pos;
    double *rim_py = rim_px + n_pos;
    double *sec_r = rim_py + n_pos;
    double *edges = sec_r + n_pos;
    double *dbl_row = sc->dbl_tmp2;
    uint8_t *inlier_mask = sc->marks;
    int32_t *sec_ids = sc->p_idx;
    Py_ssize_t *sec_counts = sc->idx_tmp;
    Py_ssize_t *sec_offsets = sec_counts + ns_pos;
    Py_ssize_t *sec_pts = sec_offsets + (ns_pos + 1);

    int fallback = 0;
    double cx_plane = 0.0, cy_plane = 0.0, cz_plane = 0.0;
    double nx_p = 0.0, ny_p = 0.0, nz_p = 1.0;
    double ux = 1.0, uy = 0.0, uz = 0.0;
    double vx = 0.0, vy = 1.0, vz = 0.0;
    double plane_rmse = 0.0, coverage_ratio = 0.0;
    double kasa_cx = 0.0, kasa_cy = 0.0, radius_m = 0.0, radial_rmse = 0.0;
    Py_ssize_t m_in = 0, rim_count = 0;

    Py_BEGIN_ALLOW_THREADS
    for (int i = 0; i < 3; i++) {
        if (!isfinite(up_raw[i])) { fallback = 1; break; }
    }
    double up_norm = fallback ? 0.0 : sqrt(up_raw[0]*up_raw[0] + up_raw[1]*up_raw[1] + up_raw[2]*up_raw[2]);
    if (up_norm < 1e-9) fallback = 1;

    if (!fallback) {
        double up0 = up_raw[0] / up_norm, up1 = up_raw[1] / up_norm, up2 = up_raw[2] / up_norm;
        for (Py_ssize_t i = 0; i < n; i++) {
            double x = pts[3 * i + 0], y = pts[3 * i + 1], z = pts[3 * i + 2];
            if (!isfinite(x) || !isfinite(y) || !isfinite(z)) { fallback = 1; break; }
            px[i] = x; py[i] = y; pz[i] = z;
        }
        if (!fallback) {
            const double thresh = 0.008;
            nx_p = up0; ny_p = up1; nz_p = up2;
            for (int it = 0; it < 3; it++) {
                double cov[3][3] = {{0.0}};
                if (it == 0) {
                    double w_sum = (double)n;
                    cx_plane = pairwise_sum_f64(px, n) / w_sum;
                    cy_plane = pairwise_sum_f64(py, n) / w_sum;
                    cz_plane = pairwise_sum_f64(pz, n) / w_sum;
                    double c00 = 0.0, c01 = 0.0, c02 = 0.0, c11 = 0.0, c12 = 0.0, c22 = 0.0;
                    for (Py_ssize_t i = 0; i < n; i++) {
                        double dx = px[i] - cx_plane, dy = py[i] - cy_plane, dz = pz[i] - cz_plane;
                        c00 += dx * dx; c01 += dx * dy; c02 += dx * dz;
                        c11 += dy * dy; c12 += dy * dz; c22 += dz * dz;
                    }
                    cov[0][0] = c00 / w_sum; cov[0][1] = cov[1][0] = c01 / w_sum; cov[0][2] = cov[2][0] = c02 / w_sum;
                    cov[1][1] = c11 / w_sum; cov[1][2] = cov[2][1] = c12 / w_sum; cov[2][2] = c22 / w_sum;
                } else {
                    double w_sum = pairwise_sum_f64(weights, n);
                    if (w_sum <= 1e-12) {
                        for (Py_ssize_t i = 0; i < n; i++) {
                            residuals[i] = fabs((px[i]-cx_plane)*nx_p + (py[i]-cy_plane)*ny_p + (pz[i]-cz_plane)*nz_p);
                        }
                        break;
                    }
                    double sx = 0.0, sy = 0.0, sz = 0.0;
                    for (Py_ssize_t i = 0; i < n; i++) {
                        double wi = weights[i];
                        sx += px[i] * wi; sy += py[i] * wi; sz += pz[i] * wi;
                    }
                    cx_plane = sx / w_sum; cy_plane = sy / w_sum; cz_plane = sz / w_sum;
                    double c00 = 0.0, c01 = 0.0, c02 = 0.0, c11 = 0.0, c12 = 0.0, c22 = 0.0;
                    for (Py_ssize_t i = 0; i < n; i++) {
                        double wi = weights[i];
                        double dx = px[i] - cx_plane, dy = py[i] - cy_plane, dz = pz[i] - cz_plane;
                        c00 += wi * dx * dx; c01 += wi * dx * dy; c02 += wi * dx * dz;
                        c11 += wi * dy * dy; c12 += wi * dy * dz; c22 += wi * dz * dz;
                    }
                    cov[0][0] = c00 / w_sum; cov[0][1] = cov[1][0] = c01 / w_sum; cov[0][2] = cov[2][0] = c02 / w_sum;
                    cov[1][1] = c11 / w_sum; cov[1][2] = cov[2][1] = c12 / w_sum; cov[2][2] = c22 / w_sum;
                }
                double evals[3], evecs[3][3];
                jacobi_eigh_3x3(cov, evals, evecs);
                int min_i = 0;
                if (evals[1] < evals[min_i]) min_i = 1;
                if (evals[2] < evals[min_i]) min_i = 2;
                nx_p = evecs[0][min_i]; ny_p = evecs[1][min_i]; nz_p = evecs[2][min_i];
                double nn = sqrt(nx_p*nx_p + ny_p*ny_p + nz_p*nz_p);
                if (nn < 1e-12) nn = 1e-12;
                nx_p /= nn; ny_p /= nn; nz_p /= nn;
                if (nx_p * up0 + ny_p * up1 + nz_p * up2 < 0.0) {
                    nx_p = -nx_p; ny_p = -ny_p; nz_p = -nz_p;
                }
                for (Py_ssize_t i = 0; i < n; i++) {
                    double r_val = fabs((px[i]-cx_plane)*nx_p + (py[i]-cy_plane)*ny_p + (pz[i]-cz_plane)*nz_p);
                    residuals[i] = r_val;
                    if (it + 1 < 3) {
                        weights[i] = (r_val <= thresh) ? 1.0 : (thresh / (r_val > 1e-9 ? r_val : 1e-9));
                    }
                }
            }

            Py_ssize_t n_in0 = 0;
            for (Py_ssize_t i = 0; i < n; i++) {
                uint8_t ok = (residuals[i] <= thresh) ? 1 : 0;
                inlier_mask[i] = ok;
                if (ok) {
                    in_px[n_in0] = px[i]; in_py[n_in0] = py[i]; in_pz[n_in0] = pz[i];
                    n_in0++;
                }
            }
            if (n_in0 >= 3) {
                double n_d = (double)n_in0;
                cx_plane = pairwise_sum_f64(in_px, n_in0) / n_d;
                cy_plane = pairwise_sum_f64(in_py, n_in0) / n_d;
                cz_plane = pairwise_sum_f64(in_pz, n_in0) / n_d;
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c11 = 0.0, c12 = 0.0, c22 = 0.0;
                for (Py_ssize_t i = 0; i < n_in0; i++) {
                    double dx = in_px[i] - cx_plane, dy = in_py[i] - cy_plane, dz = in_pz[i] - cz_plane;
                    c00 += dx * dx; c01 += dx * dy; c02 += dx * dz;
                    c11 += dy * dy; c12 += dy * dz; c22 += dz * dz;
                }
                double scatter[3][3] = {
                    {c00, c01, c02},
                    {c01, c11, c12},
                    {c02, c12, c22},
                };
                double evals[3], evecs[3][3];
                jacobi_eigh_3x3(scatter, evals, evecs);
                int min_i = 0;
                if (evals[1] < evals[min_i]) min_i = 1;
                if (evals[2] < evals[min_i]) min_i = 2;
                nx_p = evecs[0][min_i]; ny_p = evecs[1][min_i]; nz_p = evecs[2][min_i];
                double nn = sqrt(nx_p*nx_p + ny_p*ny_p + nz_p*nz_p);
                if (nn < 1e-12) nn = 1e-12;
                nx_p /= nn; ny_p /= nn; nz_p /= nn;
                if (nx_p * up0 + ny_p * up1 + nz_p * up2 < 0.0) {
                    nx_p = -nx_p; ny_p = -ny_p; nz_p = -nz_p;
                }
                for (Py_ssize_t i = 0; i < n; i++) {
                    double r_val = fabs((px[i]-cx_plane)*nx_p + (py[i]-cy_plane)*ny_p + (pz[i]-cz_plane)*nz_p);
                    residuals[i] = r_val;
                    inlier_mask[i] = (r_val <= thresh) ? 1 : 0;
                }
            }

            /* Final plane normalization + plane RMSE + select inlier_pts */
            double n_norm = sqrt(nx_p*nx_p + ny_p*ny_p + nz_p*nz_p);
            if (n_norm < 1e-9) {
                fallback = 1;
            } else {
                nx_p /= n_norm; ny_p /= n_norm; nz_p /= n_norm;
                Py_ssize_t n_in = 0;
                for (Py_ssize_t i = 0; i < n; i++) {
                    if (inlier_mask[i]) {
                        dbl_row[n_in] = residuals[i] * residuals[i];
                        in_px[n_in] = px[i]; in_py[n_in] = py[i]; in_pz[n_in] = pz[i];
                        n_in++;
                    }
                }
                if (n_in > 0) {
                    plane_rmse = sqrt(pairwise_sum_f64(dbl_row, n_in) / (double)n_in);
                } else {
                    for (Py_ssize_t i = 0; i < n; i++) dbl_row[i] = residuals[i] * residuals[i];
                    plane_rmse = sqrt(pairwise_sum_f64(dbl_row, n) / (double)n);
                }
                const double *src_x = px, *src_y = py, *src_z = pz;
                if (n_in >= 6) {
                    m_in = n_in; src_x = in_px; src_y = in_py; src_z = in_pz;
                } else {
                    m_in = n;
                }

                /* _orthonormal_plane_basis */
                double ax = 1.0, ay = 0.0, az = 0.0;
                if (fabs(nx_p) > 0.85) { ax = 0.0; ay = 1.0; }
                ux = ny_p * az - nz_p * ay;
                uy = nz_p * ax - nx_p * az;
                uz = nx_p * ay - ny_p * ax;
                double un = sqrt(ux*ux + uy*uy + uz*uz);
                if (un < 1e-12) {
                    fallback = 1;
                } else {
                    ux /= un; uy /= un; uz /= un;
                    vx = ny_p * uz - nz_p * uy;
                    vy = nz_p * ux - nx_p * uz;
                    vz = nx_p * uy - ny_p * ux;
                    double vn = sqrt(vx*vx + vy*vy + vz*vz);
                    vx /= vn; vy /= vn; vz /= vn;

                    /* Project and sector rim filter */
                    memset(sec_counts, 0, ns_pos * sizeof(Py_ssize_t));
                    const double pi_val = 3.14159265358979323846;
                    double step = (2.0 * pi_val) / (double)num_sectors;
                    for (int s = 0; s < num_sectors; s++) edges[s] = -pi_val + (double)s * step;
                    edges[num_sectors] = pi_val;

                    for (Py_ssize_t i = 0; i < m_in; i++) {
                        double rx = src_x[i] - cx_plane;
                        double ry = src_y[i] - cy_plane;
                        double rz = src_z[i] - cz_plane;
                        double x = rx * ux + ry * uy + rz * uz;
                        double y = rx * vx + ry * vy + rz * vz;
                        xs[i] = x;
                        ys[i] = y;
                        rads[i] = sqrt(x * x + y * y);
                        double ang = atan2(y, x);
                        int sid = (int)((ang + pi_val) / step);
                        if (sid < 0) sid = 0;
                        if (sid >= num_sectors) sid = num_sectors - 1;
                        while (sid > 0 && ang < edges[sid]) sid--;
                        while (sid + 1 < num_sectors && ang >= edges[sid + 1]) sid++;
                        if (sid == num_sectors - 1 && ang > edges[num_sectors]) sid = -1;
                        if (sid == 0 && ang < edges[0]) sid = -1;
                        sec_ids[i] = (int32_t)sid;
                        if (sid >= 0) sec_counts[sid]++;
                    }

                    Py_ssize_t cum = 0;
                    int occupied = 0;
                    for (int s = 0; s < num_sectors; s++) {
                        sec_offsets[s] = cum;
                        cum += sec_counts[s];
                        if (sec_counts[s] > 0) occupied++;
                    }
                    sec_offsets[num_sectors] = cum;
                    coverage_ratio = (double)occupied / (double)num_sectors;
                    if (coverage_ratio < min_cov) {
                        fallback = 1;
                    } else {
                        Py_ssize_t write_pos[256];
                        Py_ssize_t *wp = num_sectors <= 256 ? write_pos
                                         : (Py_ssize_t *)malloc((size_t)num_sectors * sizeof(Py_ssize_t));
                        if (!wp) {
                            fallback = 1;
                        } else {
                            for (int s = 0; s < num_sectors; s++) wp[s] = sec_offsets[s];
                            for (Py_ssize_t i = 0; i < m_in; i++) {
                                int sid = (int)sec_ids[i];
                                if (sid >= 0) sec_pts[wp[sid]++] = i;
                            }
                            if (wp != write_pos) free(wp);

                            rim_count = 0;
                            for (int s = 0; s < num_sectors; s++) {
                                Py_ssize_t start = sec_offsets[s];
                                Py_ssize_t mc = sec_counts[s];
                                if (mc <= 0) continue;
                                for (Py_ssize_t k = 0; k < mc; k++) sec_r[k] = rads[sec_pts[start + k]];
                                sort_doubles_asc(sec_r, mc);
                                double pos = 0.70 * (double)(mc - 1);
                                Py_ssize_t lo = (Py_ssize_t)floor(pos);
                                Py_ssize_t hi = (lo + 1 < mc) ? (lo + 1) : (mc - 1);
                                double frac = pos - (double)lo;
                                double q70 = sec_r[lo] + (sec_r[hi] - sec_r[lo]) * frac;
                                for (Py_ssize_t k = 0; k < mc; k++) {
                                    Py_ssize_t pid = sec_pts[start + k];
                                    if (rads[pid] >= q70) {
                                        rim_px[rim_count] = xs[pid];
                                        rim_py[rim_count] = ys[pid];
                                        rim_count++;
                                    }
                                }
                            }
                            if (rim_count < 6) {
                                memcpy(rim_px, xs, (size_t)m_in * sizeof(double));
                                memcpy(rim_py, ys, (size_t)m_in * sizeof(double));
                                rim_count = m_in;
                            }

                            /* Kasa 3x3 normal-equation circle fit */
                            double s_x = 0.0, s_y = 0.0, s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;
                            double s_r2 = 0.0, s_xr2 = 0.0, s_yr2 = 0.0;
                            for (Py_ssize_t k = 0; k < rim_count; k++) {
                                double x = rim_px[k], y = rim_py[k];
                                double r2 = x * x + y * y;
                                s_x += x; s_y += y;
                                s_xx += x * x; s_xy += x * y; s_yy += y * y;
                                s_r2 += r2; s_xr2 += x * r2; s_yr2 += y * r2;
                            }
                            double rk_d = (double)rim_count;
                            double mx = s_x / rk_d, my = s_y / rk_d;
                            double vx2 = s_xx / rk_d - mx * mx, vy2 = s_yy / rk_d - my * my;
                            double cxy2 = s_xy / rk_d - mx * my;
                            double tr2 = vx2 + vy2;
                            double det2 = vx2 * vy2 - cxy2 * cxy2;
                            if (tr2 <= 1e-18 || det2 <= 1e-16 * tr2 * tr2) {
                                fallback = 1;
                            } else {
                                double M[3][4] = {
                                    {4.0 * s_xx, 4.0 * s_xy, 2.0 * s_x, 2.0 * s_xr2},
                                    {4.0 * s_xy, 4.0 * s_yy, 2.0 * s_y, 2.0 * s_yr2},
                                    {2.0 * s_x,  2.0 * s_y,  rk_d,      s_r2},
                                };
                                double scale_m = fabs(M[0][0]) + fabs(M[1][1]) + fabs(M[2][2]) + 1.0;
                                int sing = 0;
                                for (int col = 0; col < 3; col++) {
                                    int piv = col;
                                    double best = fabs(M[col][col]);
                                    for (int ri = col + 1; ri < 3; ri++) {
                                        double v = fabs(M[ri][col]);
                                        if (v > best) { best = v; piv = ri; }
                                    }
                                    if (best <= 1e-14 * scale_m) { sing = 1; break; }
                                    if (piv != col) {
                                        for (int c = col; c < 4; c++) {
                                            double tmp = M[col][c]; M[col][c] = M[piv][c]; M[piv][c] = tmp;
                                        }
                                    }
                                    double inv = 1.0 / M[col][col];
                                    for (int ri = col + 1; ri < 3; ri++) {
                                        double fac = M[ri][col] * inv;
                                        for (int c = col; c < 4; c++) M[ri][c] -= fac * M[col][c];
                                    }
                                }
                                if (sing || fabs(M[2][2]) <= 1e-14 * scale_m) {
                                    fallback = 1;
                                } else {
                                    double sol2 = M[2][3] / M[2][2];
                                    double sol1 = (M[1][3] - M[1][2] * sol2) / M[1][1];
                                    double sol0 = (M[0][3] - M[0][1] * sol1 - M[0][2] * sol2) / M[0][0];
                                    kasa_cx = sol0;
                                    kasa_cy = sol1;
                                    double rad_sq = sol2 + kasa_cx * kasa_cx + kasa_cy * kasa_cy;
                                    if (!isfinite(kasa_cx) || !isfinite(kasa_cy) ||
                                        !isfinite(rad_sq) || rad_sq <= 1e-10) {
                                        fallback = 1;
                                    } else {
                                        radius_m = sqrt(rad_sq);
                                        if (!(radius_m >= min_rad && radius_m <= max_rad)) {
                                            fallback = 1;
                                        } else {
                                            for (Py_ssize_t k = 0; k < rim_count; k++) {
                                                double ddx = rim_px[k] - kasa_cx;
                                                double ddy = rim_py[k] - kasa_cy;
                                                double dr = sqrt(ddx * ddx + ddy * ddy) - radius_m;
                                                dbl_row[k] = dr * dr;
                                            }
                                            radial_rmse = sqrt(pairwise_sum_f64(dbl_row, rim_count) / rk_d);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (fallback) {
        ret_obj = Py_NotImplemented;
        Py_INCREF(ret_obj);
        goto cleanup;
    }

    PyObject *tri_obj = PyObject_CallFunction(triplets_fn, "nl", rim_count, 256L);
    if (!tri_obj) goto cleanup;
    Py_buffer tri_b = {0};
    if (PyObject_GetBuffer(tri_obj, &tri_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) {
        Py_DECREF(tri_obj);
        goto cleanup;
    }
    Py_ssize_t h_tri = tri_b.len / (3 * (Py_ssize_t)sizeof(int64_t));
    const int64_t *tri_ptr = (const int64_t *)tri_b.buf;

    int agreed = 0;
    double cc_x = kasa_cx, cc_y = kasa_cy, cc_r = radius_m;
    Py_BEGIN_ALLOW_THREADS
    circle_consensus_2d_core(
        rim_px, rim_py, rim_count, tri_ptr, h_tri,
        kasa_cx, kasa_cy, radius_m, 0.006, 0.015, 0.015,
        dbl_row, &agreed, &cc_x, &cc_y, &cc_r
    );
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&tri_b);
    Py_DECREF(tri_obj);

    double *c3_out = NULL, *n3_out = NULL, *cov_out = NULL;
    PyObject *c3_arr = alloc_fast_1d(3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&c3_out);
    PyObject *n3_arr = c3_arr ? alloc_fast_1d(3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&n3_out) : NULL;
    PyObject *cov_arr = n3_arr ? alloc_fast_2d(3, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&cov_out) : NULL;
    if (!c3_arr || !n3_arr || !cov_arr) {
        Py_XDECREF(c3_arr); Py_XDECREF(n3_arr); Py_XDECREF(cov_arr);
        goto cleanup;
    }

    c3_out[0] = cx_plane + kasa_cx * ux + kasa_cy * vx;
    c3_out[1] = cy_plane + kasa_cx * uy + kasa_cy * vy;
    c3_out[2] = cz_plane + kasa_cx * uz + kasa_cy * vz;
    n3_out[0] = nx_p;
    n3_out[1] = ny_p;
    n3_out[2] = nz_p;

    double in_plane_var = (radial_rmse * radial_rmse) / (double)(rim_count > 4 ? rim_count : 4);
    if (in_plane_var < 1e-6) in_plane_var = 1e-6;
    double normal_var = plane_rmse * plane_rmse;
    if (normal_var < 1e-6) normal_var = 1e-6;
    double bv[3] = {ux, uy, uz}, bw[3] = {vx, vy, vz}, bn[3] = {nx_p, ny_p, nz_p};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cov_out[3 * i + j] = in_plane_var * (bv[i] * bv[j] + bw[i] * bw[j]) + normal_var * bn[i] * bn[j];
        }
    }
    double radius_std = radial_rmse > 0.0015 ? radial_rmse : 0.0015;

    ret_obj = Py_BuildValue(
        "(OOOddddnO)",
        c3_arr, n3_arr, cov_arr,
        radius_m, radius_std, plane_rmse, coverage_ratio,
        m_in, agreed ? Py_True : Py_False
    );
    Py_DECREF(c3_arr);
    Py_DECREF(n3_arr);
    Py_DECREF(cov_arr);

cleanup:
    if (got_p) PyBuffer_Release(&pts_b);
    if (got_u) PyBuffer_Release(&up_b);
    return ret_obj;
}

static PyObject *py_aperture_sector_rim_filter(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pts_obj, *cent_obj, *u_obj, *v_obj;
    int num_sectors = 8;
    if (!PyArg_ParseTuple(args, "OOOOi", &pts_obj, &cent_obj, &u_obj, &v_obj, &num_sectors)) {
        return NULL;
    }

    Py_buffer pts_b = {0}, cent_b = {0}, u_b = {0}, v_b = {0};
    int got_p = 0, got_c = 0, got_u = 0, got_v = 0;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(pts_obj, &pts_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_p = 1;
    if (PyObject_GetBuffer(cent_obj, &cent_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_c = 1;
    if (PyObject_GetBuffer(u_obj, &u_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_u = 1;
    if (PyObject_GetBuffer(v_obj, &v_b, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_v = 1;

    Py_ssize_t n = pts_b.len / (3 * (Py_ssize_t)sizeof(double));
    const double *pts = (const double *)pts_b.buf;
    const double *cent = (const double *)cent_b.buf;
    const double *u = (const double *)u_b.buf;
    const double *v = (const double *)v_b.buf;

    size_t n_pos = (size_t)(n > 0 ? n : 1);
    size_t ns_pos = (size_t)(num_sectors > 0 ? num_sectors : 1);
    ThreadScratch *sc = get_scratch();
    if (!sc ||
        ensure_dbl_tmp(sc, (ns_pos + 1) + 4 * n_pos) < 0 ||
        ensure_p_idx(sc, n_pos) < 0 ||
        ensure_idx_tmp(sc, 2 * ns_pos + 1 + n_pos) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }

    double *edges = sc->dbl_tmp;
    double *xs = edges + (ns_pos + 1);
    double *ys = xs + n_pos;
    double *rads = ys + n_pos;
    double *sec_r = rads + n_pos;
    int32_t *sec_ids = sc->p_idx;
    Py_ssize_t *sec_counts = sc->idx_tmp;
    Py_ssize_t *sec_offsets = sec_counts + ns_pos;
    Py_ssize_t *sec_pts = sec_offsets + (ns_pos + 1);
    memset(sec_counts, 0, ns_pos * sizeof(Py_ssize_t));

    const double pi_val = 3.14159265358979323846;
    double step = (2.0 * pi_val) / (double)num_sectors;
    for (int s = 0; s < num_sectors; s++) edges[s] = -pi_val + (double)s * step;
    edges[num_sectors] = pi_val;

    int occupied = 0;
    Py_ssize_t rim_count = 0;
    int oom_wp = 0;

    Py_BEGIN_ALLOW_THREADS
    double cx0 = cent[0], cy0 = cent[1], cz0 = cent[2];
    double ux = u[0], uy = u[1], uz = u[2];
    double vx = v[0], vy = v[1], vz = v[2];
    for (Py_ssize_t i = 0; i < n; i++) {
        double rx = pts[3 * i + 0] - cx0;
        double ry = pts[3 * i + 1] - cy0;
        double rz = pts[3 * i + 2] - cz0;
        double x = rx * ux + ry * uy + rz * uz;
        double y = rx * vx + ry * vy + rz * vz;
        xs[i] = x;
        ys[i] = y;
        rads[i] = sqrt(x * x + y * y);
        double ang = atan2(y, x);
        int sid = (int)((ang + pi_val) / step);
        if (sid < 0) sid = 0;
        if (sid >= num_sectors) sid = num_sectors - 1;
        while (sid > 0 && ang < edges[sid]) sid--;
        while (sid + 1 < num_sectors && ang >= edges[sid + 1]) sid++;
        if (sid == num_sectors - 1 && ang > edges[num_sectors]) sid = -1;
        if (sid == 0 && ang < edges[0]) sid = -1;
        sec_ids[i] = (int32_t)sid;
        if (sid >= 0) sec_counts[sid]++;
    }

    Py_ssize_t cum = 0;
    for (int s = 0; s < num_sectors; s++) {
        sec_offsets[s] = cum;
        cum += sec_counts[s];
        if (sec_counts[s] > 0) occupied++;
    }
    sec_offsets[num_sectors] = cum;

    Py_ssize_t write_pos[256];
    Py_ssize_t *wp = num_sectors <= 256 ? write_pos : (Py_ssize_t *)malloc((size_t)num_sectors * sizeof(Py_ssize_t));
    if (!wp) {
        oom_wp = 1;
    } else {
        for (int s = 0; s < num_sectors; s++) wp[s] = sec_offsets[s];
        for (Py_ssize_t i = 0; i < n; i++) {
            int sid = (int)sec_ids[i];
            if (sid >= 0) sec_pts[wp[sid]++] = i;
        }
        if (wp != write_pos) free(wp);

        for (int s = 0; s < num_sectors; s++) {
            Py_ssize_t start = sec_offsets[s];
            Py_ssize_t m = sec_counts[s];
            if (m <= 0) continue;
            for (Py_ssize_t k = 0; k < m; k++) sec_r[k] = rads[sec_pts[start + k]];
            sort_doubles_asc(sec_r, m);
            double pos = 0.70 * (double)(m - 1);
            Py_ssize_t lo = (Py_ssize_t)floor(pos);
            Py_ssize_t hi = (lo + 1 < m) ? (lo + 1) : (m - 1);
            double frac = pos - (double)lo;
            double q70 = sec_r[lo] + (sec_r[hi] - sec_r[lo]) * frac;
            Py_ssize_t kept_s = 0;
            for (Py_ssize_t k = 0; k < m; k++) {
                Py_ssize_t pid = sec_pts[start + k];
                if (rads[pid] >= q70) {
                    sec_pts[start + kept_s] = pid;
                    kept_s++;
                    rim_count++;
                }
            }
            sec_counts[s] = kept_s;
        }
    }
    Py_END_ALLOW_THREADS
    if (oom_wp) { PyErr_NoMemory(); goto cleanup; }

    double *rim_out = NULL;
    double *full_out = NULL;
    PyObject *rim_arr = alloc_fast_2d(rim_count, 2, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&rim_out);
    PyObject *full_arr = rim_arr ? alloc_fast_2d(n, 2, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&full_out) : NULL;
    if (!rim_arr || !full_arr) {
        Py_XDECREF(rim_arr); Py_XDECREF(full_arr);
        goto cleanup;
    }

    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t rj = 0;
    for (int s = 0; s < num_sectors; s++) {
        Py_ssize_t start = sec_offsets[s];
        Py_ssize_t kept_s = sec_counts[s];
        for (Py_ssize_t k = 0; k < kept_s; k++) {
            Py_ssize_t pid = sec_pts[start + k];
            rim_out[2 * rj + 0] = xs[pid];
            rim_out[2 * rj + 1] = ys[pid];
            rj++;
        }
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        full_out[2 * i + 0] = xs[i];
        full_out[2 * i + 1] = ys[i];
    }
    Py_END_ALLOW_THREADS

    ret_obj = Py_BuildValue("(OiO)", rim_arr, occupied, full_arr);
    Py_DECREF(rim_arr);
    Py_DECREF(full_arr);

cleanup:
    if (got_p) PyBuffer_Release(&pts_b);
    if (got_c) PyBuffer_Release(&cent_b);
    if (got_u) PyBuffer_Release(&u_b);
    if (got_v) PyBuffer_Release(&v_b);
    return ret_obj;
}

/* ============================================================================
 * 8. is_valid_se3
 * ============================================================================ */

static PyObject *py_is_valid_se3(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pose_obj = NULL;
    double bottom_atol = 1e-6, ortho_atol = 1e-5, det_atol = 1e-5;
    if (!PyArg_ParseTuple(args, "O|ddd", &pose_obj, &bottom_atol, &ortho_atol, &det_atol)) {
        return NULL;
    }
    Py_buffer p_buf;
    if (PyObject_GetBuffer(pose_obj, &p_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) {
        PyErr_Clear();
        Py_RETURN_FALSE;
    }
    if (p_buf.itemsize != (Py_ssize_t)sizeof(double) || p_buf.len != 16 * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&p_buf);
        Py_RETURN_FALSE;
    }
    const double *m = (const double *)p_buf.buf;
    int valid = 1;
    for (int i = 0; i < 16; i++) {
        if (!isfinite(m[i])) { valid = 0; break; }
    }
    if (valid) {
        if (fabs(m[12]) > bottom_atol || fabs(m[13]) > bottom_atol ||
            fabs(m[14]) > bottom_atol || fabs(m[15] - 1.0) > bottom_atol) {
            valid = 0;
        }
    }
    if (valid) {
        for (int i = 0; i < 3 && valid; i++) {
            for (int j = 0; j < 3; j++) {
                double dot = m[0 * 4 + i] * m[0 * 4 + j]
                           + m[1 * 4 + i] * m[1 * 4 + j]
                           + m[2 * 4 + i] * m[2 * 4 + j];
                double target = (i == j) ? 1.0 : 0.0;
                if (fabs(dot - target) > ortho_atol) { valid = 0; break; }
            }
        }
    }
    if (valid) {
        double det = m[0] * (m[5] * m[10] - m[6] * m[9])
                   - m[1] * (m[4] * m[10] - m[6] * m[8])
                   + m[2] * (m[4] * m[9] - m[5] * m[8]);
        if (fabs(det - 1.0) > det_atol) valid = 0;
    }
    PyBuffer_Release(&p_buf);
    if (valid) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* ============================================================================
 * 9. build_se3_landmark_jacobians_fast
 * ============================================================================ */

static PyObject *py_build_se3_landmark_jacobians_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *poses_obj, *lms_obj, *prior_slots_obj;
    double prior_weight = 1000.0;
    PyObject *obs_ps_obj, *obs_ls_obj, *obs_w_obj;
    PyObject *btw_pa_obj, *btw_pb_obj, *btw_w_obj;
    if (!PyArg_ParseTuple(
            args, "OOOdOOOOOO",
            &poses_obj, &lms_obj, &prior_slots_obj, &prior_weight,
            &obs_ps_obj, &obs_ls_obj, &obs_w_obj,
            &btw_pa_obj, &btw_pb_obj, &btw_w_obj)) {
        return NULL;
    }

    Py_buffer pb = {0}, lb = {0}, prb = {0}, opsb = {0}, olsb = {0}, owb = {0}, bpa = {0}, bpb = {0}, bwb = {0};
    int gp = 0, gl = 0, gpr = 0, gops = 0, gols = 0, gow = 0, gbpa = 0, gbpb = 0, gbw = 0;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(poses_obj, &pb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gp = 1;
    if (PyObject_GetBuffer(lms_obj, &lb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gl = 1;
    if (PyObject_GetBuffer(prior_slots_obj, &prb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gpr = 1;
    if (PyObject_GetBuffer(obs_ps_obj, &opsb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gops = 1;
    if (PyObject_GetBuffer(obs_ls_obj, &olsb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gols = 1;
    if (PyObject_GetBuffer(obs_w_obj, &owb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gow = 1;
    if (PyObject_GetBuffer(btw_pa_obj, &bpa, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gbpa = 1;
    if (PyObject_GetBuffer(btw_pb_obj, &bpb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gbpb = 1;
    if (PyObject_GetBuffer(btw_w_obj, &bwb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gbw = 1;

    Py_ssize_t n_poses = pb.len / (16 * (Py_ssize_t)sizeof(double));
    Py_ssize_t n_lms = lb.len / (3 * (Py_ssize_t)sizeof(double));
    Py_ssize_t n_priors = prb.len / (Py_ssize_t)sizeof(int64_t);
    Py_ssize_t n_obs = opsb.len / (Py_ssize_t)sizeof(int64_t);
    Py_ssize_t n_btw = bpa.len / (Py_ssize_t)sizeof(int64_t);

    Py_ssize_t num_rows = 6 * n_priors + 3 * n_obs + 6 * n_btw;
    Py_ssize_t ax_cols = 6 * n_poses;
    Py_ssize_t ay_cols = 3 * n_lms;

    double *ax = NULL, *ay = NULL;
    PyObject *ax_arr = alloc_fast_2d(num_rows, ax_cols, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&ax);
    PyObject *ay_arr = ax_arr ? alloc_fast_2d(num_rows, ay_cols, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&ay) : NULL;
    if (!ax_arr || !ay_arr) {
        Py_XDECREF(ax_arr); Py_XDECREF(ay_arr);
        goto cleanup;
    }

    const double *poses = (const double *)pb.buf;
    const double *lms = (const double *)lb.buf;
    const int64_t *prior_slots = (const int64_t *)prb.buf;
    const int64_t *obs_ps = (const int64_t *)opsb.buf;
    const int64_t *obs_ls = (const int64_t *)olsb.buf;
    const double *obs_w = (const double *)owb.buf;
    const int64_t *btw_pa = (const int64_t *)bpa.buf;
    const int64_t *btw_pb = (const int64_t *)bpb.buf;
    const double *btw_w = (const double *)bwb.buf;

    Py_BEGIN_ALLOW_THREADS
    if (num_rows > 0 && ax_cols > 0) memset(ax, 0, (size_t)num_rows * (size_t)ax_cols * sizeof(double));
    if (num_rows > 0 && ay_cols > 0) memset(ay, 0, (size_t)num_rows * (size_t)ay_cols * sizeof(double));

    Py_ssize_t row = 0;
    for (Py_ssize_t i = 0; i < n_priors; i++) {
        Py_ssize_t p_col = 6 * (Py_ssize_t)prior_slots[i];
        for (int k = 0; k < 6; k++) {
            ax[(row + k) * ax_cols + (p_col + k)] = prior_weight;
        }
        row += 6;
    }

    for (Py_ssize_t i = 0; i < n_obs; i++) {
        Py_ssize_t ps = (Py_ssize_t)obs_ps[i];
        Py_ssize_t ls = (Py_ssize_t)obs_ls[i];
        double w = obs_w[i];
        const double *T = poses + 16 * ps;
        const double *pw = lms + 3 * ls;
        double dx = pw[0] - T[3];
        double dy = pw[1] - T[7];
        double dz = pw[2] - T[11];
        /* q_c = R_wc^T * (p_w - t_wc) */
        double qx = T[0] * dx + T[4] * dy + T[8] * dz;
        double qy = T[1] * dx + T[5] * dy + T[9] * dz;
        double qz = T[2] * dx + T[6] * dy + T[10] * dz;

        Py_ssize_t p_col = 6 * ps;
        Py_ssize_t l_col = 3 * ls;
        double *r0 = ax + (row + 0) * ax_cols + p_col;
        double *r1 = ax + (row + 1) * ax_cols + p_col;
        double *r2 = ax + (row + 2) * ax_cols + p_col;
        r0[0] = 0.0;     r0[1] = -w * qz; r0[2] = w * qy;  r0[3] = -w; r0[4] = 0.0; r0[5] = 0.0;
        r1[0] = w * qz;  r1[1] = 0.0;     r1[2] = -w * qx; r1[3] = 0.0; r1[4] = -w; r1[5] = 0.0;
        r2[0] = -w * qy; r2[1] = w * qx;  r2[2] = 0.0;     r2[3] = 0.0; r2[4] = 0.0; r2[5] = -w;

        double *y0 = ay + (row + 0) * ay_cols + l_col;
        double *y1 = ay + (row + 1) * ay_cols + l_col;
        double *y2 = ay + (row + 2) * ay_cols + l_col;
        y0[0] = w * T[0]; y0[1] = w * T[4]; y0[2] = w * T[8];
        y1[0] = w * T[1]; y1[1] = w * T[5]; y1[2] = w * T[9];
        y2[0] = w * T[2]; y2[1] = w * T[6]; y2[2] = w * T[10];
        row += 3;
    }

    for (Py_ssize_t i = 0; i < n_btw; i++) {
        Py_ssize_t pa = (Py_ssize_t)btw_pa[i];
        Py_ssize_t pb_s = (Py_ssize_t)btw_pb[i];
        double w = btw_w[i];
        const double *Ta = poses + 16 * pa;
        const double *Tb = poses + 16 * pb_s;
        /* T_ba = T_wb^{-1} * T_wa */
        double Rba[3][3], tba[3];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                Rba[r][c] = Tb[0 * 4 + r] * Ta[0 * 4 + c]
                          + Tb[1 * 4 + r] * Ta[1 * 4 + c]
                          + Tb[2 * 4 + r] * Ta[2 * 4 + c];
            }
            double dt0 = Ta[3] - Tb[3];
            double dt1 = Ta[7] - Tb[7];
            double dt2 = Ta[11] - Tb[11];
            tba[r] = Tb[0 * 4 + r] * dt0 + Tb[1 * 4 + r] * dt1 + Tb[2 * 4 + r] * dt2;
        }
        double SxR[3][3];
        double tx = tba[0], ty = tba[1], tz = tba[2];
        for (int c = 0; c < 3; c++) {
            SxR[0][c] = -tz * Rba[1][c] + ty * Rba[2][c];
            SxR[1][c] =  tz * Rba[0][c] - tx * Rba[2][c];
            SxR[2][c] = -ty * Rba[0][c] + tx * Rba[1][c];
        }
        Py_ssize_t ca = 6 * pa;
        Py_ssize_t cb = 6 * pb_s;
        for (int r = 0; r < 3; r++) {
            double *ra = ax + (row + r) * ax_cols + ca;
            double *rb = ax + (row + r) * ax_cols + cb;
            for (int c = 0; c < 3; c++) ra[c] = -w * Rba[r][c];
            rb[r] = w;
            double *ra2 = ax + (row + 3 + r) * ax_cols + ca;
            double *rb2 = ax + (row + 3 + r) * ax_cols + cb;
            for (int c = 0; c < 3; c++) {
                ra2[c] = -w * SxR[r][c];
                ra2[3 + c] = -w * Rba[r][c];
            }
            rb2[3 + r] = w;
        }
        row += 6;
    }
    Py_END_ALLOW_THREADS

    ret_obj = PyTuple_Pack(2, ax_arr, ay_arr);
    Py_DECREF(ax_arr);
    Py_DECREF(ay_arr);

cleanup:
    if (gp) PyBuffer_Release(&pb);
    if (gl) PyBuffer_Release(&lb);
    if (gpr) PyBuffer_Release(&prb);
    if (gops) PyBuffer_Release(&opsb);
    if (gols) PyBuffer_Release(&olsb);
    if (gow) PyBuffer_Release(&owb);
    if (gbpa) PyBuffer_Release(&bpa);
    if (gbpb) PyBuffer_Release(&bpb);
    if (gbw) PyBuffer_Release(&bwb);
    return ret_obj;
}

/* ============================================================================
 * 10. backproject_masked_depth
 * ============================================================================ */

/* dtype_code: 0=float64, 1=float32, 2=uint16, 3=int32 */
static PyObject *py_backproject_masked_depth(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *d_obj, *m_obj;
    Py_ssize_t height = 0, width = 0, stride = 1;
    double min_d = 0.15, max_d = 3.0, depth_scale = 1.0;
    double fx = 1.0, fy = 1.0, cx = 0.0, cy = 0.0;
    int dtype_code = 0;
    if (!PyArg_ParseTuple(
            args, "OOnnndddddddi",
            &d_obj, &m_obj, &height, &width, &stride,
            &min_d, &max_d, &depth_scale, &fx, &fy, &cx, &cy, &dtype_code)) {
        return NULL;
    }

    Py_buffer db = {0}, mb = {0};
    int gd = 0, gm = 0;
    PyObject *ret_obj = NULL;
    if (PyObject_GetBuffer(d_obj, &db, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gd = 1;
    if (PyObject_GetBuffer(m_obj, &mb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gm = 1;

    const void *depth_ptr = db.buf;
    const uint8_t *mask_ptr = (const uint8_t *)mb.buf;

    Py_ssize_t out_count = 0;
    double depth_divisor = 1.0 / depth_scale;
    int scaled_int = (dtype_code >= 2 && depth_scale != 1.0);
    double z_lo = scaled_int ? min_d * depth_divisor : min_d;
    double z_hi = scaled_int ? max_d * depth_divisor : max_d;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < height; r += stride) {
        const uint8_t *mrow = mask_ptr + r * width;
        Py_ssize_t roff = r * width;
        for (Py_ssize_t c = 0; c < width; c += stride) {
            if (!mrow[c]) continue;
            double z = 0.0;
            Py_ssize_t idx = roff + c;
            if (dtype_code == 0) z = ((const double *)depth_ptr)[idx];
            else if (dtype_code == 1) z = (double)((const float *)depth_ptr)[idx];
            else if (dtype_code == 2) z = (double)((const uint16_t *)depth_ptr)[idx];
            else z = (double)((const int32_t *)depth_ptr)[idx];
            if (isfinite(z) && z >= z_lo && z <= z_hi && z > 0.0) out_count++;
        }
    }
    Py_END_ALLOW_THREADS

    double *out = NULL;
    PyObject *out_arr = alloc_fast_2d(out_count, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out);
    if (!out_arr) goto cleanup;

    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t k = 0;
    for (Py_ssize_t r = 0; r < height; r += stride) {
        const uint8_t *mrow = mask_ptr + r * width;
        Py_ssize_t roff = r * width;
        double dy = (double)r - cy;
        for (Py_ssize_t c = 0; c < width; c += stride) {
            if (!mrow[c]) continue;
            double raw_z = 0.0;
            Py_ssize_t idx = roff + c;
            if (dtype_code == 0) raw_z = ((const double *)depth_ptr)[idx];
            else if (dtype_code == 1) raw_z = (double)((const float *)depth_ptr)[idx];
            else if (dtype_code == 2) raw_z = (double)((const uint16_t *)depth_ptr)[idx];
            else raw_z = (double)((const int32_t *)depth_ptr)[idx];
            if (isfinite(raw_z) && raw_z >= z_lo && raw_z <= z_hi && raw_z > 0.0) {
                double z = scaled_int ? (raw_z / depth_divisor) : raw_z;
                out[3 * k + 0] = ((double)c - cx) * z / fx;
                out[3 * k + 1] = dy * z / fy;
                out[3 * k + 2] = z;
                k++;
            }
        }
    }
    Py_END_ALLOW_THREADS

    ret_obj = out_arr;

cleanup:
    if (gd) PyBuffer_Release(&db);
    if (gm) PyBuffer_Release(&mb);
    return ret_obj;
}

/* ============================================================================
 * 11. splat_synthetic_rgbd
 * ============================================================================ */

static PyObject *py_splat_synthetic_rgbd(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *rows_obj, *cols_obj, *z_obj, *d_obj, *m_obj, *rgb_obj;
    Py_ssize_t height = 0, width = 0;
    unsigned int cr = 60, cg = 140, cb = 220;
    if (!PyArg_ParseTuple(args, "OOOOOOnnIII", &rows_obj, &cols_obj, &z_obj,
                          &d_obj, &m_obj, &rgb_obj, &height, &width, &cr, &cg, &cb)) {
        return NULL;
    }

    Py_buffer rb = {0}, cb_buf = {0}, zb = {0}, db = {0}, mb = {0}, rgbb = {0};
    int gr = 0, gc = 0, gz = 0, gd = 0, gm = 0, grgb = 0;
    if (PyObject_GetBuffer(rows_obj, &rb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gr = 1;
    if (PyObject_GetBuffer(cols_obj, &cb_buf, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gc = 1;
    if (PyObject_GetBuffer(z_obj, &zb, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gz = 1;
    if (PyObject_GetBuffer(d_obj, &db, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gd = 1;
    if (PyObject_GetBuffer(m_obj, &mb, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    gm = 1;
    if (PyObject_GetBuffer(rgb_obj, &rgbb, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    grgb = 1;

    Py_ssize_t n = zb.len / (Py_ssize_t)sizeof(double);
    const int64_t *rows = (const int64_t *)rb.buf;
    const int64_t *cols = (const int64_t *)cb_buf.buf;
    const double *zm = (const double *)zb.buf;
    uint16_t *depth_mm = (uint16_t *)db.buf;
    uint8_t *mask = (uint8_t *)mb.buf;
    uint8_t *rgb = (uint8_t *)rgbb.buf;
    uint8_t ur = (uint8_t)cr, ug = (uint8_t)cg, ub = (uint8_t)cb;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        int64_t r = rows[i];
        int64_t c = cols[i];
        /* Match Python's round(float * 1000.0) round-half-to-even via rint() */
        uint16_t z_mm = (uint16_t)(int64_t)rint(zm[i] * 1000.0);
        for (int dr = 0; dr <= 1; dr++) {
            Py_ssize_t rr = (Py_ssize_t)(r + dr);
            Py_ssize_t base = rr * width + (Py_ssize_t)c;
            for (int dc = 0; dc <= 1; dc++) {
                Py_ssize_t idx = base + dc;
                if (depth_mm[idx] == 0 || z_mm < depth_mm[idx]) {
                    depth_mm[idx] = z_mm;
                }
                mask[idx] = 1;
                rgb[3 * idx + 0] = ur;
                rgb[3 * idx + 1] = ug;
                rgb[3 * idx + 2] = ub;
            }
        }
    }
    Py_END_ALLOW_THREADS

cleanup:
    if (gr) PyBuffer_Release(&rb);
    if (gc) PyBuffer_Release(&cb_buf);
    if (gz) PyBuffer_Release(&zb);
    if (gd) PyBuffer_Release(&db);
    if (gm) PyBuffer_Release(&mb);
    if (grgb) PyBuffer_Release(&rgbb);
    if (PyErr_Occurred()) return NULL;
    Py_RETURN_NONE;
}

/* ============================================================================
 * 12. evaluate_schur_gate_fast (Hestenes One-Sided Jacobi SVD + Jacobi Eigh)
 * ============================================================================ */

static void jacobi_eigh_sym_small(int d, double *a, double *evals, double *evecs) {
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            evecs[i * d + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int sweep = 0; sweep < 20; sweep++) {
        double off = 0.0, diag = 1.0;
        for (int i = 0; i < d; i++) {
            diag += fabs(a[i * d + i]);
            for (int j = i + 1; j < d; j++) off += fabs(a[i * d + j]);
        }
        if (off <= 1e-32 * diag) break;
        for (int p = 0; p < d - 1; p++) {
            for (int q = p + 1; q < d; q++) {
                double apq = a[p * d + q];
                if (fabs(apq) <= 1e-32 * diag) continue;
                double tau = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                double t = (tau >= 0.0)
                    ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                    : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;
                a[p * d + p] -= t * apq;
                a[q * d + q] += t * apq;
                a[p * d + q] = a[q * d + p] = 0.0;
                for (int r = 0; r < d; r++) {
                    if (r == p || r == q) continue;
                    double arp = a[r * d + p], arq = a[r * d + q];
                    a[r * d + p] = a[p * d + r] = c * arp - s * arq;
                    a[r * d + q] = a[q * d + r] = s * arp + c * arq;
                }
                for (int r = 0; r < d; r++) {
                    double vp = evecs[r * d + p], vq = evecs[r * d + q];
                    evecs[r * d + p] = c * vp - s * vq;
                    evecs[r * d + q] = s * vp + c * vq;
                }
            }
        }
    }
    for (int i = 0; i < d; i++) evals[i] = a[i * d + i];
}

#define SCHUR_MAX_COLS 192

static void hestenes_svd(double *w, Py_ssize_t m, Py_ssize_t n, double *s_out, int need_u) {
    if (n <= 0) return;
    if (n == 1) {
        double sum = 0.0;
        for (Py_ssize_t r = 0; r < m; r++) sum += w[r] * w[r];
        double sn = sqrt(sum);
        s_out[0] = sn;
        if (need_u && sn > 0.0) {
            double inv = 1.0 / sn;
            for (Py_ssize_t r = 0; r < m; r++) w[r] *= inv;
        }
        return;
    }

    for (int sweep = 0; sweep < 25; sweep++) {
        int rotated = 0;
        for (Py_ssize_t p = 0; p < n - 1; p++) {
            for (Py_ssize_t q = p + 1; q < n; q++) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (Py_ssize_t r = 0; r < m; r++) {
                    double wp = w[r * n + p];
                    double wq = w[r * n + q];
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                double prod = sqrt(alpha * beta);
                if (prod <= 0.0 || fabs(gamma) <= 1e-15 * prod) continue;
                rotated = 1;
                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = (zeta >= 0.0)
                    ? (1.0 / (zeta + sqrt(1.0 + zeta * zeta)))
                    : (-1.0 / (-zeta + sqrt(1.0 + zeta * zeta)));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s_rot = c * t;
                for (Py_ssize_t r = 0; r < m; r++) {
                    double wp = w[r * n + p];
                    double wq = w[r * n + q];
                    w[r * n + p] = c * wp - s_rot * wq;
                    w[r * n + q] = s_rot * wp + c * wq;
                }
            }
        }
        if (!rotated) break;
    }

    double col_norms[SCHUR_MAX_COLS];
    int order[SCHUR_MAX_COLS];
    for (Py_ssize_t k = 0; k < n; k++) {
        double sum = 0.0;
        for (Py_ssize_t r = 0; r < m; r++) {
            double v = w[r * n + k];
            sum += v * v;
        }
        col_norms[k] = sqrt(sum);
        order[k] = (int)k;
    }
    for (Py_ssize_t i = 0; i < n - 1; i++) {
        for (Py_ssize_t j = i + 1; j < n; j++) {
            if (col_norms[order[j]] > col_norms[order[i]]) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }
    if (need_u) {
        double inv_s[SCHUR_MAX_COLS];
        for (Py_ssize_t k = 0; k < n; k++) {
            double sv = col_norms[order[k]];
            s_out[k] = sv;
            inv_s[k] = (sv > 0.0) ? (1.0 / sv) : 0.0;
        }
        double tmp_r[SCHUR_MAX_COLS];
        for (Py_ssize_t r = 0; r < m; r++) {
            double *row_p = w + r * n;
            for (Py_ssize_t k = 0; k < n; k++) tmp_r[k] = row_p[order[k]] * inv_s[k];
            for (Py_ssize_t k = 0; k < n; k++) row_p[k] = tmp_r[k];
        }
    } else {
        for (Py_ssize_t k = 0; k < n; k++) s_out[k] = col_norms[order[k]];
    }
}

static PyObject *py_evaluate_schur_gate_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "evaluate_schur_gate_fast expects (ax, ay, candidate_pose_index)");
        return NULL;
    }
    long cand_idx = PyLong_AsLong(args[2]);
    if (cand_idx == -1 && PyErr_Occurred()) return NULL;

    Py_buffer ax_b = {0}, ay_b = {0};
    int got_ax = 0, got_ay = 0;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(args[0], &ax_b, PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_ax = 1;
    if (ax_b.ndim != 2 || ax_b.itemsize != (Py_ssize_t)sizeof(double)) {
        ret_obj = Py_NotImplemented; Py_INCREF(ret_obj); goto cleanup;
    }
    Py_ssize_t m = ax_b.shape[0];
    Py_ssize_t p_cols = ax_b.shape[1];
    if (p_cols <= 0 || p_cols > SCHUR_MAX_COLS || (p_cols % 6) != 0 || m < p_cols ||
        cand_idx < 0 || cand_idx >= p_cols / 6) {
        ret_obj = Py_NotImplemented; Py_INCREF(ret_obj); goto cleanup;
    }

    int has_ay = (args[1] != Py_None);
    Py_ssize_t l_cols = 0;
    if (has_ay) {
        if (PyObject_GetBuffer(args[1], &ay_b, PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        got_ay = 1;
        if (ay_b.ndim != 2 || ay_b.itemsize != (Py_ssize_t)sizeof(double) || ay_b.shape[0] != m) {
            ret_obj = Py_NotImplemented; Py_INCREF(ret_obj); goto cleanup;
        }
        l_cols = ay_b.shape[1];
        if (l_cols == 0) {
            has_ay = 0;
        } else if ((l_cols % 3) != 0) {
            ret_obj = Py_NotImplemented; Py_INCREF(ret_obj); goto cleanup;
        }
    }

    const double *ax = (const double *)ax_b.buf;
    const double *ay = has_ay ? (const double *)ay_b.buf : NULL;
    Py_ssize_t num_lm = l_cols / 3;

    ThreadScratch *sc = get_scratch();
    size_t need_dbl = (size_t)m * (size_t)(2 * p_cols + 9) + (size_t)(l_cols > 0 ? l_cols : 1);
    size_t need_idx = has_ay ? (size_t)(2 * m + 3 * num_lm) : 1;
    if (!sc || ensure_dbl_tmp(sc, need_dbl) < 0 || ensure_idx_tmp(sc, need_idx) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }

    double *a_bar_x = sc->dbl_tmp;
    double *w_work = a_bar_x + m * p_cols;
    double *a_i_marg = w_work + m * p_cols;
    double *u_all = a_i_marg + m * 6;
    double *s_all = u_all + m * 3;

    Py_ssize_t o_cols = p_cols - 6;
    double s_full_local[SCHUR_MAX_COLS] = {0.0};
    double s_o_local[SCHUR_MAX_COLS] = {0.0};
    double h_ii_local[36] = {0.0};
    double r_i_local[9] = {0.0};
    double trans_min_sv = 0.0, trans_max_sv = 0.0;
    double pose_min_sv = 0.0, pose_max_sv = 0.0;
    int fallback = 0;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < m * p_cols; i++) {
        if (!isfinite(ax[i])) { fallback = 1; break; }
    }

    if (!fallback && has_ay) {
        Py_ssize_t *row_owner = sc->idx_tmp;
        Py_ssize_t *blk_rows = row_owner + m;
        Py_ssize_t *blk_cnt = blk_rows + m;
        Py_ssize_t *blk_off = blk_cnt + num_lm;
        memset(blk_cnt, 0, (size_t)num_lm * sizeof(Py_ssize_t));

        for (Py_ssize_t r = 0; r < m && !fallback; r++) {
            row_owner[r] = -1;
            const double *row_p = ay + r * l_cols;
            for (Py_ssize_t b = 0; b < num_lm; b++) {
                double v0 = row_p[3 * b + 0], v1 = row_p[3 * b + 1], v2 = row_p[3 * b + 2];
                if (!isfinite(v0) || !isfinite(v1) || !isfinite(v2)) {
                    fallback = 1; break;
                }
                if (v0 != 0.0 || v1 != 0.0 || v2 != 0.0) {
                    if (row_owner[r] != -1) {
                        fallback = 1; break;
                    }
                    row_owner[r] = b;
                    blk_cnt[b]++;
                }
            }
        }
        if (!fallback) {
            Py_ssize_t cum = 0;
            for (Py_ssize_t b = 0; b < num_lm; b++) {
                if (blk_cnt[b] > 0 && blk_cnt[b] < 3) {
                    fallback = 1; break;
                }
                blk_off[b] = cum;
                cum += blk_cnt[b];
            }
        }
        if (!fallback) {
            for (Py_ssize_t r = 0; r < m; r++) {
                Py_ssize_t b = row_owner[r];
                if (b >= 0) blk_rows[blk_off[b]++] = r;
            }
            for (Py_ssize_t b = 0; b < num_lm; b++) blk_off[b] -= blk_cnt[b];

            memcpy(a_bar_x, ax, (size_t)m * (size_t)p_cols * sizeof(double));
            double max_s = 0.0;
            for (Py_ssize_t b = 0; b < num_lm; b++) {
                Py_ssize_t mb = blk_cnt[b];
                if (mb == 0) continue;
                Py_ssize_t off = blk_off[b];
                double *ub = u_all + off * 3;
                for (Py_ssize_t k = 0; k < mb; k++) {
                    const double *src = ay + blk_rows[off + k] * l_cols + 3 * b;
                    ub[3 * k + 0] = src[0];
                    ub[3 * k + 1] = src[1];
                    ub[3 * k + 2] = src[2];
                }
                hestenes_svd(ub, mb, 3, s_all + 3 * b, 1);
                if (s_all[3 * b] > max_s) max_s = s_all[3 * b];
            }

            Py_ssize_t max_y_dim = m > l_cols ? m : l_cols;
            double tol_y = DBL_EPSILON * (double)max_y_dim * max_s;
            for (Py_ssize_t b = 0; b < num_lm; b++) {
                Py_ssize_t mb = blk_cnt[b];
                if (mb == 0) continue;
                int rb = 0;
                for (int k = 0; k < 3; k++) {
                    if (s_all[3 * b + k] > tol_y) rb++;
                }
                if (rb == 0) continue;
                Py_ssize_t off = blk_off[b];
                const double *ub = u_all + off * 3;
                for (Py_ssize_t c = 0; c < p_cols; c++) {
                    double d0 = 0.0, d1 = 0.0, d2 = 0.0;
                    for (Py_ssize_t k = 0; k < mb; k++) {
                        double v = a_bar_x[blk_rows[off + k] * p_cols + c];
                        d0 += ub[3 * k + 0] * v;
                        if (rb >= 2) d1 += ub[3 * k + 1] * v;
                        if (rb == 3) d2 += ub[3 * k + 2] * v;
                    }
                    for (Py_ssize_t k = 0; k < mb; k++) {
                        double sub = ub[3 * k + 0] * d0;
                        if (rb >= 2) sub += ub[3 * k + 1] * d1;
                        if (rb == 3) sub += ub[3 * k + 2] * d2;
                        a_bar_x[blk_rows[off + k] * p_cols + c] -= sub;
                    }
                }
            }
        }
    } else if (!fallback) {
        memcpy(a_bar_x, ax, (size_t)m * (size_t)p_cols * sizeof(double));
    }

    if (!fallback) {
        memcpy(w_work, a_bar_x, (size_t)m * (size_t)p_cols * sizeof(double));
        hestenes_svd(w_work, m, p_cols, s_full_local, 0);

        Py_ssize_t col_start = 6 * cand_idx;
        for (Py_ssize_t r = 0; r < m; r++) {
            const double *src = a_bar_x + r * p_cols + col_start;
            double *dst = a_i_marg + r * 6;
            for (int c = 0; c < 6; c++) dst[c] = src[c];
        }

        if (o_cols > 0) {
            for (Py_ssize_t r = 0; r < m; r++) {
                const double *src = a_bar_x + r * p_cols;
                double *dst = w_work + r * o_cols;
                Py_ssize_t dc = 0;
                for (Py_ssize_t c = 0; c < col_start; c++) dst[dc++] = src[c];
                for (Py_ssize_t c = col_start + 6; c < p_cols; c++) dst[dc++] = src[c];
            }
            hestenes_svd(w_work, m, o_cols, s_o_local, 1);

            Py_ssize_t max_o_dim = m > o_cols ? m : o_cols;
            double tol_o = DBL_EPSILON * (double)max_o_dim * s_o_local[0];
            Py_ssize_t rank_o = 0;
            for (Py_ssize_t k = 0; k < o_cols; k++) {
                if (s_o_local[k] > tol_o) rank_o++;
            }
            if (rank_o > 0) {
                double dots[SCHUR_MAX_COLS][6];
                for (Py_ssize_t k = 0; k < rank_o; k++) {
                    for (int c = 0; c < 6; c++) dots[k][c] = 0.0;
                }
                for (Py_ssize_t r = 0; r < m; r++) {
                    const double *ur = w_work + r * o_cols;
                    const double *ar = a_i_marg + r * 6;
                    for (Py_ssize_t k = 0; k < rank_o; k++) {
                        double uk = ur[k];
                        for (int c = 0; c < 6; c++) dots[k][c] += uk * ar[c];
                    }
                }
                for (Py_ssize_t r = 0; r < m; r++) {
                    const double *ur = w_work + r * o_cols;
                    double *ar = a_i_marg + r * 6;
                    for (int c = 0; c < 6; c++) {
                        double sub = 0.0;
                        for (Py_ssize_t k = 0; k < rank_o; k++) sub += ur[k] * dots[k][c];
                        ar[c] -= sub;
                    }
                }
            }
        }

        double h_raw[36] = {0.0};
        for (Py_ssize_t r = 0; r < m; r++) {
            const double *ar = a_i_marg + r * 6;
            for (int i = 0; i < 6; i++) {
                double ai = ar[i];
                for (int j = i; j < 6; j++) h_raw[i * 6 + j] += ai * ar[j];
            }
        }
        for (int i = 0; i < 6; i++) {
            h_ii_local[i * 6 + i] = h_raw[i * 6 + i];
            for (int j = i + 1; j < 6; j++) {
                h_ii_local[i * 6 + j] = h_ii_local[j * 6 + i] = h_raw[i * 6 + j];
            }
        }

        double h_rr[9], eval_rr[3], evec_rr[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) h_rr[i * 3 + j] = h_ii_local[i * 6 + j];
        }
        jacobi_eigh_sym_small(3, h_rr, eval_rr, evec_rr);
        double max_ev_rr = 0.0;
        for (int k = 0; k < 3; k++) {
            double ae = fabs(eval_rr[k]);
            if (ae > max_ev_rr) max_ev_rr = ae;
        }
        double cutoff = 1e-12 * max_ev_rr;
        double pinv_rr[9] = {0.0};
        for (int k = 0; k < 3; k++) {
            if (fabs(eval_rr[k]) > cutoff) {
                double inv_ev = 1.0 / eval_rr[k];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        pinv_rr[i * 3 + j] += inv_ev * evec_rr[i * 3 + k] * evec_rr[j * 3 + k];
                    }
                }
            }
        }

        double tmp_3x3[9] = {0.0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double acc = 0.0;
                for (int k = 0; k < 3; k++) {
                    acc += h_ii_local[(3 + i) * 6 + k] * pinv_rr[k * 3 + j];
                }
                tmp_3x3[i * 3 + j] = acc;
            }
        }
        double r_raw[9] = {0.0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double acc = 0.0;
                for (int k = 0; k < 3; k++) {
                    acc += tmp_3x3[i * 3 + k] * h_ii_local[k * 6 + (3 + j)];
                }
                r_raw[i * 3 + j] = h_ii_local[(3 + i) * 6 + (3 + j)] - acc;
            }
        }
        for (int i = 0; i < 3; i++) {
            r_i_local[i * 3 + i] = r_raw[i * 3 + i];
            for (int j = i + 1; j < 3; j++) {
                double avg = 0.5 * (r_raw[i * 3 + j] + r_raw[j * 3 + i]);
                r_i_local[i * 3 + j] = r_i_local[j * 3 + i] = avg;
            }
        }

        double r_work[9], eval_ri[3], evec_ri[9];
        memcpy(r_work, r_i_local, sizeof(r_work));
        jacobi_eigh_sym_small(3, r_work, eval_ri, evec_ri);
        trans_min_sv = HUGE_VAL;
        trans_max_sv = 0.0;
        for (int k = 0; k < 3; k++) {
            double sv = fabs(eval_ri[k]);
            if (sv < trans_min_sv) trans_min_sv = sv;
            if (sv > trans_max_sv) trans_max_sv = sv;
        }

        double h_work[36], eval_h[6], evec_h[36];
        memcpy(h_work, h_ii_local, sizeof(h_work));
        jacobi_eigh_sym_small(6, h_work, eval_h, evec_h);
        pose_min_sv = HUGE_VAL;
        pose_max_sv = 0.0;
        for (int k = 0; k < 6; k++) {
            double sv = fabs(eval_h[k]);
            if (sv < pose_min_sv) pose_min_sv = sv;
            if (sv > pose_max_sv) pose_max_sv = sv;
        }
    }
    Py_END_ALLOW_THREADS

    if (fallback) {
        ret_obj = Py_NotImplemented; Py_INCREF(ret_obj); goto cleanup;
    }

    double *sf_out = NULL, *so_out = NULL, *h_out = NULL, *r_out = NULL;
    PyObject *sf_arr = alloc_fast_1d(p_cols, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&sf_out);
    PyObject *so_arr = sf_arr ? alloc_fast_1d(o_cols, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&so_out) : NULL;
    PyObject *h_arr = so_arr ? alloc_fast_2d(6, 6, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&h_out) : NULL;
    PyObject *r_arr = h_arr ? alloc_fast_2d(3, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&r_out) : NULL;
    if (!sf_arr || !so_arr || !h_arr || !r_arr) {
        Py_XDECREF(sf_arr); Py_XDECREF(so_arr); Py_XDECREF(h_arr); Py_XDECREF(r_arr);
        goto cleanup;
    }
    memcpy(sf_out, s_full_local, (size_t)p_cols * sizeof(double));
    if (o_cols > 0) memcpy(so_out, s_o_local, (size_t)o_cols * sizeof(double));
    memcpy(h_out, h_ii_local, 36 * sizeof(double));
    memcpy(r_out, r_i_local, 9 * sizeof(double));

    ret_obj = Py_BuildValue(
        "(OOOOdddd)",
        sf_arr, so_arr, h_arr, r_arr,
        trans_min_sv, trans_max_sv, pose_min_sv, pose_max_sv
    );
    Py_DECREF(sf_arr);
    Py_DECREF(so_arr);
    Py_DECREF(h_arr);
    Py_DECREF(r_arr);

cleanup:
    if (got_ax) PyBuffer_Release(&ax_b);
    if (got_ay) PyBuffer_Release(&ay_b);
    return ret_obj;
}

/* ============================================================================
 * 13. nn_min_dists_3d (Flat Median-Split C11/SIMD 3D KD-Tree)
 * ============================================================================ */

#define KD_LEAF_SIZE 16

static int32_t kd_build_recursive(
    const double *ref, int32_t *perm, KDNode *nodes, int32_t *next_id,
    int32_t start, int32_t end)
{
    int32_t nid = (*next_id)++;
    KDNode *nd = &nodes[nid];
    nd->start = start;
    nd->end = end;

    double bmin0 = ref[3 * (Py_ssize_t)perm[start] + 0], bmax0 = bmin0;
    double bmin1 = ref[3 * (Py_ssize_t)perm[start] + 1], bmax1 = bmin1;
    double bmin2 = ref[3 * (Py_ssize_t)perm[start] + 2], bmax2 = bmin2;
    for (int32_t k = start + 1; k < end; k++) {
        const double *p = ref + 3 * (Py_ssize_t)perm[k];
        double x = p[0], y = p[1], z = p[2];
        if (x < bmin0) bmin0 = x; else if (x > bmax0) bmax0 = x;
        if (y < bmin1) bmin1 = y; else if (y > bmax1) bmax1 = y;
        if (z < bmin2) bmin2 = z; else if (z > bmax2) bmax2 = z;
    }
    nd->bmin[0] = bmin0; nd->bmin[1] = bmin1; nd->bmin[2] = bmin2;
    nd->bmax[0] = bmax0; nd->bmax[1] = bmax1; nd->bmax[2] = bmax2;

    if (end - start <= KD_LEAF_SIZE) {
        nd->axis = -1;
        nd->left = nd->right = -1;
        return nid;
    }

    double e0 = bmax0 - bmin0, e1 = bmax1 - bmin1, e2 = bmax2 - bmin2;
    int32_t ax = (e0 >= e1 && e0 >= e2) ? 0 : ((e1 >= e2) ? 1 : 2);
    double span = (ax == 0) ? e0 : ((ax == 1) ? e1 : e2);
    if (span <= 0.0) {
        nd->axis = -1;
        nd->left = nd->right = -1;
        return nid;
    }

    int32_t mid = start + ((end - start) >> 1);
    int32_t lo = start, hi = end - 1;
    while (lo < hi) {
        int32_t piv_idx = lo + ((hi - lo) >> 1);
        double a_v = ref[3 * (Py_ssize_t)perm[lo] + ax];
        double m_v = ref[3 * (Py_ssize_t)perm[piv_idx] + ax];
        if (a_v > m_v) { int32_t t = perm[lo]; perm[lo] = perm[piv_idx]; perm[piv_idx] = t; }
        double b_v = ref[3 * (Py_ssize_t)perm[hi] + ax];
        if (ref[3 * (Py_ssize_t)perm[lo] + ax] > b_v) { int32_t t = perm[lo]; perm[lo] = perm[hi]; perm[hi] = t; }
        if (ref[3 * (Py_ssize_t)perm[piv_idx] + ax] > ref[3 * (Py_ssize_t)perm[hi] + ax]) {
            int32_t t = perm[piv_idx]; perm[piv_idx] = perm[hi]; perm[hi] = t;
        }
        double piv = ref[3 * (Py_ssize_t)perm[piv_idx] + ax];
        int32_t i = lo, j = hi;
        while (i <= j) {
            while (ref[3 * (Py_ssize_t)perm[i] + ax] < piv) i++;
            while (ref[3 * (Py_ssize_t)perm[j] + ax] > piv) j--;
            if (i <= j) {
                int32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
                i++; j--;
            }
        }
        if (j < mid) lo = i;
        if (i > mid) hi = j;
    }

    nd->axis = ax;
    nd->left = kd_build_recursive(ref, perm, nodes, next_id, start, mid);
    nd->right = kd_build_recursive(ref, perm, nodes, next_id, mid, end);
    return nid;
}

static inline double point_box_d2(double qx, double qy, double qz, const double bmin[3], const double bmax[3]) {
    double ox = (qx < bmin[0]) ? (bmin[0] - qx) : ((qx > bmax[0]) ? (qx - bmax[0]) : 0.0);
    double oy = (qy < bmin[1]) ? (bmin[1] - qy) : ((qy > bmax[1]) ? (qy - bmax[1]) : 0.0);
    double oz = (qz < bmin[2]) ? (bmin[2] - qz) : ((qz > bmax[2]) ? (qz - bmax[2]) : 0.0);
    return ox * ox + oy * oy + oz * oz;
}

static inline double leaf_min_d2_scalar(
    const double *rx, const double *ry, const double *rz,
    int32_t start, int32_t end,
    double qx, double qy, double qz, double best_d2)
{
    for (int32_t k = start; k < end; k++) {
        double dx = rx[k] - qx;
        double dy = ry[k] - qy;
        double dz = rz[k] - qz;
        double d2 = (dx * dx + dy * dy) + dz * dz;
        if (d2 < best_d2) best_d2 = d2;
    }
    return best_d2;
}

#if ES_HAVE_X86_RUNTIME_SIMD
__attribute__((target("avx512f,avx512dq,avx512vl")))
static void nn_min_dists_query_avx512(
    const double *rx, const double *ry, const double *rz,
    const KDNode *nodes, int use_tree, int32_t m32,
    const double *qry, Py_ssize_t n, double *out)
{
    typedef struct { int32_t nid; double d2; } StackItem;
    StackItem stk[64];
    for (Py_ssize_t q = 0; q < n; q++) {
        double qx = qry[3 * q + 0], qy = qry[3 * q + 1], qz = qry[3 * q + 2];
        double best_d2 = HUGE_VAL;
        __m512d v_qx = _mm512_set1_pd(qx);
        __m512d v_qy = _mm512_set1_pd(qy);
        __m512d v_qz = _mm512_set1_pd(qz);
        if (!use_tree) {
            int32_t k = 0;
            if (m32 >= 8) {
                __m512d v_min = _mm512_set1_pd(best_d2);
                for (; k + 8 <= m32; k += 8) {
                    __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(rx + k), v_qx);
                    __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(ry + k), v_qy);
                    __m512d dz = _mm512_sub_pd(_mm512_loadu_pd(rz + k), v_qz);
                    __m512d d2 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)), _mm512_mul_pd(dz, dz));
                    v_min = _mm512_min_pd(v_min, d2);
                }
                double m_red = _mm512_reduce_min_pd(v_min);
                if (m_red < best_d2) best_d2 = m_red;
            }
            for (; k < m32; k++) {
                double dx = rx[k] - qx, dy = ry[k] - qy, dz = rz[k] - qz;
                double d2 = (dx * dx + dy * dy) + dz * dz;
                if (d2 < best_d2) best_d2 = d2;
            }
        } else {
            int sp = 0;
            stk[sp].nid = 0;
            stk[sp].d2 = point_box_d2(qx, qy, qz, nodes[0].bmin, nodes[0].bmax);
            sp++;
            while (sp > 0) {
                StackItem cur = stk[--sp];
                if (cur.d2 >= best_d2) continue;
                const KDNode *nd = &nodes[cur.nid];
                if (nd->axis < 0) {
                    int32_t k = nd->start, end = nd->end;
                    if (end - k >= 8) {
                        __m512d v_min = _mm512_set1_pd(best_d2);
                        for (; k + 8 <= end; k += 8) {
                            __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(rx + k), v_qx);
                            __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(ry + k), v_qy);
                            __m512d dz = _mm512_sub_pd(_mm512_loadu_pd(rz + k), v_qz);
                            __m512d d2 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)), _mm512_mul_pd(dz, dz));
                            v_min = _mm512_min_pd(v_min, d2);
                        }
                        double m_red = _mm512_reduce_min_pd(v_min);
                        if (m_red < best_d2) best_d2 = m_red;
                    }
                    for (; k < end; k++) {
                        double dx = rx[k] - qx, dy = ry[k] - qy, dz = rz[k] - qz;
                        double d2 = (dx * dx + dy * dy) + dz * dz;
                        if (d2 < best_d2) best_d2 = d2;
                    }
                    continue;
                }
                int32_t lid = nd->left, rid = nd->right;
                double ld2 = point_box_d2(qx, qy, qz, nodes[lid].bmin, nodes[lid].bmax);
                double rd2 = point_box_d2(qx, qy, qz, nodes[rid].bmin, nodes[rid].bmax);
                if (ld2 <= rd2) {
                    if (rd2 < best_d2 && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                    if (ld2 < best_d2 && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                } else {
                    if (ld2 < best_d2 && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                    if (rd2 < best_d2 && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                }
            }
        }
        out[q] = sqrt(best_d2);
    }
}

__attribute__((target("avx2")))
static void nn_min_dists_query_avx2(
    const double *rx, const double *ry, const double *rz,
    const KDNode *nodes, int use_tree, int32_t m32,
    const double *qry, Py_ssize_t n, double *out)
{
    typedef struct { int32_t nid; double d2; } StackItem;
    StackItem stk[64];
    for (Py_ssize_t q = 0; q < n; q++) {
        double qx = qry[3 * q + 0], qy = qry[3 * q + 1], qz = qry[3 * q + 2];
        double best_d2 = HUGE_VAL;
        __m256d v_qx = _mm256_set1_pd(qx);
        __m256d v_qy = _mm256_set1_pd(qy);
        __m256d v_qz = _mm256_set1_pd(qz);
        if (!use_tree) {
            int32_t k = 0;
            if (m32 >= 4) {
                __m256d v_min = _mm256_set1_pd(best_d2);
                for (; k + 4 <= m32; k += 4) {
                    __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(rx + k), v_qx);
                    __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(ry + k), v_qy);
                    __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(rz + k), v_qz);
                    __m256d d2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)), _mm256_mul_pd(dz, dz));
                    v_min = _mm256_min_pd(v_min, d2);
                }
                double tmp[4];
                _mm256_storeu_pd(tmp, v_min);
                for (int i = 0; i < 4; i++) if (tmp[i] < best_d2) best_d2 = tmp[i];
            }
            for (; k < m32; k++) {
                double dx = rx[k] - qx, dy = ry[k] - qy, dz = rz[k] - qz;
                double d2 = (dx * dx + dy * dy) + dz * dz;
                if (d2 < best_d2) best_d2 = d2;
            }
        } else {
            int sp = 0;
            stk[sp].nid = 0;
            stk[sp].d2 = point_box_d2(qx, qy, qz, nodes[0].bmin, nodes[0].bmax);
            sp++;
            while (sp > 0) {
                StackItem cur = stk[--sp];
                if (cur.d2 >= best_d2) continue;
                const KDNode *nd = &nodes[cur.nid];
                if (nd->axis < 0) {
                    int32_t k = nd->start, end = nd->end;
                    if (end - k >= 4) {
                        __m256d v_min = _mm256_set1_pd(best_d2);
                        for (; k + 4 <= end; k += 4) {
                            __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(rx + k), v_qx);
                            __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(ry + k), v_qy);
                            __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(rz + k), v_qz);
                            __m256d d2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)), _mm256_mul_pd(dz, dz));
                            v_min = _mm256_min_pd(v_min, d2);
                        }
                        double tmp[4];
                        _mm256_storeu_pd(tmp, v_min);
                        for (int i = 0; i < 4; i++) if (tmp[i] < best_d2) best_d2 = tmp[i];
                    }
                    for (; k < end; k++) {
                        double dx = rx[k] - qx, dy = ry[k] - qy, dz = rz[k] - qz;
                        double d2 = (dx * dx + dy * dy) + dz * dz;
                        if (d2 < best_d2) best_d2 = d2;
                    }
                    continue;
                }
                int32_t lid = nd->left, rid = nd->right;
                double ld2 = point_box_d2(qx, qy, qz, nodes[lid].bmin, nodes[lid].bmax);
                double rd2 = point_box_d2(qx, qy, qz, nodes[rid].bmin, nodes[rid].bmax);
                if (ld2 <= rd2) {
                    if (rd2 < best_d2 && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                    if (ld2 < best_d2 && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                } else {
                    if (ld2 < best_d2 && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                    if (rd2 < best_d2 && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                }
            }
        }
        out[q] = sqrt(best_d2);
    }
}
#endif

static PyObject *py_nn_min_dists_3d(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "nn_min_dists_3d expects (ref_pts, query_pts)");
        return NULL;
    }

    Py_buffer ref_b = {0}, q_b = {0};
    int got_r = 0, got_q = 0;
    PyObject *ret_obj = NULL;

    if (PyObject_GetBuffer(args[0], &ref_b, PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_r = 1;
    if (PyObject_GetBuffer(args[1], &q_b, PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_q = 1;

    if (ref_b.ndim != 2 || q_b.ndim != 2 ||
        ref_b.itemsize != (Py_ssize_t)sizeof(double) || q_b.itemsize != (Py_ssize_t)sizeof(double) ||
        ref_b.shape[1] != 3 || q_b.shape[1] != 3 ||
        ref_b.shape[0] <= 0 || ref_b.shape[0] > INT32_MAX) {
        ret_obj = Py_NotImplemented; Py_INCREF(ret_obj); goto cleanup;
    }

    Py_ssize_t m = ref_b.shape[0];
    Py_ssize_t n = q_b.shape[0];
    if (n == 0) {
        ret_obj = alloc_fast_1d(0, (Py_ssize_t)sizeof(double), g_np_float64, NULL);
        goto cleanup;
    }

    const double *ref = (const double *)ref_b.buf;
    const double *qry = (const double *)q_b.buf;

    ThreadScratch *sc = get_scratch();
    size_t min_leaf = (size_t)(KD_LEAF_SIZE >> 1);
    size_t max_nodes = (size_t)(2 * (((size_t)m + min_leaf - 1) / min_leaf) + 32);
    if (!sc ||
        ensure_p_idx(sc, (size_t)m) < 0 ||
        ensure_dbl_tmp(sc, (size_t)m * 3) < 0 ||
        ensure_kd_nodes(sc, max_nodes) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }

    double *rx = sc->dbl_tmp;
    double *ry = rx + m;
    double *rz = ry + m;
    int32_t *perm = sc->p_idx;
    KDNode *nodes = sc->kd_nodes;

    int non_finite = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < 3 * m; i++) {
        if (!isfinite(ref[i])) { non_finite = 1; break; }
    }
    for (Py_ssize_t i = 0; i < 3 * n && !non_finite; i++) {
        if (!isfinite(qry[i])) { non_finite = 1; break; }
    }
    Py_END_ALLOW_THREADS

    if (non_finite) {
        ret_obj = Py_NotImplemented; Py_INCREF(ret_obj); goto cleanup;
    }

    double *out = NULL;
    PyObject *out_arr = alloc_fast_1d(n, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out);
    if (!out_arr) goto cleanup;

    Py_BEGIN_ALLOW_THREADS
    int use_tree = (m > 32);
    if (!use_tree) {
        for (Py_ssize_t i = 0; i < m; i++) {
            rx[i] = ref[3 * i + 0];
            ry[i] = ref[3 * i + 1];
            rz[i] = ref[3 * i + 2];
        }
    } else {
        for (Py_ssize_t i = 0; i < m; i++) perm[i] = (int32_t)i;
        int32_t next_id = 0;
        kd_build_recursive(ref, perm, nodes, &next_id, 0, (int32_t)m);
        for (Py_ssize_t i = 0; i < m; i++) {
            Py_ssize_t src = (Py_ssize_t)perm[i];
            rx[i] = ref[3 * src + 0];
            ry[i] = ref[3 * src + 1];
            rz[i] = ref[3 * src + 2];
        }
    }
#if ES_HAVE_X86_RUNTIME_SIMD
    if (g_simd_level >= 2) {
        nn_min_dists_query_avx512(rx, ry, rz, nodes, use_tree, (int32_t)m, qry, n, out);
    } else if (g_simd_level == 1) {
        nn_min_dists_query_avx2(rx, ry, rz, nodes, use_tree, (int32_t)m, qry, n, out);
    } else
#endif
    {
        if (!use_tree) {
            int32_t m32 = (int32_t)m;
            for (Py_ssize_t q = 0; q < n; q++) {
                double best_d2 = leaf_min_d2_scalar(rx, ry, rz, 0, m32, qry[3 * q + 0], qry[3 * q + 1], qry[3 * q + 2], HUGE_VAL);
                out[q] = sqrt(best_d2);
            }
        } else {
            typedef struct { int32_t nid; double d2; } StackItem;
            StackItem stk[64];
            for (Py_ssize_t q = 0; q < n; q++) {
                double qx = qry[3 * q + 0], qy = qry[3 * q + 1], qz = qry[3 * q + 2];
                double best_d2 = HUGE_VAL;
                int sp = 0;
                stk[sp].nid = 0;
                stk[sp].d2 = point_box_d2(qx, qy, qz, nodes[0].bmin, nodes[0].bmax);
                sp++;
                while (sp > 0) {
                    StackItem cur = stk[--sp];
                    if (cur.d2 >= best_d2) continue;
                    const KDNode *nd = &nodes[cur.nid];
                    if (nd->axis < 0) {
                        best_d2 = leaf_min_d2_scalar(rx, ry, rz, nd->start, nd->end, qx, qy, qz, best_d2);
                        continue;
                    }
                    int32_t lid = nd->left, rid = nd->right;
                    double ld2 = point_box_d2(qx, qy, qz, nodes[lid].bmin, nodes[lid].bmax);
                    double rd2 = point_box_d2(qx, qy, qz, nodes[rid].bmin, nodes[rid].bmax);
                    if (ld2 <= rd2) {
                        if (rd2 < best_d2 && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                        if (ld2 < best_d2 && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                    } else {
                        if (ld2 < best_d2 && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                        if (rd2 < best_d2 && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                    }
                }
                out[q] = sqrt(best_d2);
            }
        }
    }
    Py_END_ALLOW_THREADS

    ret_obj = out_arr;

cleanup:
    if (got_r) PyBuffer_Release(&ref_b);
    if (got_q) PyBuffer_Release(&q_b);
    return ret_obj;
}

/* ============================================================================
 * 16. Measured & Persistent Planar Surface Meshing / Aggregation Kernels
 * ============================================================================ */

static inline uint64_t hash_u64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline int check_and_eval_tri(
    const double *p0, const double *p1, const double *p2,
    int has_normal, double nx, double ny, double nz,
    double max_edge_m, double max_normal_step_m,
    double *out_nx, double *out_ny, double *out_nz,
    double *out_area2, double *out_max_edge
) {
    double e0x = p0[0] - p2[0], e0y = p0[1] - p2[1], e0z = p0[2] - p2[2];
    double e1x = p1[0] - p0[0], e1y = p1[1] - p0[1], e1z = p1[2] - p0[2];
    double e2x = p2[0] - p1[0], e2y = p2[1] - p1[1], e2z = p2[2] - p1[2];

    double l0 = sqrt(e0x * e0x + e0y * e0y + e0z * e0z);
    double l1 = sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
    double l2 = sqrt(e2x * e2x + e2y * e2y + e2z * e2z);
    double ml = l0 > l1 ? l0 : l1;
    if (l2 > ml) ml = l2;
    if (!(ml <= max_edge_m)) return 0;

    double u2x = p2[0] - p0[0], u2y = p2[1] - p0[1], u2z = p2[2] - p0[2];
    double cx = e1y * u2z - e1z * u2y;
    double cy = e1z * u2x - e1x * u2z;
    double cz = e1x * u2y - e1y * u2x;
    double area2 = sqrt(cx * cx + cy * cy + cz * cz);
    if (!(area2 > 1e-12)) return 0;

    if (has_normal) {
        double d0 = fabs(e0x * nx + e0y * ny + e0z * nz);
        double d1 = fabs(e1x * nx + e1y * ny + e1z * nz);
        double d2 = fabs(e2x * nx + e2y * ny + e2z * nz);
        double mn = d0 > d1 ? d0 : d1;
        if (d2 > mn) mn = d2;
        if (!(mn <= max_normal_step_m)) return 0;
        double cn = cx * nx + cy * ny + cz * nz;
        if (!(cn > 1e-12)) return 0;
    }

    *out_nx = cx / area2;
    *out_ny = cy / area2;
    *out_nz = cz / area2;
    *out_area2 = area2;
    *out_max_edge = ml;
    return 1;
}

static PyObject *py_planar_mesh_finish_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pts_obj, *col_obj, *ids_obj, *grid_obj, *norm_obj;
    double max_edge_m, max_normal_step_m;
    Py_ssize_t max_vertices;
    if (!PyArg_ParseTuple(args, "OOOOOddn", &pts_obj, &col_obj, &ids_obj, &grid_obj, &norm_obj,
                          &max_edge_m, &max_normal_step_m, &max_vertices)) {
        return NULL;
    }

    Py_buffer pts_b, col_b, ids_b, grid_b, norm_b;
    int got_p = 0, got_c = 0, got_i = 0, got_g = 0, got_n = 0;
    int has_normal = (norm_obj != Py_None);
    PyObject *ret = NULL;

    int32_t *q0 = NULL, *q1 = NULL, *q2 = NULL, *q3 = NULL;
    int32_t *kf0 = NULL, *kf1 = NULL, *kf2 = NULL;
    double *kn = NULL;
    int32_t *remap = NULL;

    if (PyObject_GetBuffer(pts_obj, &pts_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_p = 1;
    if (PyObject_GetBuffer(col_obj, &col_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_c = 1;
    if (PyObject_GetBuffer(ids_obj, &ids_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_i = 1;
    if (PyObject_GetBuffer(grid_obj, &grid_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_g = 1;
    if (has_normal) {
        if (PyObject_GetBuffer(norm_obj, &norm_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        got_n = 1;
    }

    if (pts_b.ndim != 2 || pts_b.shape[1] != 3 || pts_b.itemsize != 8 ||
        col_b.ndim != 2 || col_b.shape[0] != pts_b.shape[0] || col_b.shape[1] != 3 || col_b.itemsize != 1 ||
        ids_b.ndim != 1 || ids_b.itemsize != 8 ||
        grid_b.ndim != 2 || grid_b.itemsize != 4 ||
        (has_normal && (norm_b.ndim != 1 || norm_b.shape[0] != 3 || norm_b.itemsize != 8))) {
        PyErr_SetString(PyExc_ValueError, "invalid buffer shapes or dtypes for planar_mesh_finish_fast");
        goto cleanup;
    }

    Py_ssize_t n_pts = pts_b.shape[0];
    Py_ssize_t m_ids = ids_b.shape[0];
    Py_ssize_t gh = grid_b.shape[0];
    Py_ssize_t gw = grid_b.shape[1];
    const double *pts = (const double *)pts_b.buf;
    const uint8_t *cols = (const uint8_t *)col_b.buf;
    const int64_t *ids = (const int64_t *)ids_b.buf;
    const int32_t *grid = (const int32_t *)grid_b.buf;
    double nx = 0.0, ny = 0.0, nz = 0.0;
    if (has_normal) {
        const double *nv = (const double *)norm_b.buf;
        nx = nv[0]; ny = nv[1]; nz = nv[2];
    }

    if (gh < 2 || gw < 2 || m_ids <= 0) {
        ret = Py_BuildValue("(iOOOOOdd)", 1, Py_None, Py_None, Py_None, Py_None, Py_None, 0.0, 0.0);
        goto cleanup;
    }

    Py_ssize_t max_q = (gh - 1) * (gw - 1);
    Py_ssize_t n_q = 0;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < gh - 1; r++) {
        const int32_t *row0 = grid + r * gw;
        const int32_t *row1 = row0 + gw;
        for (Py_ssize_t c = 0; c < gw - 1; c++) {
            if (row0[c] >= 0 && row1[c] >= 0 && row0[c + 1] >= 0 && row1[c + 1] >= 0) {
                n_q++;
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (n_q == 0) {
        ret = Py_BuildValue("(iOOOOOdd)", 1, Py_None, Py_None, Py_None, Py_None, Py_None, 0.0, 0.0);
        goto cleanup;
    }

    q0 = (int32_t *)malloc((size_t)n_q * sizeof(int32_t));
    q1 = (int32_t *)malloc((size_t)n_q * sizeof(int32_t));
    q2 = (int32_t *)malloc((size_t)n_q * sizeof(int32_t));
    q3 = (int32_t *)malloc((size_t)n_q * sizeof(int32_t));
    kf0 = (int32_t *)malloc((size_t)(2 * n_q) * sizeof(int32_t));
    kf1 = (int32_t *)malloc((size_t)(2 * n_q) * sizeof(int32_t));
    kf2 = (int32_t *)malloc((size_t)(2 * n_q) * sizeof(int32_t));
    kn = (double *)malloc((size_t)(6 * n_q) * sizeof(double));
    remap = (int32_t *)malloc((size_t)m_ids * sizeof(int32_t));
    if (!q0 || !q1 || !q2 || !q3 || !kf0 || !kf1 || !kf2 || !kn || !remap) {
        PyErr_NoMemory();
        goto cleanup;
    }
    (void)max_q;

    Py_ssize_t n_kept = 0;
    Py_ssize_t n_used = 0;
    double area2_sum = 0.0;
    double max_edge_seen = 0.0;
    int bad_idx = 0;

    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t qi = 0;
    for (Py_ssize_t r = 0; r < gh - 1; r++) {
        const int32_t *row0 = grid + r * gw;
        const int32_t *row1 = row0 + gw;
        for (Py_ssize_t c = 0; c < gw - 1; c++) {
            int32_t a = row0[c], b = row1[c], d = row0[c + 1], e = row1[c + 1];
            if (a >= 0 && b >= 0 && d >= 0 && e >= 0) {
                if (a >= m_ids || b >= m_ids || d >= m_ids || e >= m_ids) { bad_idx = 1; break; }
                q0[qi] = a; q1[qi] = b; q2[qi] = d; q3[qi] = e;
                qi++;
            }
        }
        if (bad_idx) break;
    }

    if (!bad_idx) {
        for (Py_ssize_t k = 0; k < m_ids; k++) remap[k] = -1;
        /* Pass 1: first triangle of each quad [0, 1, 2], matching _quads order */
        for (Py_ssize_t i = 0; i < n_q; i++) {
            int32_t i0 = q0[i], i1 = q1[i], i2 = q2[i];
            int64_t p0i = ids[i0], p1i = ids[i1], p2i = ids[i2];
            if (p0i < 0 || p0i >= n_pts || p1i < 0 || p1i >= n_pts || p2i < 0 || p2i >= n_pts) { bad_idx = 1; break; }
            double tnx, tny, tnz, a2, ml;
            if (check_and_eval_tri(pts + 3 * p0i, pts + 3 * p1i, pts + 3 * p2i,
                                   has_normal, nx, ny, nz, max_edge_m, max_normal_step_m,
                                   &tnx, &tny, &tnz, &a2, &ml)) {
                kf0[n_kept] = i0; kf1[n_kept] = i1; kf2[n_kept] = i2;
                kn[3 * n_kept + 0] = tnx; kn[3 * n_kept + 1] = tny; kn[3 * n_kept + 2] = tnz;
                n_kept++;
                area2_sum += a2;
                if (ml > max_edge_seen) max_edge_seen = ml;
                remap[i0] = 1; remap[i1] = 1; remap[i2] = 1;
            }
        }
    }
    if (!bad_idx) {
        /* Pass 2: second triangle of each quad [1, 3, 2], matching _quads order */
        for (Py_ssize_t i = 0; i < n_q; i++) {
            int32_t i0 = q1[i], i1 = q3[i], i2 = q2[i];
            int64_t p0i = ids[i0], p1i = ids[i1], p2i = ids[i2];
            if (p0i < 0 || p0i >= n_pts || p1i < 0 || p1i >= n_pts || p2i < 0 || p2i >= n_pts) { bad_idx = 1; break; }
            double tnx, tny, tnz, a2, ml;
            if (check_and_eval_tri(pts + 3 * p0i, pts + 3 * p1i, pts + 3 * p2i,
                                   has_normal, nx, ny, nz, max_edge_m, max_normal_step_m,
                                   &tnx, &tny, &tnz, &a2, &ml)) {
                kf0[n_kept] = i0; kf1[n_kept] = i1; kf2[n_kept] = i2;
                kn[3 * n_kept + 0] = tnx; kn[3 * n_kept + 1] = tny; kn[3 * n_kept + 2] = tnz;
                n_kept++;
                area2_sum += a2;
                if (ml > max_edge_seen) max_edge_seen = ml;
                remap[i0] = 1; remap[i1] = 1; remap[i2] = 1;
            }
        }
    }
    if (!bad_idx && n_kept > 0) {
        for (Py_ssize_t k = 0; k < m_ids; k++) {
            if (remap[k] > 0) remap[k] = (int32_t)(n_used++);
        }
    }
    Py_END_ALLOW_THREADS

    if (bad_idx) {
        PyErr_SetString(PyExc_IndexError, "grid or id index out of bounds in planar_mesh_finish_fast");
        goto cleanup;
    }
    if (n_kept == 0) {
        ret = Py_BuildValue("(iOOOOOdd)", 2, Py_None, Py_None, Py_None, Py_None, Py_None, 0.0, 0.0);
        goto cleanup;
    }
    if (n_used > max_vertices) {
        ret = Py_BuildValue("(iOOOOOdd)", 3, Py_None, Py_None, Py_None, Py_None, Py_None, 0.0, 0.0);
        goto cleanup;
    }

    void *v_ptr = NULL, *c_ptr = NULL, *t_ptr = NULL, *s_ptr = NULL, *fn_ptr = NULL;
    PyObject *v_arr = alloc_fast_2d(n_used, 3, 8, g_np_float64, &v_ptr);
    PyObject *c_arr = alloc_fast_2d(n_used, 3, 1, g_np_uint8, &c_ptr);
    PyObject *t_arr = alloc_fast_2d(n_kept, 3, 4, g_np_int32, &t_ptr);
    PyObject *s_arr = alloc_fast_1d(n_used, 8, g_np_int64, &s_ptr);
    PyObject *fn_arr = alloc_fast_2d(n_kept, 3, 8, g_np_float64, &fn_ptr);
    if (!v_arr || !c_arr || !t_arr || !s_arr || !fn_arr) {
        Py_XDECREF(v_arr); Py_XDECREF(c_arr); Py_XDECREF(t_arr); Py_XDECREF(s_arr); Py_XDECREF(fn_arr);
        goto cleanup;
    }

    double *vo = (double *)v_ptr;
    uint8_t *co = (uint8_t *)c_ptr;
    int32_t *to = (int32_t *)t_ptr;
    int64_t *so = (int64_t *)s_ptr;
    double *fno = (double *)fn_ptr;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t k = 0; k < m_ids; k++) {
        int32_t u = remap[k];
        if (u >= 0) {
            int64_t src = ids[k];
            so[u] = src;
            vo[3 * u + 0] = pts[3 * src + 0];
            vo[3 * u + 1] = pts[3 * src + 1];
            vo[3 * u + 2] = pts[3 * src + 2];
            co[3 * u + 0] = cols[3 * src + 0];
            co[3 * u + 1] = cols[3 * src + 1];
            co[3 * u + 2] = cols[3 * src + 2];
        }
    }
    for (Py_ssize_t f = 0; f < n_kept; f++) {
        to[3 * f + 0] = remap[kf0[f]];
        to[3 * f + 1] = remap[kf1[f]];
        to[3 * f + 2] = remap[kf2[f]];
    }
    memcpy(fno, kn, (size_t)(3 * n_kept) * sizeof(double));
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("(iNNNNNdd)", 0, v_arr, c_arr, t_arr, s_arr, fn_arr, area2_sum * 0.5, max_edge_seen);

cleanup:
    free(q0); free(q1); free(q2); free(q3);
    free(kf0); free(kf1); free(kf2); free(kn); free(remap);
    if (got_p) PyBuffer_Release(&pts_b);
    if (got_c) PyBuffer_Release(&col_b);
    if (got_i) PyBuffer_Release(&ids_b);
    if (got_g) PyBuffer_Release(&grid_b);
    if (got_n) PyBuffer_Release(&norm_b);
    return ret;
}

static PyObject *py_planar_tile_quads_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *grid_obj, *xyz_obj, *norm_obj;
    double max_edge_m, max_normal_step_m;
    if (!PyArg_ParseTuple(args, "OOOdd", &grid_obj, &xyz_obj, &norm_obj, &max_edge_m, &max_normal_step_m)) {
        return NULL;
    }
    Py_buffer grid_b, xyz_b, norm_b;
    int got_g = 0, got_x = 0, got_n = 0;
    PyObject *ret = NULL;
    int32_t *kf = NULL;
    double *kn = NULL;

    if (PyObject_GetBuffer(grid_obj, &grid_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_g = 1;
    if (PyObject_GetBuffer(xyz_obj, &xyz_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_x = 1;
    if (PyObject_GetBuffer(norm_obj, &norm_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    got_n = 1;

    if (grid_b.ndim != 2 || grid_b.itemsize != 4 ||
        xyz_b.ndim != 2 || xyz_b.shape[1] != 3 || xyz_b.itemsize != 8 ||
        norm_b.ndim != 1 || norm_b.shape[0] != 3 || norm_b.itemsize != 8) {
        PyErr_SetString(PyExc_ValueError, "invalid shapes/dtypes for planar_tile_quads_fast");
        goto cleanup;
    }

    Py_ssize_t gh = grid_b.shape[0], gw = grid_b.shape[1];
    Py_ssize_t n_slots = xyz_b.shape[0];
    const int32_t *grid = (const int32_t *)grid_b.buf;
    const double *xyz = (const double *)xyz_b.buf;
    const double *nv = (const double *)norm_b.buf;
    double nx = nv[0], ny = nv[1], nz = nv[2];

    Py_ssize_t max_tris = (gh >= 2 && gw >= 2) ? 2 * (gh - 1) * (gw - 1) : 0;
    if (max_tris == 0) {
        void *p1 = NULL, *p2 = NULL;
        PyObject *f_arr = alloc_fast_2d(0, 3, 4, g_np_int32, &p1);
        PyObject *n_arr = alloc_fast_2d(0, 3, 8, g_np_float64, &p2);
        if (!f_arr || !n_arr) { Py_XDECREF(f_arr); Py_XDECREF(n_arr); goto cleanup; }
        ret = Py_BuildValue("(NNdd)", f_arr, n_arr, 0.0, 0.0);
        goto cleanup;
    }

    kf = (int32_t *)malloc((size_t)(3 * max_tris) * sizeof(int32_t));
    kn = (double *)malloc((size_t)(3 * max_tris) * sizeof(double));
    if (!kf || !kn) { PyErr_NoMemory(); goto cleanup; }

    Py_ssize_t n_kept = 0;
    double area2_sum = 0.0, max_edge_seen = 0.0;
    int bad_idx = 0;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < gh - 1; r++) {
        const int32_t *row0 = grid + r * gw;
        const int32_t *row1 = row0 + gw;
        for (Py_ssize_t c = 0; c < gw - 1; c++) {
            int32_t a = row0[c], b = row1[c], d = row0[c + 1], e = row1[c + 1];
            if (a < 0 || b < 0 || d < 0 || e < 0) continue;
            if (a >= n_slots || b >= n_slots || d >= n_slots || e >= n_slots) { bad_idx = 1; break; }
            double tnx, tny, tnz, a2, ml;
            if (check_and_eval_tri(xyz + 3 * a, xyz + 3 * b, xyz + 3 * d,
                                   1, nx, ny, nz, max_edge_m, max_normal_step_m,
                                   &tnx, &tny, &tnz, &a2, &ml)) {
                kf[3 * n_kept + 0] = a; kf[3 * n_kept + 1] = b; kf[3 * n_kept + 2] = d;
                kn[3 * n_kept + 0] = tnx; kn[3 * n_kept + 1] = tny; kn[3 * n_kept + 2] = tnz;
                n_kept++; area2_sum += a2; if (ml > max_edge_seen) max_edge_seen = ml;
            }
            if (check_and_eval_tri(xyz + 3 * b, xyz + 3 * e, xyz + 3 * d,
                                   1, nx, ny, nz, max_edge_m, max_normal_step_m,
                                   &tnx, &tny, &tnz, &a2, &ml)) {
                kf[3 * n_kept + 0] = b; kf[3 * n_kept + 1] = e; kf[3 * n_kept + 2] = d;
                kn[3 * n_kept + 0] = tnx; kn[3 * n_kept + 1] = tny; kn[3 * n_kept + 2] = tnz;
                n_kept++; area2_sum += a2; if (ml > max_edge_seen) max_edge_seen = ml;
            }
        }
        if (bad_idx) break;
    }
    Py_END_ALLOW_THREADS

    if (bad_idx) {
        PyErr_SetString(PyExc_IndexError, "slot index out of bounds in planar_tile_quads_fast");
        goto cleanup;
    }
    void *fp = NULL, *np_ = NULL;
    PyObject *f_arr = alloc_fast_2d(n_kept, 3, 4, g_np_int32, &fp);
    PyObject *n_arr = alloc_fast_2d(n_kept, 3, 8, g_np_float64, &np_);
    if (!f_arr || !n_arr) { Py_XDECREF(f_arr); Py_XDECREF(n_arr); goto cleanup; }
    if (n_kept > 0) {
        memcpy(fp, kf, (size_t)(3 * n_kept) * sizeof(int32_t));
        memcpy(np_, kn, (size_t)(3 * n_kept) * sizeof(double));
    }
    ret = Py_BuildValue("(NNdd)", f_arr, n_arr, area2_sum * 0.5, max_edge_seen);

cleanup:
    free(kf); free(kn);
    if (got_g) PyBuffer_Release(&grid_b);
    if (got_x) PyBuffer_Release(&xyz_b);
    if (got_n) PyBuffer_Release(&norm_b);
    return ret;
}

static PyObject *py_planar_compact_triangles_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *tri_obj, *xyz_obj, *rgb_obj, *obs_obj, *ts_obj;
    Py_ssize_t max_vertices;
    if (!PyArg_ParseTuple(args, "OOOOOn", &tri_obj, &xyz_obj, &rgb_obj, &obs_obj, &ts_obj, &max_vertices)) {
        return NULL;
    }
    Py_buffer tri_b, xyz_b, rgb_b, obs_b, ts_b;
    int g_t = 0, g_x = 0, g_r = 0, g_o = 0, g_s = 0;
    int32_t *remap = NULL;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(tri_obj, &tri_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_t = 1;
    if (PyObject_GetBuffer(xyz_obj, &xyz_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_x = 1;
    if (PyObject_GetBuffer(rgb_obj, &rgb_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_r = 1;
    if (PyObject_GetBuffer(obs_obj, &obs_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_o = 1;
    if (PyObject_GetBuffer(ts_obj, &ts_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_s = 1;

    Py_ssize_t n_f = tri_b.shape[0];
    Py_ssize_t n_slots = xyz_b.shape[0];
    const int32_t *tris = (const int32_t *)tri_b.buf;
    const double *xyz = (const double *)xyz_b.buf;
    const uint8_t *rgb = (const uint8_t *)rgb_b.buf;
    const int32_t *obs = (const int32_t *)obs_b.buf;
    const int64_t *ts = (const int64_t *)ts_b.buf;

    remap = (int32_t *)malloc((size_t)(n_slots > 0 ? n_slots : 1) * sizeof(int32_t));
    if (!remap) { PyErr_NoMemory(); goto cleanup; }

    Py_ssize_t n_used = 0;
    int bad = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n_slots; i++) remap[i] = -1;
    for (Py_ssize_t i = 0; i < 3 * n_f; i++) {
        int32_t s = tris[i];
        if (s < 0 || s >= n_slots) { bad = 1; break; }
        remap[s] = 1;
    }
    if (!bad) {
        for (Py_ssize_t i = 0; i < n_slots; i++) {
            if (remap[i] > 0) remap[i] = (int32_t)(n_used++);
        }
    }
    Py_END_ALLOW_THREADS

    if (bad) { PyErr_SetString(PyExc_IndexError, "slot out of range"); goto cleanup; }
    if (n_used > max_vertices) {
        ret = Py_BuildValue("(iOOOOO)", 3, Py_None, Py_None, Py_None, Py_None, Py_None);
        goto cleanup;
    }

    void *vp = NULL, *cp = NULL, *tp = NULL, *op = NULL, *tsp = NULL;
    PyObject *v_arr = alloc_fast_2d(n_used, 3, 8, g_np_float64, &vp);
    PyObject *c_arr = alloc_fast_2d(n_used, 3, 1, g_np_uint8, &cp);
    PyObject *t_arr = alloc_fast_2d(n_f, 3, 4, g_np_int32, &tp);
    PyObject *o_arr = alloc_fast_1d(n_used, 4, g_np_int32, &op);
    PyObject *ts_arr = alloc_fast_1d(n_used, 8, g_np_int64, &tsp);
    if (!v_arr || !c_arr || !t_arr || !o_arr || !ts_arr) {
        Py_XDECREF(v_arr); Py_XDECREF(c_arr); Py_XDECREF(t_arr); Py_XDECREF(o_arr); Py_XDECREF(ts_arr);
        goto cleanup;
    }

    double *vo = (double *)vp;
    uint8_t *co = (uint8_t *)cp;
    int32_t *to = (int32_t *)tp;
    int32_t *oo = (int32_t *)op;
    int64_t *tso = (int64_t *)tsp;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t s = 0; s < n_slots; s++) {
        int32_t u = remap[s];
        if (u >= 0) {
            vo[3 * u + 0] = xyz[3 * s + 0];
            vo[3 * u + 1] = xyz[3 * s + 1];
            vo[3 * u + 2] = xyz[3 * s + 2];
            co[3 * u + 0] = rgb[3 * s + 0];
            co[3 * u + 1] = rgb[3 * s + 1];
            co[3 * u + 2] = rgb[3 * s + 2];
            oo[u] = obs[s];
            tso[u] = ts[s];
        }
    }
    for (Py_ssize_t i = 0; i < 3 * n_f; i++) {
        to[i] = remap[tris[i]];
    }
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("(iNNNNN)", 0, v_arr, c_arr, t_arr, o_arr, ts_arr);

cleanup:
    free(remap);
    if (g_t) PyBuffer_Release(&tri_b);
    if (g_x) PyBuffer_Release(&xyz_b);
    if (g_r) PyBuffer_Release(&rgb_b);
    if (g_o) PyBuffer_Release(&obs_b);
    if (g_s) PyBuffer_Release(&ts_b);
    return ret;
}

typedef struct {
    int64_t key;
    double sw_geo;
    double sw_rgb;
    double sx, sy, sz;
    double sr, sg, sb;
    int32_t cnt;
} CellAggEntry;

static int cmp_cell_agg_entry(const void *a, const void *b) {
    int64_t ka = ((const CellAggEntry *)a)->key;
    int64_t kb = ((const CellAggEntry *)b)->key;
    return (ka > kb) - (ka < kb);
}

static PyObject *py_planar_cell_aggregate_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pts_obj, *rgb_obj, *wg_obj, *wr_obj, *keys_obj;
    double obs_weight;
    Py_ssize_t max_cells;
    if (!PyArg_ParseTuple(args, "OOOOOdn", &pts_obj, &rgb_obj, &wg_obj, &wr_obj, &keys_obj, &obs_weight, &max_cells)) {
        return NULL;
    }
    Py_buffer pts_b, rgb_b, wg_b, wr_b, keys_b;
    int g_p = 0, g_r = 0, g_wg = 0, g_wr = 0, g_k = 0;
    int32_t *ht = NULL;
    CellAggEntry *entries = NULL;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(pts_obj, &pts_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_p = 1;
    if (PyObject_GetBuffer(rgb_obj, &rgb_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_r = 1;
    if (PyObject_GetBuffer(wg_obj, &wg_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_wg = 1;
    if (PyObject_GetBuffer(wr_obj, &wr_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_wr = 1;
    if (PyObject_GetBuffer(keys_obj, &keys_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_k = 1;

    Py_ssize_t n = keys_b.shape[0];
    if (n == 0) {
        void *p1 = NULL, *p2 = NULL, *p3 = NULL, *p4 = NULL;
        PyObject *k_a = alloc_fast_1d(0, 8, g_np_int64, &p1);
        PyObject *p_a = alloc_fast_2d(0, 3, 8, g_np_float64, &p2);
        PyObject *c_a = alloc_fast_2d(0, 3, 4, g_np_float32, &p3);
        PyObject *w_a = alloc_fast_1d(0, 8, g_np_float64, &p4);
        if (!k_a || !p_a || !c_a || !w_a) { Py_XDECREF(k_a); Py_XDECREF(p_a); Py_XDECREF(c_a); Py_XDECREF(w_a); goto cleanup; }
        ret = Py_BuildValue("(NNNN)", k_a, p_a, c_a, w_a);
        goto cleanup;
    }

    size_t hcap = next_pow2((size_t)n * 2ULL);
    size_t mask = hcap - 1ULL;
    ht = (int32_t *)malloc(hcap * sizeof(int32_t));
    entries = (CellAggEntry *)malloc((size_t)n * sizeof(CellAggEntry));
    if (!ht || !entries) { PyErr_NoMemory(); goto cleanup; }

    const double *pts = (const double *)pts_b.buf;
    const uint8_t *rgb = (const uint8_t *)rgb_b.buf;
    const double *wg = (const double *)wg_b.buf;
    const double *wr = (const double *)wr_b.buf;
    const int64_t *keys = (const int64_t *)keys_b.buf;

    Py_ssize_t n_uniq = 0;
    Py_BEGIN_ALLOW_THREADS
    for (size_t i = 0; i < hcap; i++) ht[i] = -1;
    for (Py_ssize_t i = 0; i < n; i++) {
        int64_t k = keys[i];
        double gwi = wg[i], rwi = wr[i];
        size_t pos = (size_t)hash_u64((uint64_t)k) & mask;
        int32_t idx;
        while (1) {
            idx = ht[pos];
            if (idx < 0) {
                idx = (int32_t)(n_uniq++);
                ht[pos] = idx;
                entries[idx].key = k;
                entries[idx].sw_geo = gwi;
                entries[idx].sw_rgb = rwi;
                entries[idx].sx = gwi * pts[3 * i + 0];
                entries[idx].sy = gwi * pts[3 * i + 1];
                entries[idx].sz = gwi * pts[3 * i + 2];
                entries[idx].sr = rwi * (double)rgb[3 * i + 0];
                entries[idx].sg = rwi * (double)rgb[3 * i + 1];
                entries[idx].sb = rwi * (double)rgb[3 * i + 2];
                entries[idx].cnt = 1;
                break;
            }
            if (entries[idx].key == k) {
                entries[idx].sw_geo += gwi;
                entries[idx].sw_rgb += rwi;
                entries[idx].sx += gwi * pts[3 * i + 0];
                entries[idx].sy += gwi * pts[3 * i + 1];
                entries[idx].sz += gwi * pts[3 * i + 2];
                entries[idx].sr += rwi * (double)rgb[3 * i + 0];
                entries[idx].sg += rwi * (double)rgb[3 * i + 1];
                entries[idx].sb += rwi * (double)rgb[3 * i + 2];
                entries[idx].cnt += 1;
                break;
            }
            pos = (pos + 1ULL) & mask;
        }
    }
    qsort(entries, (size_t)n_uniq, sizeof(CellAggEntry), cmp_cell_agg_entry);
    Py_END_ALLOW_THREADS

    Py_ssize_t n_out = n_uniq > max_cells ? max_cells : n_uniq;
    void *kp = NULL, *pp = NULL, *cp = NULL, *wp = NULL;
    PyObject *k_arr = alloc_fast_1d(n_out, 8, g_np_int64, &kp);
    PyObject *p_arr = alloc_fast_2d(n_out, 3, 8, g_np_float64, &pp);
    PyObject *c_arr = alloc_fast_2d(n_out, 3, 4, g_np_float32, &cp);
    PyObject *w_arr = alloc_fast_1d(n_out, 8, g_np_float64, &wp);
    if (!k_arr || !p_arr || !c_arr || !w_arr) {
        Py_XDECREF(k_arr); Py_XDECREF(p_arr); Py_XDECREF(c_arr); Py_XDECREF(w_arr);
        goto cleanup;
    }

    int64_t *ko = (int64_t *)kp;
    double *po = (double *)pp;
    float *co = (float *)cp;
    double *wo = (double *)wp;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n_out; i++) {
        const CellAggEntry *e = &entries[i];
        ko[i] = e->key;
        double inv_g = 1.0 / e->sw_geo;
        double inv_r = 1.0 / e->sw_rgb;
        po[3 * i + 0] = e->sx * inv_g;
        po[3 * i + 1] = e->sy * inv_g;
        po[3 * i + 2] = e->sz * inv_g;
        co[3 * i + 0] = (float)(e->sr * inv_r);
        co[3 * i + 1] = (float)(e->sg * inv_r);
        co[3 * i + 2] = (float)(e->sb * inv_r);
        wo[i] = obs_weight * (e->sw_geo / (double)e->cnt);
    }
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("(NNNN)", k_arr, p_arr, c_arr, w_arr);

cleanup:
    free(ht); free(entries);
    if (g_p) PyBuffer_Release(&pts_b);
    if (g_r) PyBuffer_Release(&rgb_b);
    if (g_wg) PyBuffer_Release(&wg_b);
    if (g_wr) PyBuffer_Release(&wr_b);
    if (g_k) PyBuffer_Release(&keys_b);
    return ret;
}

static const int64_t HT_EMPTY = (-9223372036854775807LL - 1LL);

static PyObject *py_planar_ht_lookup_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *hk_obj, *hs_obj, *qk_obj;
    if (!PyArg_ParseTuple(args, "OOO", &hk_obj, &hs_obj, &qk_obj)) return NULL;
    Py_buffer hk_b, hs_b, qk_b;
    int g_hk = 0, g_hs = 0, g_qk = 0;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(hk_obj, &hk_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_hk = 1;
    if (PyObject_GetBuffer(hs_obj, &hs_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_hs = 1;
    if (PyObject_GetBuffer(qk_obj, &qk_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_qk = 1;

    size_t hcap = (size_t)hk_b.shape[0];
    size_t mask = hcap - 1ULL;
    Py_ssize_t nq = qk_b.shape[0];
    const int64_t *hk = (const int64_t *)hk_b.buf;
    const int32_t *hs = (const int32_t *)hs_b.buf;
    const int64_t *qk = (const int64_t *)qk_b.buf;

    void *op = NULL;
    PyObject *out_arr = alloc_fast_1d(nq, 4, g_np_int32, &op);
    if (!out_arr) goto cleanup;
    int32_t *out = (int32_t *)op;

    Py_ssize_t missing = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nq; i++) {
        int64_t k = qk[i];
        size_t pos = (size_t)hash_u64((uint64_t)k) & mask;
        int32_t found = -1;
        while (1) {
            int64_t tk = hk[pos];
            if (tk == k) { found = hs[pos]; break; }
            if (tk == HT_EMPTY) break;
            pos = (pos + 1ULL) & mask;
        }
        out[i] = found;
        if (found < 0) missing++;
    }
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("(Nn)", out_arr, missing);

cleanup:
    if (g_hk) PyBuffer_Release(&hk_b);
    if (g_hs) PyBuffer_Release(&hs_b);
    if (g_qk) PyBuffer_Release(&qk_b);
    return ret;
}

static PyObject *py_planar_ht_rebuild_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *active_obj, *keys_obj, *hk_obj, *hs_obj;
    Py_ssize_t count;
    if (!PyArg_ParseTuple(args, "OOnOO", &active_obj, &keys_obj, &count, &hk_obj, &hs_obj)) return NULL;
    Py_buffer a_b, k_b, hk_b, hs_b;
    int g_a = 0, g_k = 0, g_hk = 0, g_hs = 0;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(active_obj, &a_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_a = 1;
    if (PyObject_GetBuffer(keys_obj, &k_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_k = 1;
    if (PyObject_GetBuffer(hk_obj, &hk_b, PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_hk = 1;
    if (PyObject_GetBuffer(hs_obj, &hs_b, PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_hs = 1;

    size_t hcap = (size_t)hk_b.shape[0];
    size_t mask = hcap - 1ULL;
    const uint8_t *act = (const uint8_t *)a_b.buf;
    const int64_t *keys = (const int64_t *)k_b.buf;
    int64_t *hk = (int64_t *)hk_b.buf;
    int32_t *hs = (int32_t *)hs_b.buf;

    Py_ssize_t inserted = 0;
    Py_BEGIN_ALLOW_THREADS
    for (size_t i = 0; i < hcap; i++) { hk[i] = HT_EMPTY; hs[i] = -1; }
    for (Py_ssize_t s = 0; s < count; s++) {
        if (!act[s]) continue;
        int64_t k = keys[s];
        size_t pos = (size_t)hash_u64((uint64_t)k) & mask;
        while (hk[pos] != HT_EMPTY) pos = (pos + 1ULL) & mask;
        hk[pos] = k;
        hs[pos] = (int32_t)s;
        inserted++;
    }
    Py_END_ALLOW_THREADS

    ret = PyLong_FromSsize_t(inserted);

cleanup:
    if (g_a) PyBuffer_Release(&a_b);
    if (g_k) PyBuffer_Release(&k_b);
    if (g_hk) PyBuffer_Release(&hk_b);
    if (g_hs) PyBuffer_Release(&hs_b);
    return ret;
}

static PyObject *py_planar_window_accumulate_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *slots_o, *ww_o, *wxyz_o, *wrgb_o, *wcnt_o, *lts_o, *lseq_o, *conf_o;
    PyObject *cpts_o, *crgb_o, *cw_o;
    PyObject *bw_o, *bxyz_o, *brgb_o, *bcnt_o, *fxyz_o, *frgb_o, *tobs_o;
    int64_t timestamp_ns, update_seq;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOOLL",
                          &slots_o, &ww_o, &wxyz_o, &wrgb_o, &wcnt_o, &lts_o, &lseq_o, &conf_o,
                          &cpts_o, &crgb_o, &cw_o,
                          &bw_o, &bxyz_o, &brgb_o, &bcnt_o, &fxyz_o, &frgb_o, &tobs_o,
                          &timestamp_ns, &update_seq)) {
        return NULL;
    }
    Py_buffer b[18];
    PyObject *objs[18] = {
        slots_o, ww_o, wxyz_o, wrgb_o, wcnt_o, lts_o, lseq_o, conf_o,
        cpts_o, crgb_o, cw_o,
        bw_o, bxyz_o, brgb_o, bcnt_o, fxyz_o, frgb_o, tobs_o
    };
    int flags[18] = {
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
    };
    int got = 0;
    for (int i = 0; i < 18; i++) {
        if (PyObject_GetBuffer(objs[i], &b[i], flags[i]) < 0) goto cleanup;
        got++;
    }

    Py_ssize_t n = b[0].shape[0];
    Py_ssize_t cap = b[1].shape[0];
    const int32_t *slots = (const int32_t *)b[0].buf;
    double *ww = (double *)b[1].buf;
    double *wxyz = (double *)b[2].buf;
    float *wrgb = (float *)b[3].buf;
    int32_t *wcnt = (int32_t *)b[4].buf;
    int64_t *lts = (int64_t *)b[5].buf;
    int64_t *lseq = (int64_t *)b[6].buf;
    int32_t *conf = (int32_t *)b[7].buf;
    const double *cpts = (const double *)b[8].buf;
    const float *crgb = (const float *)b[9].buf;
    const double *cw = (const double *)b[10].buf;
    const double *bw = (const double *)b[11].buf;
    const double *bxyz = (const double *)b[12].buf;
    const float *brgb = (const float *)b[13].buf;
    const int32_t *bcnt = (const int32_t *)b[14].buf;
    double *fxyz = (double *)b[15].buf;
    uint8_t *frgb = (uint8_t *)b[16].buf;
    int32_t *tobs = (int32_t *)b[17].buf;

    int bad = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        int32_t s = slots[i];
        if (s < 0 || s >= cap) { bad = 1; break; }
        double w = cw[i];
        float wf = (float)w;
        double nw = ww[s] + w;
        ww[s] = nw;
        double wx = wxyz[3 * s + 0] + w * cpts[3 * i + 0];
        double wy = wxyz[3 * s + 1] + w * cpts[3 * i + 1];
        double wz = wxyz[3 * s + 2] + w * cpts[3 * i + 2];
        wxyz[3 * s + 0] = wx; wxyz[3 * s + 1] = wy; wxyz[3 * s + 2] = wz;
        float wr = wrgb[3 * s + 0] + wf * crgb[3 * i + 0];
        float wg = wrgb[3 * s + 1] + wf * crgb[3 * i + 1];
        float wb = wrgb[3 * s + 2] + wf * crgb[3 * i + 2];
        wrgb[3 * s + 0] = wr; wrgb[3 * s + 1] = wg; wrgb[3 * s + 2] = wb;
        int32_t nwc = wcnt[s] + 1;
        wcnt[s] = nwc;
        if (timestamp_ns > lts[s]) lts[s] = timestamp_ns;
        lseq[s] = update_seq;
        conf[s] = 0;

        /* Fused refresh of _total_obs, _fused_xyz, _fused_rgb */
        tobs[s] = bcnt[s] + nwc;
        double tw = bw[s] + nw;
        if (tw > 0.0) {
            double inv = 1.0 / tw;
            fxyz[3 * s + 0] = (bxyz[3 * s + 0] + wx) * inv;
            fxyz[3 * s + 1] = (bxyz[3 * s + 1] + wy) * inv;
            fxyz[3 * s + 2] = (bxyz[3 * s + 2] + wz) * inv;
            double r0 = rint(((double)brgb[3 * s + 0] + (double)wr) * inv);
            double r1 = rint(((double)brgb[3 * s + 1] + (double)wg) * inv);
            double r2 = rint(((double)brgb[3 * s + 2] + (double)wb) * inv);
            frgb[3 * s + 0] = (uint8_t)(r0 < 0.0 ? 0.0 : (r0 > 255.0 ? 255.0 : r0));
            frgb[3 * s + 1] = (uint8_t)(r1 < 0.0 ? 0.0 : (r1 > 255.0 ? 255.0 : r1));
            frgb[3 * s + 2] = (uint8_t)(r2 < 0.0 ? 0.0 : (r2 > 255.0 ? 255.0 : r2));
        }
    }
    Py_END_ALLOW_THREADS

    for (int i = 0; i < got; i++) PyBuffer_Release(&b[i]);
    if (bad) {
        PyErr_SetString(PyExc_IndexError, "slot out of range in planar_window_accumulate_fast");
        return NULL;
    }
    Py_RETURN_NONE;

cleanup:
    for (int i = 0; i < got; i++) PyBuffer_Release(&b[i]);
    return NULL;
}

static PyObject *py_planar_compact_obs_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *slots_o, *act_o, *cpts_o, *crgb_o, *cw_o;
    PyObject *ww_o, *wxyz_o, *wrgb_o, *wcnt_o;
    PyObject *bw_o, *bxyz_o, *brgb_o, *bcnt_o, *blts_o;
    PyObject *fxyz_o, *frgb_o, *tobs_o;
    double max_w;
    int64_t obs_ts;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOdL",
                          &slots_o, &act_o, &cpts_o, &crgb_o, &cw_o,
                          &ww_o, &wxyz_o, &wrgb_o, &wcnt_o,
                          &bw_o, &bxyz_o, &brgb_o, &bcnt_o, &blts_o,
                          &fxyz_o, &frgb_o, &tobs_o,
                          &max_w, &obs_ts)) {
        return NULL;
    }
    Py_buffer b[17];
    PyObject *objs[17] = {
        slots_o, act_o, cpts_o, crgb_o, cw_o,
        ww_o, wxyz_o, wrgb_o, wcnt_o,
        bw_o, bxyz_o, brgb_o, bcnt_o, blts_o,
        fxyz_o, frgb_o, tobs_o
    };
    int flags[17] = {
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
    };
    int got = 0;
    int32_t *capped = NULL;
    PyObject *ret = NULL;
    for (int i = 0; i < 17; i++) {
        if (PyObject_GetBuffer(objs[i], &b[i], flags[i]) < 0) goto cleanup;
        got++;
    }

    Py_ssize_t n = b[0].shape[0];
    Py_ssize_t cap = b[1].shape[0];
    const int32_t *slots = (const int32_t *)b[0].buf;
    const uint8_t *act = (const uint8_t *)b[1].buf;
    const double *cpts = (const double *)b[2].buf;
    const float *crgb = (const float *)b[3].buf;
    const double *cw = (const double *)b[4].buf;
    double *ww = (double *)b[5].buf;
    double *wxyz = (double *)b[6].buf;
    float *wrgb = (float *)b[7].buf;
    int32_t *wcnt = (int32_t *)b[8].buf;
    double *bw = (double *)b[9].buf;
    double *bxyz = (double *)b[10].buf;
    float *brgb = (float *)b[11].buf;
    int32_t *bcnt = (int32_t *)b[12].buf;
    int64_t *blts = (int64_t *)b[13].buf;
    double *fxyz = (double *)b[14].buf;
    uint8_t *frgb = (uint8_t *)b[15].buf;
    int32_t *tobs = (int32_t *)b[16].buf;

    capped = (int32_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    if (!capped) { PyErr_NoMemory(); goto cleanup; }

    Py_ssize_t n_capped = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        int32_t s = slots[i];
        if (s < 0 || s >= cap || !act[s]) continue;
        double w = cw[i];
        float wf = (float)w;
        int32_t nwc = wcnt[s] - 1;
        if (nwc <= 0) {
            wcnt[s] = 0;
            ww[s] = 0.0;
            wxyz[3 * s + 0] = 0.0; wxyz[3 * s + 1] = 0.0; wxyz[3 * s + 2] = 0.0;
            wrgb[3 * s + 0] = 0.0f; wrgb[3 * s + 1] = 0.0f; wrgb[3 * s + 2] = 0.0f;
        } else {
            wcnt[s] = nwc;
            double nw = ww[s] - w;
            ww[s] = nw > 0.0 ? nw : 0.0;
            wxyz[3 * s + 0] -= w * cpts[3 * i + 0];
            wxyz[3 * s + 1] -= w * cpts[3 * i + 1];
            wxyz[3 * s + 2] -= w * cpts[3 * i + 2];
            wrgb[3 * s + 0] -= wf * crgb[3 * i + 0];
            wrgb[3 * s + 1] -= wf * crgb[3 * i + 1];
            wrgb[3 * s + 2] -= wf * crgb[3 * i + 2];
        }

        double nbw = bw[s] + w;
        double nbx = bxyz[3 * s + 0] + w * cpts[3 * i + 0];
        double nby = bxyz[3 * s + 1] + w * cpts[3 * i + 1];
        double nbz = bxyz[3 * s + 2] + w * cpts[3 * i + 2];
        double nbr = (double)brgb[3 * s + 0] + w * (double)crgb[3 * i + 0];
        double nbg = (double)brgb[3 * s + 1] + w * (double)crgb[3 * i + 1];
        double nbb = (double)brgb[3 * s + 2] + w * (double)crgb[3 * i + 2];
        int was_capped = 0;
        if (nbw > max_w) {
            double sc = max_w / nbw;
            nbw = max_w;
            nbx *= sc; nby *= sc; nbz *= sc;
            nbr *= sc; nbg *= sc; nbb *= sc;
            was_capped = 1;
        }
        bw[s] = nbw;
        bxyz[3 * s + 0] = nbx; bxyz[3 * s + 1] = nby; bxyz[3 * s + 2] = nbz;
        brgb[3 * s + 0] = (float)nbr; brgb[3 * s + 1] = (float)nbg; brgb[3 * s + 2] = (float)nbb;
        bcnt[s] += 1;
        if (obs_ts > blts[s]) blts[s] = obs_ts;
        tobs[s] = bcnt[s] + wcnt[s];

        if (was_capped) {
            double tw = nbw + ww[s];
            if (tw > 0.0) {
                double inv = 1.0 / tw;
                fxyz[3 * s + 0] = (nbx + wxyz[3 * s + 0]) * inv;
                fxyz[3 * s + 1] = (nby + wxyz[3 * s + 1]) * inv;
                fxyz[3 * s + 2] = (nbz + wxyz[3 * s + 2]) * inv;
                double r0 = rint((nbr + (double)wrgb[3 * s + 0]) * inv);
                double r1 = rint((nbg + (double)wrgb[3 * s + 1]) * inv);
                double r2 = rint((nbb + (double)wrgb[3 * s + 2]) * inv);
                frgb[3 * s + 0] = (uint8_t)(r0 < 0.0 ? 0.0 : (r0 > 255.0 ? 255.0 : r0));
                frgb[3 * s + 1] = (uint8_t)(r1 < 0.0 ? 0.0 : (r1 > 255.0 ? 255.0 : r1));
                frgb[3 * s + 2] = (uint8_t)(r2 < 0.0 ? 0.0 : (r2 > 255.0 ? 255.0 : r2));
            }
            capped[n_capped++] = s;
        }
    }
    Py_END_ALLOW_THREADS

    void *cp = NULL;
    PyObject *c_arr = alloc_fast_1d(n_capped, 4, g_np_int32, &cp);
    if (!c_arr) goto cleanup;
    if (n_capped > 0) memcpy(cp, capped, (size_t)n_capped * sizeof(int32_t));
    ret = c_arr;

cleanup:
    free(capped);
    for (int i = 0; i < got; i++) PyBuffer_Release(&b[i]);
    return ret;
}

/* ============================================================================
 * 17. v2.5.0 Kernels: kNN RGB Transfer, Fused Planar Project/Aggregate,
 *     Free-Space Carving, & Deterministic RANSAC Plane Consensus
 * ============================================================================ */

static inline void knn_leaf_update_scalar(
    const double *rx, const double *ry, const double *rz,
    const int32_t *perm,
    int32_t start, int32_t end,
    double qx, double qy, double qz,
    int k, double *best_d2, int32_t *best_idx)
{
    double worst = best_d2[k - 1];
    for (int32_t i = start; i < end; i++) {
        double dx = rx[i] - qx;
        double dy = ry[i] - qy;
        double dz = rz[i] - qz;
        double d2 = (dx * dx + dy * dy) + dz * dz;
        int32_t orig = perm ? perm[i] : i;
        if (d2 < worst || (d2 == worst && orig < best_idx[k - 1])) {
            int pos = k - 1;
            while (pos > 0 && (d2 < best_d2[pos - 1] ||
                               (d2 == best_d2[pos - 1] && orig < best_idx[pos - 1]))) {
                best_d2[pos] = best_d2[pos - 1];
                best_idx[pos] = best_idx[pos - 1];
                pos--;
            }
            best_d2[pos] = d2;
            best_idx[pos] = orig;
            worst = best_d2[k - 1];
        }
    }
}

#if ES_HAVE_X86_RUNTIME_SIMD
__attribute__((target("avx512f,avx512dq,avx512vl")))
static void knn_rgb_query_avx512(
    const double *rx, const double *ry, const double *rz,
    const int32_t *perm, const KDNode *nodes, int use_tree, int32_t m32,
    const double *qry, Py_ssize_t n, int k,
    int rgb_is_u8, const uint8_t *rgb_u8, const double *rgb_f64, uint8_t *out)
{
    typedef struct { int32_t nid; double d2; } StackItem;
    StackItem stk[64];
    double best_d2[32];
    int32_t best_idx[32];

    for (Py_ssize_t q = 0; q < n; q++) {
        double qx = qry[3 * q + 0], qy = qry[3 * q + 1], qz = qry[3 * q + 2];
        for (int j = 0; j < k; j++) { best_d2[j] = HUGE_VAL; best_idx[j] = INT32_MAX; }
        __m512d v_qx = _mm512_set1_pd(qx);
        __m512d v_qy = _mm512_set1_pd(qy);
        __m512d v_qz = _mm512_set1_pd(qz);
        if (!use_tree) {
            int32_t i = 0;
            double worst = best_d2[k - 1];
            __m512d v_worst = _mm512_set1_pd(worst);
            for (; i + 8 <= m32; i += 8) {
                __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(rx + i), v_qx);
                __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(ry + i), v_qy);
                __m512d dz = _mm512_sub_pd(_mm512_loadu_pd(rz + i), v_qz);
                __m512d d2 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)), _mm512_mul_pd(dz, dz));
                __mmask8 mask = _mm512_cmp_pd_mask(d2, v_worst, _CMP_LE_OQ);
                if (__builtin_expect(mask != 0, 0)) {
                    double tmp[8];
                    _mm512_storeu_pd(tmp, d2);
                    unsigned um = (unsigned)mask;
                    while (um) {
                        int lane = __builtin_ctz(um);
                        um &= (um - 1u);
                        int32_t orig = i + lane;
                        double dv = tmp[lane];
                        if (dv < worst || (dv == worst && orig < best_idx[k - 1])) {
                            int pos = k - 1;
                            while (pos > 0 && (dv < best_d2[pos - 1] ||
                                               (dv == best_d2[pos - 1] && orig < best_idx[pos - 1]))) {
                                best_d2[pos] = best_d2[pos - 1];
                                best_idx[pos] = best_idx[pos - 1];
                                pos--;
                            }
                            best_d2[pos] = dv;
                            best_idx[pos] = orig;
                            worst = best_d2[k - 1];
                            v_worst = _mm512_set1_pd(worst);
                        }
                    }
                }
            }
            for (; i < m32; i++) {
                double dx = rx[i] - qx, dy = ry[i] - qy, dz = rz[i] - qz;
                double dv = (dx * dx + dy * dy) + dz * dz;
                if (dv < worst || (dv == worst && i < best_idx[k - 1])) {
                    int pos = k - 1;
                    while (pos > 0 && (dv < best_d2[pos - 1] ||
                                       (dv == best_d2[pos - 1] && i < best_idx[pos - 1]))) {
                        best_d2[pos] = best_d2[pos - 1];
                        best_idx[pos] = best_idx[pos - 1];
                        pos--;
                    }
                    best_d2[pos] = dv;
                    best_idx[pos] = i;
                    worst = best_d2[k - 1];
                }
            }
        } else {
            int sp = 0;
            stk[sp].nid = 0;
            stk[sp].d2 = point_box_d2(qx, qy, qz, nodes[0].bmin, nodes[0].bmax);
            sp++;
            while (sp > 0) {
                StackItem cur = stk[--sp];
                if (cur.d2 > best_d2[k - 1]) continue;
                const KDNode *nd = &nodes[cur.nid];
                if (nd->axis < 0) {
                    int32_t i = nd->start, end = nd->end;
                    double worst = best_d2[k - 1];
                    __m512d v_worst = _mm512_set1_pd(worst);
                    for (; i + 8 <= end; i += 8) {
                        __m512d dx = _mm512_sub_pd(_mm512_loadu_pd(rx + i), v_qx);
                        __m512d dy = _mm512_sub_pd(_mm512_loadu_pd(ry + i), v_qy);
                        __m512d dz = _mm512_sub_pd(_mm512_loadu_pd(rz + i), v_qz);
                        __m512d d2 = _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(dx, dx), _mm512_mul_pd(dy, dy)), _mm512_mul_pd(dz, dz));
                        __mmask8 mask = _mm512_cmp_pd_mask(d2, v_worst, _CMP_LE_OQ);
                        if (__builtin_expect(mask != 0, 0)) {
                            double tmp[8];
                            _mm512_storeu_pd(tmp, d2);
                            unsigned um = (unsigned)mask;
                            while (um) {
                                int lane = __builtin_ctz(um);
                                um &= (um - 1u);
                                int32_t orig = perm[i + lane];
                                double dv = tmp[lane];
                                if (dv < worst || (dv == worst && orig < best_idx[k - 1])) {
                                    int pos = k - 1;
                                    while (pos > 0 && (dv < best_d2[pos - 1] ||
                                                       (dv == best_d2[pos - 1] && orig < best_idx[pos - 1]))) {
                                        best_d2[pos] = best_d2[pos - 1];
                                        best_idx[pos] = best_idx[pos - 1];
                                        pos--;
                                    }
                                    best_d2[pos] = dv;
                                    best_idx[pos] = orig;
                                    worst = best_d2[k - 1];
                                    v_worst = _mm512_set1_pd(worst);
                                }
                            }
                        }
                    }
                    for (; i < end; i++) {
                        double dx = rx[i] - qx, dy = ry[i] - qy, dz = rz[i] - qz;
                        double dv = (dx * dx + dy * dy) + dz * dz;
                        int32_t orig = perm[i];
                        if (dv < worst || (dv == worst && orig < best_idx[k - 1])) {
                            int pos = k - 1;
                            while (pos > 0 && (dv < best_d2[pos - 1] ||
                                               (dv == best_d2[pos - 1] && orig < best_idx[pos - 1]))) {
                                best_d2[pos] = best_d2[pos - 1];
                                best_idx[pos] = best_idx[pos - 1];
                                pos--;
                            }
                            best_d2[pos] = dv;
                            best_idx[pos] = orig;
                            worst = best_d2[k - 1];
                        }
                    }
                    continue;
                }
                int32_t lid = nd->left, rid = nd->right;
                double ld2 = point_box_d2(qx, qy, qz, nodes[lid].bmin, nodes[lid].bmax);
                double rd2 = point_box_d2(qx, qy, qz, nodes[rid].bmin, nodes[rid].bmax);
                double bound = best_d2[k - 1];
                if (ld2 <= rd2) {
                    if (rd2 <= bound && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                    if (ld2 <= bound && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                } else {
                    if (ld2 <= bound && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                    if (rd2 <= bound && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                }
            }
        }
        double wsum = 0.0, sr = 0.0, sg = 0.0, sb = 0.0;
        for (int j = 0; j < k; j++) {
            double d = sqrt(best_d2[j]);
            double w = 1.0 / (d > 1e-4 ? d : 1e-4);
            wsum += w;
            Py_ssize_t bi = (Py_ssize_t)best_idx[j];
            if (rgb_is_u8) {
                sr += (double)rgb_u8[3 * bi + 0] * w;
                sg += (double)rgb_u8[3 * bi + 1] * w;
                sb += (double)rgb_u8[3 * bi + 2] * w;
            } else {
                sr += rgb_f64[3 * bi + 0] * w;
                sg += rgb_f64[3 * bi + 1] * w;
                sb += rgb_f64[3 * bi + 2] * w;
            }
        }
        double inv_w = 1.0 / wsum;
        double rr = rint(sr * inv_w);
        double gg = rint(sg * inv_w);
        double bb = rint(sb * inv_w);
        out[3 * q + 0] = (uint8_t)(rr < 0.0 ? 0.0 : (rr > 255.0 ? 255.0 : rr));
        out[3 * q + 1] = (uint8_t)(gg < 0.0 ? 0.0 : (gg > 255.0 ? 255.0 : gg));
        out[3 * q + 2] = (uint8_t)(bb < 0.0 ? 0.0 : (bb > 255.0 ? 255.0 : bb));
    }
}

__attribute__((target("avx2")))
static void knn_rgb_query_avx2(
    const double *rx, const double *ry, const double *rz,
    const int32_t *perm, const KDNode *nodes, int use_tree, int32_t m32,
    const double *qry, Py_ssize_t n, int k,
    int rgb_is_u8, const uint8_t *rgb_u8, const double *rgb_f64, uint8_t *out)
{
    typedef struct { int32_t nid; double d2; } StackItem;
    StackItem stk[64];
    double best_d2[32];
    int32_t best_idx[32];

    for (Py_ssize_t q = 0; q < n; q++) {
        double qx = qry[3 * q + 0], qy = qry[3 * q + 1], qz = qry[3 * q + 2];
        for (int j = 0; j < k; j++) { best_d2[j] = HUGE_VAL; best_idx[j] = INT32_MAX; }
        __m256d v_qx = _mm256_set1_pd(qx);
        __m256d v_qy = _mm256_set1_pd(qy);
        __m256d v_qz = _mm256_set1_pd(qz);
        if (!use_tree) {
            int32_t i = 0;
            double worst = best_d2[k - 1];
            __m256d v_worst = _mm256_set1_pd(worst);
            for (; i + 4 <= m32; i += 4) {
                __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(rx + i), v_qx);
                __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(ry + i), v_qy);
                __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(rz + i), v_qz);
                __m256d d2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)), _mm256_mul_pd(dz, dz));
                int mask = _mm256_movemask_pd(_mm256_cmp_pd(d2, v_worst, _CMP_LE_OQ));
                if (__builtin_expect(mask != 0, 0)) {
                    double tmp[4];
                    _mm256_storeu_pd(tmp, d2);
                    unsigned um = (unsigned)mask;
                    while (um) {
                        int lane = __builtin_ctz(um);
                        um &= (um - 1u);
                        int32_t orig = i + lane;
                        double dv = tmp[lane];
                        if (dv < worst || (dv == worst && orig < best_idx[k - 1])) {
                            int pos = k - 1;
                            while (pos > 0 && (dv < best_d2[pos - 1] ||
                                               (dv == best_d2[pos - 1] && orig < best_idx[pos - 1]))) {
                                best_d2[pos] = best_d2[pos - 1];
                                best_idx[pos] = best_idx[pos - 1];
                                pos--;
                            }
                            best_d2[pos] = dv;
                            best_idx[pos] = orig;
                            worst = best_d2[k - 1];
                            v_worst = _mm256_set1_pd(worst);
                        }
                    }
                }
            }
            for (; i < m32; i++) {
                double dx = rx[i] - qx, dy = ry[i] - qy, dz = rz[i] - qz;
                double dv = (dx * dx + dy * dy) + dz * dz;
                if (dv < worst || (dv == worst && i < best_idx[k - 1])) {
                    int pos = k - 1;
                    while (pos > 0 && (dv < best_d2[pos - 1] ||
                                       (dv == best_d2[pos - 1] && i < best_idx[pos - 1]))) {
                        best_d2[pos] = best_d2[pos - 1];
                        best_idx[pos] = best_idx[pos - 1];
                        pos--;
                    }
                    best_d2[pos] = dv;
                    best_idx[pos] = i;
                    worst = best_d2[k - 1];
                }
            }
        } else {
            int sp = 0;
            stk[sp].nid = 0;
            stk[sp].d2 = point_box_d2(qx, qy, qz, nodes[0].bmin, nodes[0].bmax);
            sp++;
            while (sp > 0) {
                StackItem cur = stk[--sp];
                if (cur.d2 > best_d2[k - 1]) continue;
                const KDNode *nd = &nodes[cur.nid];
                if (nd->axis < 0) {
                    int32_t i = nd->start, end = nd->end;
                    double worst = best_d2[k - 1];
                    __m256d v_worst = _mm256_set1_pd(worst);
                    for (; i + 4 <= end; i += 4) {
                        __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(rx + i), v_qx);
                        __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(ry + i), v_qy);
                        __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(rz + i), v_qz);
                        __m256d d2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)), _mm256_mul_pd(dz, dz));
                        int mask = _mm256_movemask_pd(_mm256_cmp_pd(d2, v_worst, _CMP_LE_OQ));
                        if (__builtin_expect(mask != 0, 0)) {
                            double tmp[4];
                            _mm256_storeu_pd(tmp, d2);
                            unsigned um = (unsigned)mask;
                            while (um) {
                                int lane = __builtin_ctz(um);
                                um &= (um - 1u);
                                int32_t orig = perm[i + lane];
                                double dv = tmp[lane];
                                if (dv < worst || (dv == worst && orig < best_idx[k - 1])) {
                                    int pos = k - 1;
                                    while (pos > 0 && (dv < best_d2[pos - 1] ||
                                                       (dv == best_d2[pos - 1] && orig < best_idx[pos - 1]))) {
                                        best_d2[pos] = best_d2[pos - 1];
                                        best_idx[pos] = best_idx[pos - 1];
                                        pos--;
                                    }
                                    best_d2[pos] = dv;
                                    best_idx[pos] = orig;
                                    worst = best_d2[k - 1];
                                    v_worst = _mm256_set1_pd(worst);
                                }
                            }
                        }
                    }
                    for (; i < end; i++) {
                        double dx = rx[i] - qx, dy = ry[i] - qy, dz = rz[i] - qz;
                        double dv = (dx * dx + dy * dy) + dz * dz;
                        int32_t orig = perm[i];
                        if (dv < worst || (dv == worst && orig < best_idx[k - 1])) {
                            int pos = k - 1;
                            while (pos > 0 && (dv < best_d2[pos - 1] ||
                                               (dv == best_d2[pos - 1] && orig < best_idx[pos - 1]))) {
                                best_d2[pos] = best_d2[pos - 1];
                                best_idx[pos] = best_idx[pos - 1];
                                pos--;
                            }
                            best_d2[pos] = dv;
                            best_idx[pos] = orig;
                            worst = best_d2[k - 1];
                        }
                    }
                    continue;
                }
                int32_t lid = nd->left, rid = nd->right;
                double ld2 = point_box_d2(qx, qy, qz, nodes[lid].bmin, nodes[lid].bmax);
                double rd2 = point_box_d2(qx, qy, qz, nodes[rid].bmin, nodes[rid].bmax);
                double bound = best_d2[k - 1];
                if (ld2 <= rd2) {
                    if (rd2 <= bound && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                    if (ld2 <= bound && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                } else {
                    if (ld2 <= bound && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                    if (rd2 <= bound && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                }
            }
        }
        double wsum = 0.0, sr = 0.0, sg = 0.0, sb = 0.0;
        for (int j = 0; j < k; j++) {
            double d = sqrt(best_d2[j]);
            double w = 1.0 / (d > 1e-4 ? d : 1e-4);
            wsum += w;
            Py_ssize_t bi = (Py_ssize_t)best_idx[j];
            if (rgb_is_u8) {
                sr += (double)rgb_u8[3 * bi + 0] * w;
                sg += (double)rgb_u8[3 * bi + 1] * w;
                sb += (double)rgb_u8[3 * bi + 2] * w;
            } else {
                sr += rgb_f64[3 * bi + 0] * w;
                sg += rgb_f64[3 * bi + 1] * w;
                sb += rgb_f64[3 * bi + 2] * w;
            }
        }
        double inv_w = 1.0 / wsum;
        double rr = rint(sr * inv_w);
        double gg = rint(sg * inv_w);
        double bb = rint(sb * inv_w);
        out[3 * q + 0] = (uint8_t)(rr < 0.0 ? 0.0 : (rr > 255.0 ? 255.0 : rr));
        out[3 * q + 1] = (uint8_t)(gg < 0.0 ? 0.0 : (gg > 255.0 ? 255.0 : gg));
        out[3 * q + 2] = (uint8_t)(bb < 0.0 ? 0.0 : (bb > 255.0 ? 255.0 : bb));
    }
}
#endif

static PyObject *py_transfer_rgb_knn_3d(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "transfer_rgb_knn_3d expects (ref_pts, ref_rgb, query_pts, k)");
        return NULL;
    }
    Py_ssize_t k_in = PyLong_AsSsize_t(args[3]);
    if (k_in == -1 && PyErr_Occurred()) return NULL;
    if (k_in < 1 || k_in > 32) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    Py_buffer ref_b = {0}, rgb_b = {0}, q_b = {0};
    int g_r = 0, g_c = 0, g_q = 0;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(args[0], &ref_b, PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_r = 1;
    if (PyObject_GetBuffer(args[1], &rgb_b, PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_c = 1;
    if (PyObject_GetBuffer(args[2], &q_b, PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_q = 1;

    if (ref_b.ndim != 2 || q_b.ndim != 2 || rgb_b.ndim != 2 ||
        ref_b.itemsize != (Py_ssize_t)sizeof(double) ||
        q_b.itemsize != (Py_ssize_t)sizeof(double) ||
        (rgb_b.itemsize != 1 && rgb_b.itemsize != (Py_ssize_t)sizeof(double)) ||
        ref_b.shape[1] != 3 || q_b.shape[1] != 3 || rgb_b.shape[1] != 3 ||
        rgb_b.shape[0] != ref_b.shape[0] ||
        ref_b.shape[0] <= 0 || ref_b.shape[0] > INT32_MAX) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    Py_ssize_t m = ref_b.shape[0];
    Py_ssize_t n = q_b.shape[0];
    int k = (int)(k_in < m ? k_in : m);
    if (n == 0) {
        ret = alloc_fast_2d(0, 3, 1, g_np_uint8, NULL);
        goto cleanup;
    }

    const double *ref = (const double *)ref_b.buf;
    const double *qry = (const double *)q_b.buf;
    int rgb_is_u8 = (rgb_b.itemsize == 1);
    const uint8_t *rgb_u8 = rgb_is_u8 ? (const uint8_t *)rgb_b.buf : NULL;
    const double *rgb_f64 = rgb_is_u8 ? NULL : (const double *)rgb_b.buf;

    ThreadScratch *sc = get_scratch();
    size_t min_leaf = (size_t)(KD_LEAF_SIZE >> 1);
    size_t max_nodes = (size_t)(2 * (((size_t)m + min_leaf - 1) / min_leaf) + 32);
    if (!sc ||
        ensure_p_idx(sc, (size_t)m) < 0 ||
        ensure_dbl_tmp(sc, (size_t)m * 3) < 0 ||
        ensure_kd_nodes(sc, max_nodes) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }

    double *rx = sc->dbl_tmp;
    double *ry = rx + m;
    double *rz = ry + m;
    int32_t *perm = sc->p_idx;
    KDNode *nodes = sc->kd_nodes;

    int non_finite = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < 3 * m; i++) {
        if (!isfinite(ref[i])) { non_finite = 1; break; }
    }
    if (!rgb_is_u8 && !non_finite) {
        for (Py_ssize_t i = 0; i < 3 * m; i++) {
            if (!isfinite(rgb_f64[i])) { non_finite = 1; break; }
        }
    }
    for (Py_ssize_t i = 0; i < 3 * n && !non_finite; i++) {
        if (!isfinite(qry[i])) { non_finite = 1; break; }
    }
    Py_END_ALLOW_THREADS
    if (non_finite) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    void *op = NULL;
    PyObject *out_arr = alloc_fast_2d(n, 3, 1, g_np_uint8, &op);
    if (!out_arr) goto cleanup;
    uint8_t *out = (uint8_t *)op;

    Py_BEGIN_ALLOW_THREADS
    int use_tree = (m > 32);
    if (!use_tree) {
        for (Py_ssize_t i = 0; i < m; i++) {
            rx[i] = ref[3 * i + 0];
            ry[i] = ref[3 * i + 1];
            rz[i] = ref[3 * i + 2];
        }
    } else {
        for (Py_ssize_t i = 0; i < m; i++) perm[i] = (int32_t)i;
        int32_t next_id = 0;
        kd_build_recursive(ref, perm, nodes, &next_id, 0, (int32_t)m);
        for (Py_ssize_t i = 0; i < m; i++) {
            Py_ssize_t src = (Py_ssize_t)perm[i];
            rx[i] = ref[3 * src + 0];
            ry[i] = ref[3 * src + 1];
            rz[i] = ref[3 * src + 2];
        }
    }

#if ES_HAVE_X86_RUNTIME_SIMD
    if (g_simd_level >= 2) {
        knn_rgb_query_avx512(rx, ry, rz, perm, nodes, use_tree, (int32_t)m, qry, n, k, rgb_is_u8, rgb_u8, rgb_f64, out);
    } else if (g_simd_level == 1) {
        knn_rgb_query_avx2(rx, ry, rz, perm, nodes, use_tree, (int32_t)m, qry, n, k, rgb_is_u8, rgb_u8, rgb_f64, out);
    } else
#endif
    {
        typedef struct { int32_t nid; double d2; } StackItem;
        StackItem stk[64];
        double best_d2[32];
        int32_t best_idx[32];

        for (Py_ssize_t q = 0; q < n; q++) {
            double qx = qry[3 * q + 0], qy = qry[3 * q + 1], qz = qry[3 * q + 2];
            for (int j = 0; j < k; j++) { best_d2[j] = HUGE_VAL; best_idx[j] = INT32_MAX; }
            if (!use_tree) {
                knn_leaf_update_scalar(rx, ry, rz, NULL, 0, (int32_t)m, qx, qy, qz, k, best_d2, best_idx);
            } else {
                int sp = 0;
                stk[sp].nid = 0;
                stk[sp].d2 = point_box_d2(qx, qy, qz, nodes[0].bmin, nodes[0].bmax);
                sp++;
                while (sp > 0) {
                    StackItem cur = stk[--sp];
                    if (cur.d2 > best_d2[k - 1]) continue;
                    const KDNode *nd = &nodes[cur.nid];
                    if (nd->axis < 0) {
                        knn_leaf_update_scalar(rx, ry, rz, perm, nd->start, nd->end, qx, qy, qz, k, best_d2, best_idx);
                        continue;
                    }
                    int32_t lid = nd->left, rid = nd->right;
                    double ld2 = point_box_d2(qx, qy, qz, nodes[lid].bmin, nodes[lid].bmax);
                    double rd2 = point_box_d2(qx, qy, qz, nodes[rid].bmin, nodes[rid].bmax);
                    double bound = best_d2[k - 1];
                    if (ld2 <= rd2) {
                        if (rd2 <= bound && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                        if (ld2 <= bound && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                    } else {
                        if (ld2 <= bound && sp < 64) { stk[sp].nid = lid; stk[sp].d2 = ld2; sp++; }
                        if (rd2 <= bound && sp < 64) { stk[sp].nid = rid; stk[sp].d2 = rd2; sp++; }
                    }
                }
            }
            double wsum = 0.0, sr = 0.0, sg = 0.0, sb = 0.0;
            for (int j = 0; j < k; j++) {
                double d = sqrt(best_d2[j]);
                double w = 1.0 / (d > 1e-4 ? d : 1e-4);
                wsum += w;
                Py_ssize_t bi = (Py_ssize_t)best_idx[j];
                if (rgb_is_u8) {
                    sr += (double)rgb_u8[3 * bi + 0] * w;
                    sg += (double)rgb_u8[3 * bi + 1] * w;
                    sb += (double)rgb_u8[3 * bi + 2] * w;
                } else {
                    sr += rgb_f64[3 * bi + 0] * w;
                    sg += rgb_f64[3 * bi + 1] * w;
                    sb += rgb_f64[3 * bi + 2] * w;
                }
            }
            double inv_w = 1.0 / wsum;
            double rr = rint(sr * inv_w);
            double gg = rint(sg * inv_w);
            double bb = rint(sb * inv_w);
            out[3 * q + 0] = (uint8_t)(rr < 0.0 ? 0.0 : (rr > 255.0 ? 255.0 : rr));
            out[3 * q + 1] = (uint8_t)(gg < 0.0 ? 0.0 : (gg > 255.0 ? 255.0 : gg));
            out[3 * q + 2] = (uint8_t)(bb < 0.0 ? 0.0 : (bb > 255.0 ? 255.0 : bb));
        }
    }
    Py_END_ALLOW_THREADS

    ret = out_arr;

cleanup:
    if (g_r) PyBuffer_Release(&ref_b);
    if (g_c) PyBuffer_Release(&rgb_b);
    if (g_q) PyBuffer_Release(&q_b);
    return ret;
}

typedef struct {
    int32_t iu;
    int32_t iv;
    int32_t pt_idx;
    int32_t pad;
    double wg;
    double wr;
} PlanarInlierRec;

typedef struct {
    int64_t key;
    int32_t idx;
} KeyIdxPair;

static int cmp_key_idx_pair(const void *a, const void *b) {
    int64_t ka = ((const KeyIdxPair *)a)->key;
    int64_t kb = ((const KeyIdxPair *)b)->key;
    return (ka < kb) ? -1 : (ka > kb);
}

static PyObject *py_planar_project_and_aggregate_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pts_o, *rgb_o, *norm_o, *bu_o, *bv_o, *uv0_o, *cam_o, *sel_o = NULL;
    double plane_offset, max_res, cos_limit, cell_size, obs_weight, spec_scale;
    int has_cam, use_physics, spec_thresh;
    Py_ssize_t max_grid_cells, max_cells;
    if (!PyArg_ParseTuple(
            args, "OOOOOOidddddiidOnn|O",
            &pts_o, &rgb_o, &norm_o, &bu_o, &bv_o, &uv0_o,
            &has_cam, &plane_offset, &max_res, &cos_limit, &cell_size, &obs_weight,
            &use_physics, &spec_thresh, &spec_scale, &cam_o, &max_grid_cells, &max_cells,
            &sel_o)) {
        return NULL;
    }

    Py_buffer pts_b = {0}, rgb_b = {0}, norm_b = {0}, bu_b = {0}, bv_b = {0}, uv0_b = {0}, cam_b = {0}, sel_b = {0};
    int g_p = 0, g_r = 0, g_n = 0, g_u = 0, g_v = 0, g_uv = 0, g_c = 0, g_s = 0;
    const int has_sel = (sel_o != NULL && sel_o != Py_None);
    int32_t *grid_or_ht = NULL;
    CellAggEntry *entries = NULL;
    int heap_grid_entries = 0;
    KeyIdxPair *ki = NULL;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(pts_o, &pts_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_p = 1;
    if (PyObject_GetBuffer(rgb_o, &rgb_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_r = 1;
    if (PyObject_GetBuffer(norm_o, &norm_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_n = 1;
    if (PyObject_GetBuffer(bu_o, &bu_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_u = 1;
    if (PyObject_GetBuffer(bv_o, &bv_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_v = 1;
    if (PyObject_GetBuffer(uv0_o, &uv0_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_uv = 1;
    if (has_cam) {
        if (PyObject_GetBuffer(cam_o, &cam_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        g_c = 1;
    }
    if (has_sel) {
        if (PyObject_GetBuffer(sel_o, &sel_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        g_s = 1;
    }

    if (pts_b.ndim != 2 || pts_b.shape[1] != 3 ||
        (pts_b.itemsize != 4 && pts_b.itemsize != 8) ||
        rgb_b.ndim != 2 || rgb_b.shape[0] != pts_b.shape[0] || rgb_b.shape[1] != 3 || rgb_b.itemsize != 1 ||
        (has_sel && (sel_b.ndim != 1 || sel_b.shape[0] != pts_b.shape[0] || sel_b.itemsize != 1)) ||
        pts_b.shape[0] > INT32_MAX) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    Py_ssize_t n = pts_b.shape[0];
    if (n == 0) {
        PyObject *k_e = alloc_fast_1d(0, 8, g_np_int64, NULL);
        PyObject *p_e = alloc_fast_2d(0, 3, 8, g_np_float64, NULL);
        PyObject *c_e = alloc_fast_2d(0, 3, 4, g_np_float32, NULL);
        PyObject *w_e = alloc_fast_1d(0, 8, g_np_float64, NULL);
        if (k_e && p_e && c_e && w_e) {
            ret = Py_BuildValue("(NNNNnO)", k_e, p_e, c_e, w_e, (Py_ssize_t)0, Py_False);
        } else {
            Py_XDECREF(k_e); Py_XDECREF(p_e); Py_XDECREF(c_e); Py_XDECREF(w_e);
        }
        goto cleanup;
    }

    const int pts_is_f32 = (pts_b.itemsize == 4);
    const float *pts_f32 = pts_is_f32 ? (const float *)pts_b.buf : NULL;
    const double *pts_f64 = pts_is_f32 ? NULL : (const double *)pts_b.buf;
    const uint8_t *rgb = (const uint8_t *)rgb_b.buf;
    const uint8_t *sel = has_sel ? (const uint8_t *)sel_b.buf : NULL;
    const double *nrm = (const double *)norm_b.buf;
    const double *bu = (const double *)bu_b.buf;
    const double *bv = (const double *)bv_b.buf;
    const double *uv0 = (const double *)uv0_b.buf;
    const double *cam = has_cam ? (const double *)cam_b.buf : NULL;

    const double nx = nrm[0], ny = nrm[1], nz = nrm[2];
    const double ux = bu[0], uy = bu[1], uz = bu[2];
    const double vx = bv[0], vy = bv[1], vz = bv[2];
    const double u0 = uv0[0], v0 = uv0[1];
    const double cx = has_cam ? cam[0] : 0.0;
    const double cy = has_cam ? cam[1] : 0.0;
    const double cz = has_cam ? cam[2] : 0.0;
    const double inv_cell = 1.0 / cell_size;
    const int64_t BIAS = (1LL << 30);
    const double COORD_LIM = (double)((1LL << 29) - 1LL);

    ThreadScratch *sc = get_scratch();
    if (!sc || ensure_byte_tmp(sc, (size_t)n * sizeof(PlanarInlierRec)) < 0 ||
        ensure_grid(sc, 65536ULL) < 0 ||
        ensure_dbl_tmp(sc, ((size_t)(n < 65536 ? n : 65536) * sizeof(CellAggEntry)) / sizeof(double) + 16) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }
    PlanarInlierRec *inliers = (PlanarInlierRec *)sc->byte_tmp;

    Py_ssize_t inlier_count = 0;
    Py_ssize_t n_entries = 0;
    int budget_exceeded = 0;
    int oom = 0;
    int used_dense_grid = 0;
    int64_t min_iu = INT64_MAX, max_iu = INT64_MIN;
    int64_t min_iv = INT64_MAX, max_iv = INT64_MIN;
    size_t grid_sz = 0;

    Py_BEGIN_ALLOW_THREADS
    /* Pass 1: Filter points onto reference plane, compute physics weights & cell indices. */
    for (Py_ssize_t i = 0; i < n; i++) {
        if (sel && !sel[i]) continue;
        double x, y, z;
        if (pts_is_f32) {
            x = (double)pts_f32[3 * i + 0];
            y = (double)pts_f32[3 * i + 1];
            z = (double)pts_f32[3 * i + 2];
        } else {
            x = pts_f64[3 * i + 0];
            y = pts_f64[3 * i + 1];
            z = pts_f64[3 * i + 2];
        }
        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) continue;
        double res = fabs(x * nx + y * ny + z * nz + plane_offset);
        if (res > max_res) continue;

        double cos_t = 1.0, rn = 1.0;
        if (has_cam) {
            double dx = cx - x, dy = cy - y, dz = cz - z;
            rn = sqrt(dx * dx + dy * dy + dz * dz);
            double denom = rn > 1e-12 ? rn : 1e-12;
            cos_t = fabs((dx * nx + dy * ny + dz * nz) / denom);
            if (cos_limit > 0.0 && (rn <= 1e-6 || cos_t < cos_limit)) continue;
        }
        Py_ssize_t cur_inlier = inlier_count++;
        if (budget_exceeded) continue;

        double u = x * ux + y * uy + z * uz;
        double v = x * vx + y * vy + z * vz;
        double su = floor((u - u0) * inv_cell + 1e-9);
        double sv = floor((v - v0) * inv_cell + 1e-9);
        if (!isfinite(su) || !isfinite(sv) || fabs(su) >= COORD_LIM || fabs(sv) >= COORD_LIM) {
            budget_exceeded = 1;
            continue;
        }
        int64_t iu = (int64_t)su;
        int64_t iv = (int64_t)sv;
        if (iu < min_iu) min_iu = iu;
        if (iu > max_iu) max_iu = iu;
        if (iv < min_iv) min_iv = iv;
        if (iv > max_iv) max_iv = iv;
        if ((max_iu - min_iu + 1) * (max_iv - min_iv + 1) > (int64_t)max_grid_cells) {
            budget_exceeded = 1;
            continue;
        }

        double wg = 1.0, wr = 1.0;
        if (use_physics && has_cam) {
            double ze = rn > 0.2 ? rn : 0.2;
            wg = cos_t / (ze * ze);
            if (wg < 0.05) wg = 0.05;
            else if (wg > 8.0) wg = 8.0;
            if (spec_thresh <= 255) {
                uint8_t r8 = rgb[3 * i + 0], g8 = rgb[3 * i + 1], b8 = rgb[3 * i + 2];
                int mn = r8 < g8 ? (r8 < b8 ? r8 : b8) : (g8 < b8 ? g8 : b8);
                wr = (mn >= spec_thresh) ? (wg * spec_scale) : wg;
            } else {
                wr = wg;
            }
        }
        inliers[cur_inlier].iu = (int32_t)iu;
        inliers[cur_inlier].iv = (int32_t)iv;
        inliers[cur_inlier].pt_idx = (int32_t)i;
        inliers[cur_inlier].wg = wg;
        inliers[cur_inlier].wr = wr;
    }

    if (!budget_exceeded && inlier_count > 0) {
        int64_t span_u = max_iu - min_iu + 1;
        int64_t span_v = max_iv - min_iv + 1;
        grid_sz = (size_t)span_u * (size_t)span_v;

        if (grid_sz <= 65536ULL) {
            /* Fast Path A: L2-cache direct 2D grid (zero malloc/page-fault, zero hashing, zero sorting). */
            used_dense_grid = 1;
            grid_or_ht = sc->grid;
            entries = (CellAggEntry *)sc->dbl_tmp;
            {
                memset(grid_or_ht, 0xff, grid_sz * sizeof(int32_t));
                for (Py_ssize_t k = 0; k < inlier_count; k++) {
                    int64_t iu = (int64_t)inliers[k].iu;
                    int64_t iv = (int64_t)inliers[k].iv;
                    Py_ssize_t pi = (Py_ssize_t)inliers[k].pt_idx;
                    double wg = inliers[k].wg;
                    double wr = inliers[k].wr;
                    double x = pts_is_f32 ? (double)pts_f32[3 * pi + 0] : pts_f64[3 * pi + 0];
                    double y = pts_is_f32 ? (double)pts_f32[3 * pi + 1] : pts_f64[3 * pi + 1];
                    double z = pts_is_f32 ? (double)pts_f32[3 * pi + 2] : pts_f64[3 * pi + 2];
                    uint8_t r8 = rgb[3 * pi + 0], g8 = rgb[3 * pi + 1], b8 = rgb[3 * pi + 2];
                    size_t gidx = (size_t)(iu - min_iu) * (size_t)span_v + (size_t)(iv - min_iv);
                    int32_t e = grid_or_ht[gidx];
                    if (e < 0) {
                        e = (int32_t)n_entries++;
                        grid_or_ht[gidx] = e;
                        entries[e].key = ((iu + BIAS) << 32) | ((iv + BIAS) & 0xFFFFFFFFLL);
                        entries[e].sx = x * wg;
                        entries[e].sy = y * wg;
                        entries[e].sz = z * wg;
                        entries[e].sr = (double)r8 * wr;
                        entries[e].sg = (double)g8 * wr;
                        entries[e].sb = (double)b8 * wr;
                        entries[e].sw_geo = wg;
                        entries[e].sw_rgb = wr;
                        entries[e].cnt = 1;
                    } else {
                        entries[e].sx += x * wg;
                        entries[e].sy += y * wg;
                        entries[e].sz += z * wg;
                        entries[e].sr += (double)r8 * wr;
                        entries[e].sg += (double)g8 * wr;
                        entries[e].sb += (double)b8 * wr;
                        entries[e].sw_geo += wg;
                        entries[e].sw_rgb += wr;
                        entries[e].cnt += 1;
                    }
                }
            }
        } else {
            /* Path B: Open-addressing hash table + lightweight 16-byte (key, idx) sort. */
            size_t hcap = 2048;
            size_t want_e = (size_t)(inlier_count < max_cells ? inlier_count : max_cells) + 64;
            if (want_e > (size_t)inlier_count) want_e = (size_t)inlier_count;
            while (hcap < want_e * 2 && hcap < (1ULL << 22)) hcap <<= 1;
            size_t ecap = hcap >> 1;
            if (ecap < 1024) ecap = 1024;
            grid_or_ht = (int32_t *)malloc(hcap * sizeof(int32_t));
            entries = (CellAggEntry *)malloc(ecap * sizeof(CellAggEntry));
            heap_grid_entries = 1;
            if (!grid_or_ht || !entries) {
                oom = 1;
            } else {
                memset(grid_or_ht, 0xff, hcap * sizeof(int32_t));
                size_t hmask = hcap - 1ULL;
                for (Py_ssize_t k = 0; k < inlier_count; k++) {
                    int64_t iu = (int64_t)inliers[k].iu;
                    int64_t iv = (int64_t)inliers[k].iv;
                    Py_ssize_t pi = (Py_ssize_t)inliers[k].pt_idx;
                    double wg = inliers[k].wg;
                    double wr = inliers[k].wr;
                    double x = pts_is_f32 ? (double)pts_f32[3 * pi + 0] : pts_f64[3 * pi + 0];
                    double y = pts_is_f32 ? (double)pts_f32[3 * pi + 1] : pts_f64[3 * pi + 1];
                    double z = pts_is_f32 ? (double)pts_f32[3 * pi + 2] : pts_f64[3 * pi + 2];
                    uint8_t r8 = rgb[3 * pi + 0], g8 = rgb[3 * pi + 1], b8 = rgb[3 * pi + 2];
                    int64_t key = ((iu + BIAS) << 32) | ((iv + BIAS) & 0xFFFFFFFFLL);

                    if ((size_t)n_entries + 1 >= ecap) {
                        size_t new_hcap = hcap << 1;
                        size_t new_ecap = new_hcap >> 1;
                        int32_t *new_ht = (int32_t *)malloc(new_hcap * sizeof(int32_t));
                        CellAggEntry *new_ent = (CellAggEntry *)realloc(entries, new_ecap * sizeof(CellAggEntry));
                        if (!new_ht || !new_ent) {
                            free(new_ht);
                            if (new_ent) entries = new_ent;
                            oom = 1;
                            break;
                        }
                        entries = new_ent;
                        free(grid_or_ht);
                        grid_or_ht = new_ht;
                        hcap = new_hcap;
                        ecap = new_ecap;
                        hmask = hcap - 1ULL;
                        memset(grid_or_ht, 0xff, hcap * sizeof(int32_t));
                        for (Py_ssize_t j = 0; j < n_entries; j++) {
                            size_t p2 = (size_t)hash_u64((uint64_t)entries[j].key) & hmask;
                            while (grid_or_ht[p2] >= 0) p2 = (p2 + 1ULL) & hmask;
                            grid_or_ht[p2] = (int32_t)j;
                        }
                    }

                    size_t pos = (size_t)hash_u64((uint64_t)key) & hmask;
                    while (1) {
                        int32_t e = grid_or_ht[pos];
                        if (e < 0) {
                            e = (int32_t)n_entries++;
                            grid_or_ht[pos] = e;
                            entries[e].key = key;
                            entries[e].sx = x * wg;
                            entries[e].sy = y * wg;
                            entries[e].sz = z * wg;
                            entries[e].sr = (double)r8 * wr;
                            entries[e].sg = (double)g8 * wr;
                            entries[e].sb = (double)b8 * wr;
                            entries[e].sw_geo = wg;
                            entries[e].sw_rgb = wr;
                            entries[e].cnt = 1;
                            break;
                        }
                        if (entries[e].key == key) {
                            entries[e].sx += x * wg;
                            entries[e].sy += y * wg;
                            entries[e].sz += z * wg;
                            entries[e].sr += (double)r8 * wr;
                            entries[e].sg += (double)g8 * wr;
                            entries[e].sb += (double)b8 * wr;
                            entries[e].sw_geo += wg;
                            entries[e].sw_rgb += wr;
                            entries[e].cnt += 1;
                            break;
                        }
                        pos = (pos + 1ULL) & hmask;
                    }
                }
                if (!oom && n_entries > 1) {
                    ki = (KeyIdxPair *)malloc((size_t)n_entries * sizeof(KeyIdxPair));
                    if (!ki) {
                        oom = 1;
                    } else {
                        for (Py_ssize_t j = 0; j < n_entries; j++) {
                            ki[j].key = entries[j].key;
                            ki[j].idx = (int32_t)j;
                        }
                        qsort(ki, (size_t)n_entries, sizeof(KeyIdxPair), cmp_key_idx_pair);
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (oom) { PyErr_NoMemory(); goto cleanup; }
    if (budget_exceeded || n_entries == 0) {
        PyObject *k_e = alloc_fast_1d(0, 8, g_np_int64, NULL);
        PyObject *p_e = alloc_fast_2d(0, 3, 8, g_np_float64, NULL);
        PyObject *c_e = alloc_fast_2d(0, 3, 4, g_np_float32, NULL);
        PyObject *w_e = alloc_fast_1d(0, 8, g_np_float64, NULL);
        if (k_e && p_e && c_e && w_e) {
            ret = Py_BuildValue("(NNNNnO)", k_e, p_e, c_e, w_e, inlier_count,
                                budget_exceeded ? Py_True : Py_False);
        } else {
            Py_XDECREF(k_e); Py_XDECREF(p_e); Py_XDECREF(c_e); Py_XDECREF(w_e);
        }
        goto cleanup;
    }

    Py_ssize_t n_out = (n_entries > max_cells) ? max_cells : n_entries;
    void *kp = NULL, *pp = NULL, *cp = NULL, *wp = NULL;
    PyObject *k_arr = alloc_fast_1d(n_out, 8, g_np_int64, &kp);
    PyObject *p_arr = alloc_fast_2d(n_out, 3, 8, g_np_float64, &pp);
    PyObject *c_arr = alloc_fast_2d(n_out, 3, 4, g_np_float32, &cp);
    PyObject *w_arr = alloc_fast_1d(n_out, 8, g_np_float64, &wp);
    if (!k_arr || !p_arr || !c_arr || !w_arr) {
        Py_XDECREF(k_arr); Py_XDECREF(p_arr); Py_XDECREF(c_arr); Py_XDECREF(w_arr);
        goto cleanup;
    }
    int64_t *ko = (int64_t *)kp;
    double *po = (double *)pp;
    float *co = (float *)cp;
    double *wo = (double *)wp;

    Py_BEGIN_ALLOW_THREADS
    if (used_dense_grid) {
        Py_ssize_t out_i = 0;
        for (size_t gidx = 0; gidx < grid_sz && out_i < n_out; gidx++) {
            int32_t eidx = grid_or_ht[gidx];
            if (eidx < 0) continue;
            const CellAggEntry *e = &entries[eidx];
            ko[out_i] = e->key;
            double inv_g = 1.0 / e->sw_geo;
            double inv_r = 1.0 / e->sw_rgb;
            po[3 * out_i + 0] = e->sx * inv_g;
            po[3 * out_i + 1] = e->sy * inv_g;
            po[3 * out_i + 2] = e->sz * inv_g;
            co[3 * out_i + 0] = (float)(e->sr * inv_r);
            co[3 * out_i + 1] = (float)(e->sg * inv_r);
            co[3 * out_i + 2] = (float)(e->sb * inv_r);
            wo[out_i] = obs_weight * (e->sw_geo / (double)e->cnt);
            out_i++;
        }
    } else if (ki != NULL) {
        for (Py_ssize_t i = 0; i < n_out; i++) {
            const CellAggEntry *e = &entries[ki[i].idx];
            ko[i] = e->key;
            double inv_g = 1.0 / e->sw_geo;
            double inv_r = 1.0 / e->sw_rgb;
            po[3 * i + 0] = e->sx * inv_g;
            po[3 * i + 1] = e->sy * inv_g;
            po[3 * i + 2] = e->sz * inv_g;
            co[3 * i + 0] = (float)(e->sr * inv_r);
            co[3 * i + 1] = (float)(e->sg * inv_r);
            co[3 * i + 2] = (float)(e->sb * inv_r);
            wo[i] = obs_weight * (e->sw_geo / (double)e->cnt);
        }
    } else {
        for (Py_ssize_t i = 0; i < n_out; i++) {
            const CellAggEntry *e = &entries[i];
            ko[i] = e->key;
            double inv_g = 1.0 / e->sw_geo;
            double inv_r = 1.0 / e->sw_rgb;
            po[3 * i + 0] = e->sx * inv_g;
            po[3 * i + 1] = e->sy * inv_g;
            po[3 * i + 2] = e->sz * inv_g;
            co[3 * i + 0] = (float)(e->sr * inv_r);
            co[3 * i + 1] = (float)(e->sg * inv_r);
            co[3 * i + 2] = (float)(e->sb * inv_r);
            wo[i] = obs_weight * (e->sw_geo / (double)e->cnt);
        }
    }
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("(NNNNnO)", k_arr, p_arr, c_arr, w_arr, inlier_count, Py_False);

cleanup:
    if (heap_grid_entries) { free(grid_or_ht); free(entries); }
    free(ki);
    if (g_p) PyBuffer_Release(&pts_b);
    if (g_r) PyBuffer_Release(&rgb_b);
    if (g_n) PyBuffer_Release(&norm_b);
    if (g_u) PyBuffer_Release(&bu_b);
    if (g_v) PyBuffer_Release(&bv_b);
    if (g_uv) PyBuffer_Release(&uv0_b);
    if (g_c) PyBuffer_Release(&cam_b);
    if (g_s) PyBuffer_Release(&sel_b);
    return ret;
}

static PyObject *py_planar_free_space_carve_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *act_o, *bw_o, *ww_o, *fxyz_o, *tobs_o, *conf_o, *depth_o, *sel_o, *rot_o, *trans_o;
    Py_ssize_t count;
    double fx, fy, cx, cy, min_d, max_d, margin;
    int min_obs_commit, conf_thresh;
    if (!PyArg_ParseTuple(
            args, "OOOOOOnOOOOdddddddii",
            &act_o, &bw_o, &ww_o, &fxyz_o, &tobs_o, &conf_o, &count,
            &depth_o, &sel_o, &rot_o, &trans_o,
            &fx, &fy, &cx, &cy, &min_d, &max_d, &margin,
            &min_obs_commit, &conf_thresh)) {
        return NULL;
    }

    Py_buffer b[10];
    int has_buf[10] = {0};
    PyObject *objs[10] = {act_o, bw_o, ww_o, fxyz_o, tobs_o, conf_o, depth_o, sel_o, rot_o, trans_o};
    int flags[10] = {
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
    };
    const int has_sel = (sel_o != Py_None);
    int32_t *evict = NULL;
    PyObject *ret = NULL;
    for (int i = 0; i < 10; i++) {
        if (i == 7 && !has_sel) continue;
        if (PyObject_GetBuffer(objs[i], &b[i], flags[i]) < 0) goto cleanup;
        has_buf[i] = 1;
    }
    if (b[6].ndim != 2 || (b[6].itemsize != 4 && b[6].itemsize != 8) ||
        (has_sel && (b[7].ndim != 2 || b[7].itemsize != 1 ||
                     b[6].shape[0] != b[7].shape[0] || b[6].shape[1] != b[7].shape[1]))) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    const uint8_t *act = (const uint8_t *)b[0].buf;
    const double *bw = (const double *)b[1].buf;
    const double *ww = (const double *)b[2].buf;
    const double *fxyz = (const double *)b[3].buf;
    const int32_t *tobs = (const int32_t *)b[4].buf;
    int32_t *conf = (int32_t *)b[5].buf;
    const int depth_f32 = (b[6].itemsize == 4);
    const float *df32 = depth_f32 ? (const float *)b[6].buf : NULL;
    const double *df64 = depth_f32 ? NULL : (const double *)b[6].buf;
    const uint8_t *sel = has_sel ? (const uint8_t *)b[7].buf : NULL;
    const double *rot = (const double *)b[8].buf;
    const double *trans = (const double *)b[9].buf;
    const Py_ssize_t H = b[6].shape[0];
    const Py_ssize_t W = b[6].shape[1];

    evict = (int32_t *)malloc((size_t)(count > 0 ? count : 1) * sizeof(int32_t));
    if (!evict) { PyErr_NoMemory(); goto cleanup; }

    Py_ssize_t n_evict = 0;
    const int committed_thresh = (conf_thresh + 1 > 3) ? (conf_thresh + 1) : 3;
    const double tx = trans[0], ty = trans[1], tz = trans[2];
    const double r00 = rot[0], r01 = rot[1], r02 = rot[2];
    const double r10 = rot[3], r11 = rot[4], r12 = rot[5];
    const double r20 = rot[6], r21 = rot[7], r22 = rot[8];

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t s = 0; s < count; s++) {
        if (!act[s]) continue;
        if (bw[s] + ww[s] <= 0.0) continue;
        double dx = fxyz[3 * s + 0] - tx;
        double dy = fxyz[3 * s + 1] - ty;
        double dz = fxyz[3 * s + 2] - tz;
        double zc = dx * r02 + dy * r12 + dz * r22;
        if (zc < min_d || zc > max_d) continue;
        double xc = dx * r00 + dy * r10 + dz * r20;
        double yc = dx * r01 + dy * r11 + dz * r21;
        long long u = llrint(fx * (xc / zc) + cx);
        long long v = llrint(fy * (yc / zc) + cy);
        if (u < 1 || u >= W - 1 || v < 1 || v >= H - 1) continue;
        Py_ssize_t idx = (Py_ssize_t)v * W + (Py_ssize_t)u;
        if (sel && !sel[idx]) continue;
        double md = depth_f32 ? (double)df32[idx] : df64[idx];
        if (!isfinite(md) || md < min_d || md > max_d) continue;
        if (fabs(md - zc) <= margin) {
            conf[s] = 0;
            continue;
        }
        if (md <= zc + margin) continue;

        int nbr_ok = 1;
        for (Py_ssize_t dv = -1; dv <= 1 && nbr_ok; dv++) {
            Py_ssize_t row_off = ((Py_ssize_t)v + dv) * W + (Py_ssize_t)u;
            for (Py_ssize_t du = -1; du <= 1; du++) {
                if (dv == 0 && du == 0) continue;
                Py_ssize_t ni = row_off + du;
                if (sel && !sel[ni]) { nbr_ok = 0; break; }
                double nd = depth_f32 ? (double)df32[ni] : df64[ni];
                if (!isfinite(nd) || nd <= zc + margin || nd > max_d) { nbr_ok = 0; break; }
            }
        }
        if (!nbr_ok) continue;
        conf[s] += 1;
        int thr = (tobs[s] >= min_obs_commit) ? committed_thresh : conf_thresh;
        if (conf[s] >= thr) {
            evict[n_evict++] = (int32_t)s;
        }
    }
    Py_END_ALLOW_THREADS

    void *op = NULL;
    PyObject *out_arr = alloc_fast_1d(n_evict, 4, g_np_int32, &op);
    if (!out_arr) goto cleanup;
    if (n_evict > 0) memcpy(op, evict, (size_t)n_evict * sizeof(int32_t));
    ret = out_arr;

cleanup:
    free(evict);
    for (int i = 0; i < 10; i++) {
        if (has_buf[i]) PyBuffer_Release(&b[i]);
    }
    return ret;
}

static PyObject *py_fit_plane_ransac_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "fit_plane_ransac_fast expects (points, triplets, thresh, min_points)");
        return NULL;
    }
    double thresh = PyFloat_AsDouble(args[2]);
    if (thresh == -1.0 && PyErr_Occurred()) return NULL;
    Py_ssize_t min_points = PyLong_AsSsize_t(args[3]);
    if (min_points == -1 && PyErr_Occurred()) return NULL;

    Py_buffer pts_b = {0}, tri_b = {0};
    int g_p = 0, g_t = 0;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(args[0], &pts_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_p = 1;
    if (PyObject_GetBuffer(args[1], &tri_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_t = 1;

    if (pts_b.ndim != 2 || pts_b.shape[1] != 3 || pts_b.itemsize != 8 ||
        tri_b.ndim != 2 || tri_b.shape[1] != 3 || tri_b.itemsize != 8) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    Py_ssize_t n = pts_b.shape[0];
    Py_ssize_t k_tri = tri_b.shape[0];
    if (n < 3 || k_tri <= 0) {
        ret = Py_None; Py_INCREF(ret); goto cleanup;
    }

    const double *pts = (const double *)pts_b.buf;
    const int64_t *tri = (const int64_t *)tri_b.buf;

    Py_ssize_t best_count = 0;
    Py_ssize_t best_t = -1;
    double best_nx = 0.0, best_ny = 0.0, best_nz = 0.0;
    double best_fx = 0.0, best_fy = 0.0, best_fz = 0.0;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t t = 0; t < k_tri; t++) {
        int64_t i0 = tri[3 * t + 0], i1 = tri[3 * t + 1], i2 = tri[3 * t + 2];
        if (i0 < 0 || i0 >= n || i1 < 0 || i1 >= n || i2 < 0 || i2 >= n) continue;
        double f0x = pts[3 * i0 + 0], f0y = pts[3 * i0 + 1], f0z = pts[3 * i0 + 2];
        double ax = pts[3 * i1 + 0] - f0x, ay = pts[3 * i1 + 1] - f0y, az = pts[3 * i1 + 2] - f0z;
        double bx = pts[3 * i2 + 0] - f0x, by = pts[3 * i2 + 1] - f0y, bz = pts[3 * i2 + 2] - f0z;
        double nx = ay * bz - az * by;
        double ny = az * bx - ax * bz;
        double nz = ax * by - ay * bx;
        double nrm = sqrt(nx * nx + ny * ny + nz * nz);
        if (nrm <= 1e-10) continue;
        double inv_n = 1.0 / nrm;
        nx *= inv_n; ny *= inv_n; nz *= inv_n;

        Py_ssize_t cnt = 0;
        Py_ssize_t need = best_count + 1;
        for (Py_ssize_t i = 0; i < n; i++) {
            double dx = pts[3 * i + 0] - f0x;
            double dy = pts[3 * i + 1] - f0y;
            double dz = pts[3 * i + 2] - f0z;
            if (fabs(dx * nx + dy * ny + dz * nz) <= thresh) cnt++;
            if (cnt + (n - 1 - i) < need) break;
        }
        if (cnt > best_count) {
            best_count = cnt;
            best_t = t;
            best_nx = nx; best_ny = ny; best_nz = nz;
            best_fx = f0x; best_fy = f0y; best_fz = f0z;
        }
    }
    Py_END_ALLOW_THREADS

    if (best_t < 0 || best_count < min_points) {
        ret = Py_None; Py_INCREF(ret); goto cleanup;
    }

    void *mp = NULL;
    PyObject *mask_arr = alloc_fast_1d(n, 1, g_np_bool, &mp);
    if (!mask_arr) goto cleanup;
    uint8_t *mask = (uint8_t *)mp;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        double dx = pts[3 * i + 0] - best_fx;
        double dy = pts[3 * i + 1] - best_fy;
        double dz = pts[3 * i + 2] - best_fz;
        mask[i] = (uint8_t)(fabs(dx * best_nx + dy * best_ny + dz * best_nz) <= thresh);
    }
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("(Nn)", mask_arr, best_count);

cleanup:
    if (g_p) PyBuffer_Release(&pts_b);
    if (g_t) PyBuffer_Release(&tri_b);
    return ret;
}

/* ============================================================================
 * 18. v2.6.0 Kernels: Single-Pass Hash-Table Surface Mesher, 3D Cylinder Axis
 *     Fitter, Sparse-Coordinate RLE Encoder, & ESTM Vertex Quantizer
 * ============================================================================ */

static inline int32_t ht_probe_slot(
    const int64_t *hk, const int32_t *hs, size_t hmask, int64_t key)
{
    size_t pos = (size_t)hash_u64((uint64_t)key) & hmask;
    while (1) {
        int64_t tk = hk[pos];
        if (tk == key) return hs[pos];
        if (tk == HT_EMPTY) return -1;
        pos = (pos + 1ULL) & hmask;
    }
}

static PyObject *py_planar_mesh_snapshot_ht_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *act_o, *bw_o, *ww_o, *tobs_o, *iu_o, *iv_o, *fxyz_o, *frgb_o, *lts_o, *hk_o, *hs_o, *nrm_o;
    Py_ssize_t count, min_sup, max_grid_cells, max_verts;
    int min_obs;
    double max_edge, max_nstep;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOOnOinddnn",
            &act_o, &bw_o, &ww_o, &tobs_o, &iu_o, &iv_o, &fxyz_o, &frgb_o, &lts_o, &hk_o, &hs_o,
            &count, &nrm_o, &min_obs, &min_sup, &max_edge, &max_nstep, &max_grid_cells, &max_verts)) {
        return NULL;
    }

    Py_buffer b[12];
    PyObject *objs[12] = {act_o, bw_o, ww_o, tobs_o, iu_o, iv_o, fxyz_o, frgb_o, lts_o, hk_o, hs_o, nrm_o};
    int got = 0;
    uint8_t *committed = NULL;
    uint8_t *used = NULL;
    int32_t *slot_to_vid = NULL;
    int32_t *tri_buf = NULL;
    double *fn_buf = NULL;
    PyObject *ret = NULL;

    for (int i = 0; i < 12; i++) {
        if (PyObject_GetBuffer(objs[i], &b[i], PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        got++;
    }

    const uint8_t *act = (const uint8_t *)b[0].buf;
    const double *bw = (const double *)b[1].buf;
    const double *ww = (const double *)b[2].buf;
    const int32_t *tobs = (const int32_t *)b[3].buf;
    const int32_t *iu = (const int32_t *)b[4].buf;
    const int32_t *iv = (const int32_t *)b[5].buf;
    const double *fxyz = (const double *)b[6].buf;
    const uint8_t *frgb = (const uint8_t *)b[7].buf;
    const int64_t *lts = (const int64_t *)b[8].buf;
    const int64_t *hk = (const int64_t *)b[9].buf;
    const int32_t *hs = (const int32_t *)b[10].buf;
    const double *nrm = (const double *)b[11].buf;
    size_t hcap = (size_t)b[9].shape[0];
    size_t hmask = hcap - 1ULL;

    if (count <= 0) {
        ret = Py_BuildValue("(inOOOOOOOO)", 1, (Py_ssize_t)0,
                            Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None);
        goto cleanup;
    }

    committed = (uint8_t *)calloc((size_t)count, 1);
    if (!committed) { PyErr_NoMemory(); goto cleanup; }

    Py_ssize_t num_committed = 0;
    int64_t min_iu = INT64_MAX, max_iu = INT64_MIN;
    int64_t min_iv = INT64_MAX, max_iv = INT64_MIN;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t s = 0; s < count; s++) {
        if (act[s] && tobs[s] >= min_obs && (bw[s] + ww[s]) > 0.0) {
            committed[s] = 1;
            num_committed++;
            int64_t u = (int64_t)iu[s];
            int64_t v = (int64_t)iv[s];
            if (u < min_iu) min_iu = u;
            if (u > max_iu) max_iu = u;
            if (v < min_iv) min_iv = v;
            if (v > max_iv) max_iv = v;
        }
    }
    Py_END_ALLOW_THREADS

    if (num_committed < min_sup) {
        ret = Py_BuildValue("(inOOOOOOOO)", 1, num_committed,
                            Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None);
        goto cleanup;
    }
    if (num_committed > max_verts) {
        ret = Py_BuildValue("(inOOOOOOOO)", 3, num_committed,
                            Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None);
        goto cleanup;
    }
    if ((max_iu - min_iu + 1) * (max_iv - min_iv + 1) > (int64_t)max_grid_cells) {
        ret = Py_BuildValue("(inOOOOOOOO)", 2, num_committed,
                            Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None);
        goto cleanup;
    }

    Py_ssize_t max_tris = num_committed * 2;
    used = (uint8_t *)calloc((size_t)count, 1);
    slot_to_vid = (int32_t *)malloc((size_t)count * sizeof(int32_t));
    tri_buf = (int32_t *)malloc((size_t)max_tris * 3 * sizeof(int32_t));
    fn_buf = (double *)malloc((size_t)max_tris * 3 * sizeof(double));
    if (!used || !slot_to_vid || !tri_buf || !fn_buf) { PyErr_NoMemory(); goto cleanup; }

    const double nx = nrm[0], ny = nrm[1], nz = nrm[2];
    const double max_e2 = max_edge * max_edge;
    const int64_t BIAS = (1LL << 30);
    Py_ssize_t n_tris = 0;
    double total_area2 = 0.0;
    double max_edge_seen2 = 0.0;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t s00 = 0; s00 < count; s00++) {
        if (!committed[s00]) continue;
        int64_t u = (int64_t)iu[s00];
        int64_t v = (int64_t)iv[s00];
        int64_t k10 = ((u + 1LL + BIAS) << 32) | ((v + BIAS) & 0xFFFFFFFFLL);
        int32_t s10 = ht_probe_slot(hk, hs, hmask, k10);
        if (s10 < 0 || s10 >= count || !committed[s10]) continue;
        int64_t k01 = ((u + BIAS) << 32) | ((v + 1LL + BIAS) & 0xFFFFFFFFLL);
        int32_t s01 = ht_probe_slot(hk, hs, hmask, k01);
        if (s01 < 0 || s01 >= count || !committed[s01]) continue;
        int64_t k11 = ((u + 1LL + BIAS) << 32) | ((v + 1LL + BIAS) & 0xFFFFFFFFLL);
        int32_t s11 = ht_probe_slot(hk, hs, hmask, k11);
        if (s11 < 0 || s11 >= count || !committed[s11]) continue;

        int32_t pair[2][3] = {
            {(int32_t)s00, s10, s01},
            {s10, s11, s01}
        };
        for (int t = 0; t < 2; t++) {
            int32_t i0 = pair[t][0], i1 = pair[t][1], i2 = pair[t][2];
            const double *p0 = &fxyz[3 * (Py_ssize_t)i0];
            const double *p1 = &fxyz[3 * (Py_ssize_t)i1];
            const double *p2 = &fxyz[3 * (Py_ssize_t)i2];

            double e01x = p1[0] - p0[0], e01y = p1[1] - p0[1], e01z = p1[2] - p0[2];
            double e12x = p2[0] - p1[0], e12y = p2[1] - p1[1], e12z = p2[2] - p1[2];
            double e20x = p0[0] - p2[0], e20y = p0[1] - p2[1], e20z = p0[2] - p2[2];

            double d01 = e01x * e01x + e01y * e01y + e01z * e01z;
            double d12 = e12x * e12x + e12y * e12y + e12z * e12z;
            double d20 = e20x * e20x + e20y * e20y + e20z * e20z;
            double dm = d01 > d12 ? d01 : d12;
            if (d20 > dm) dm = d20;
            if (dm > max_e2) continue;

            if (fabs(e01x * nx + e01y * ny + e01z * nz) > max_nstep) continue;
            if (fabs(e12x * nx + e12y * ny + e12z * nz) > max_nstep) continue;
            if (fabs(e20x * nx + e20y * ny + e20z * nz) > max_nstep) continue;

            double e02x = p2[0] - p0[0], e02y = p2[1] - p0[1], e02z = p2[2] - p0[2];
            double cx = e01y * e02z - e01z * e02y;
            double cy = e01z * e02x - e01x * e02z;
            double cz = e01x * e02y - e01y * e02x;
            double area2 = sqrt(cx * cx + cy * cy + cz * cz);
            if (area2 <= 1e-12) continue;
            if (cx * nx + cy * ny + cz * nz <= 1e-12) continue;

            tri_buf[3 * n_tris + 0] = i0;
            tri_buf[3 * n_tris + 1] = i1;
            tri_buf[3 * n_tris + 2] = i2;
            double inv_a = 1.0 / area2;
            fn_buf[3 * n_tris + 0] = cx * inv_a;
            fn_buf[3 * n_tris + 1] = cy * inv_a;
            fn_buf[3 * n_tris + 2] = cz * inv_a;
            n_tris++;
            total_area2 += area2;
            if (dm > max_edge_seen2) max_edge_seen2 = dm;
            used[i0] = 1; used[i1] = 1; used[i2] = 1;
        }
    }
    Py_END_ALLOW_THREADS

    if (n_tris == 0) {
        ret = Py_BuildValue("(inOOOOOOOO)", 4, num_committed,
                            Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None);
        goto cleanup;
    }

    Py_ssize_t n_verts = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t s = 0; s < count; s++) {
        if (used[s]) slot_to_vid[s] = (int32_t)(n_verts++);
        else slot_to_vid[s] = -1;
    }
    Py_END_ALLOW_THREADS

    if (n_verts > max_verts) {
        ret = Py_BuildValue("(inOOOOOOOO)", 3, num_committed,
                            Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None, Py_None);
        goto cleanup;
    }

    void *vp = NULL, *cp = NULL, *tp = NULL, *np_ptr = NULL, *op = NULL, *tsp = NULL;
    PyObject *v_arr = alloc_fast_2d(n_verts, 3, 8, g_np_float64, &vp);
    PyObject *c_arr = alloc_fast_2d(n_verts, 3, 1, g_np_uint8, &cp);
    PyObject *t_arr = alloc_fast_2d(n_tris, 3, 4, g_np_int32, &tp);
    PyObject *fn_arr = alloc_fast_2d(n_tris, 3, 8, g_np_float64, &np_ptr);
    PyObject *vo_arr = alloc_fast_1d(n_verts, 4, g_np_int32, &op);
    PyObject *vt_arr = alloc_fast_1d(n_verts, 8, g_np_int64, &tsp);
    if (!v_arr || !c_arr || !t_arr || !fn_arr || !vo_arr || !vt_arr) {
        Py_XDECREF(v_arr); Py_XDECREF(c_arr); Py_XDECREF(t_arr);
        Py_XDECREF(fn_arr); Py_XDECREF(vo_arr); Py_XDECREF(vt_arr);
        goto cleanup;
    }

    double *vo = (double *)vp;
    uint8_t *co = (uint8_t *)cp;
    int32_t *to = (int32_t *)tp;
    double *fno = (double *)np_ptr;
    int32_t *vobs = (int32_t *)op;
    int64_t *vts = (int64_t *)tsp;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t s = 0; s < count; s++) {
        int32_t vid = slot_to_vid[s];
        if (vid < 0) continue;
        Py_ssize_t v = (Py_ssize_t)vid;
        vo[3 * v + 0] = fxyz[3 * s + 0];
        vo[3 * v + 1] = fxyz[3 * s + 1];
        vo[3 * v + 2] = fxyz[3 * s + 2];
        co[3 * v + 0] = frgb[3 * s + 0];
        co[3 * v + 1] = frgb[3 * s + 1];
        co[3 * v + 2] = frgb[3 * s + 2];
        vobs[v] = tobs[s];
        vts[v] = lts[s];
    }
    for (Py_ssize_t t = 0; t < n_tris; t++) {
        to[3 * t + 0] = slot_to_vid[tri_buf[3 * t + 0]];
        to[3 * t + 1] = slot_to_vid[tri_buf[3 * t + 1]];
        to[3 * t + 2] = slot_to_vid[tri_buf[3 * t + 2]];
    }
    memcpy(fno, fn_buf, (size_t)n_tris * 3 * sizeof(double));
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("(inNNNNNNdd)", 0, num_committed,
                        v_arr, c_arr, t_arr, fn_arr, vo_arr, vt_arr,
                        0.5 * total_area2, sqrt(max_edge_seen2));

cleanup:
    free(committed); free(used); free(slot_to_vid); free(tri_buf); free(fn_buf);
    for (int i = 0; i < got; i++) PyBuffer_Release(&b[i]);
    return ret;
}

static PyObject *py_fit_cylinder_axis_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pts_o;
    double z_lo, z_hi, init_x, init_y, r_prior, min_sector_r, inlier_tol, max_shift;
    int n_bins, max_iter;
    if (!PyArg_ParseTuple(
            args, "Odddddididd",
            &pts_o, &z_lo, &z_hi, &init_x, &init_y, &r_prior,
            &n_bins, &min_sector_r, &max_iter, &inlier_tol, &max_shift)) {
        return NULL;
    }

    Py_buffer pts_b = {0};
    int g_p = 0;
    double *bx = NULL, *by = NULL;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(pts_o, &pts_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_p = 1;
    if (pts_b.ndim != 2 || pts_b.shape[1] != 3 || (pts_b.itemsize != 4 && pts_b.itemsize != 8)) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    if (n_bins < 4) n_bins = 4;
    if (n_bins > 128) n_bins = 128;
    if (max_iter < 1) max_iter = 1;
    if (max_iter > 64) max_iter = 64;

    Py_ssize_t n = pts_b.shape[0];
    int is_f32 = (pts_b.itemsize == 4);
    const float *pf32 = is_f32 ? (const float *)pts_b.buf : NULL;
    const double *pf64 = is_f32 ? NULL : (const double *)pts_b.buf;

    bx = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    by = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!bx || !by) { PyErr_NoMemory(); goto cleanup; }

    Py_ssize_t nb = 0;
    double min_x = HUGE_VAL, max_x = -HUGE_VAL;
    double min_y = HUGE_VAL, max_y = -HUGE_VAL;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        double x = is_f32 ? (double)pf32[3 * i + 0] : pf64[3 * i + 0];
        double y = is_f32 ? (double)pf32[3 * i + 1] : pf64[3 * i + 1];
        double z = is_f32 ? (double)pf32[3 * i + 2] : pf64[3 * i + 2];
        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) continue;
        if (z < z_lo || z > z_hi) continue;
        bx[nb] = x;
        by[nb] = y;
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
        nb++;
    }
    Py_END_ALLOW_THREADS

    if (nb < 6) {
        ret = Py_BuildValue("(ddddnnni)", init_x, init_y, r_prior, 0.0, (Py_ssize_t)0, (Py_ssize_t)0, nb, 0);
        goto cleanup;
    }

    double bin_max_r[128];
    double bin_px[128], bin_py[128];
    int bin_has[128];
    double cx = isfinite(init_x) ? init_x : (0.5 * (min_x + max_x));
    double cy = isfinite(init_y) ? init_y : (0.5 * (min_y + max_y));
    double seed_x = 0.5 * (min_x + max_x);
    double seed_y = 0.5 * (min_y + max_y);
    double out_r = r_prior;
    double rmse = 0.0;
    Py_ssize_t inlier_cnt = 0;
    Py_ssize_t occ_bins = 0;
    int accepted = 0;

    Py_BEGIN_ALLOW_THREADS
    for (int b = 0; b < n_bins; b++) { bin_has[b] = 0; bin_max_r[b] = -1.0; }
    const double inv_2pi = (double)n_bins / (2.0 * 3.14159265358979323846);
    const double pi_v = 3.14159265358979323846;
    for (Py_ssize_t i = 0; i < nb; i++) {
        double dx = bx[i] - seed_x;
        double dy = by[i] - seed_y;
        double r = sqrt(dx * dx + dy * dy);
        double ang = atan2(dy, dx);
        int bi = (int)floor((ang + pi_v) * inv_2pi);
        if (bi < 0) bi = 0;
        else if (bi >= n_bins) bi = n_bins - 1;
        if (r > bin_max_r[bi]) {
            bin_max_r[bi] = r;
            bin_px[bi] = bx[i];
            bin_py[bi] = by[i];
            bin_has[bi] = 1;
        }
    }

    double fx_arr[128], fy_arr[128];
    Py_ssize_t n_outer = 0;
    for (int b = 0; b < n_bins; b++) {
        if (bin_has[b]) {
            occ_bins++;
            if (bin_max_r[b] >= min_sector_r) {
                fx_arr[n_outer] = bin_px[b];
                fy_arr[n_outer] = bin_py[b];
                n_outer++;
            }
        }
    }

    const double *ux = (n_outer >= 7) ? fx_arr : bx;
    const double *uy = (n_outer >= 7) ? fy_arr : by;
    Py_ssize_t nu = (n_outer >= 7) ? n_outer : nb;

    for (int it = 0; it < max_iter; it++) {
        Py_ssize_t cnt_in = 0;
        for (Py_ssize_t i = 0; i < nu; i++) {
            double dx = ux[i] - cx, dy = uy[i] - cy;
            double d = sqrt(dx * dx + dy * dy);
            if (d < 1e-6) d = 1e-6;
            if (fabs(d - r_prior) <= inlier_tol) cnt_in++;
        }
        int use_all = (cnt_in < 6);
        double j00 = 1e-9, j01 = 0.0, j11 = 1e-9;
        double g0 = 0.0, g1 = 0.0;
        for (Py_ssize_t i = 0; i < nu; i++) {
            double dx = ux[i] - cx, dy = uy[i] - cy;
            double d = sqrt(dx * dx + dy * dy);
            if (d < 1e-6) d = 1e-6;
            double res = d - r_prior;
            if (!use_all && fabs(res) > inlier_tol) continue;
            double jx = -dx / d;
            double jy = -dy / d;
            j00 += jx * jx;
            j01 += jx * jy;
            j11 += jy * jy;
            g0 += jx * (-res);
            g1 += jy * (-res);
        }
        double det = j00 * j11 - j01 * j01;
        if (!isfinite(det) || fabs(det) < 1e-18) break;
        double sx = (j11 * g0 - j01 * g1) / det;
        double sy = (-j01 * g0 + j00 * g1) / det;
        if (!isfinite(sx) || !isfinite(sy)) break;
        cx += sx;
        cy += sy;
        if (sx * sx + sy * sy < 1e-10) break;
    }

    double shift = sqrt((cx - init_x) * (cx - init_x) + (cy - init_y) * (cy - init_y));
    if (isfinite(cx) && isfinite(cy) && (!isfinite(init_x) || !isfinite(init_y) || shift <= max_shift)) {
        accepted = 1;
    } else {
        cx = init_x;
        cy = init_y;
        accepted = 0;
    }

    double sum_r = 0.0, sum_r2 = 0.0;
    for (Py_ssize_t i = 0; i < nu; i++) {
        double dx = ux[i] - cx, dy = uy[i] - cy;
        double d = sqrt(dx * dx + dy * dy);
        if (fabs(d - r_prior) <= inlier_tol) {
            sum_r += d;
            inlier_cnt++;
        }
    }
    if (inlier_cnt > 0) {
        out_r = sum_r / (double)inlier_cnt;
        for (Py_ssize_t i = 0; i < nu; i++) {
            double dx = ux[i] - cx, dy = uy[i] - cy;
            double d = sqrt(dx * dx + dy * dy);
            if (fabs(d - r_prior) <= inlier_tol) {
                double e = d - out_r;
                sum_r2 += e * e;
            }
        }
        rmse = sqrt(sum_r2 / (double)inlier_cnt);
    } else {
        out_r = r_prior;
        rmse = 0.0;
    }
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("(ddddnnni)", cx, cy, out_r, rmse, inlier_cnt, occ_bins, nb, accepted);

cleanup:
    free(bx); free(by);
    if (g_p) PyBuffer_Release(&pts_b);
    return ret;
}

static int cmp_i64_asc(const void *a, const void *b) {
    int64_t va = *(const int64_t *)a;
    int64_t vb = *(const int64_t *)b;
    return (va > vb) - (va < vb);
}

static void radix_sort_u32(uint32_t *a, uint32_t *tmp, Py_ssize_t n, uint32_t max_val) {
    if (n <= 64) {
        for (Py_ssize_t i = 1; i < n; i++) {
            uint32_t v = a[i];
            Py_ssize_t j = i - 1;
            while (j >= 0 && a[j] > v) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = v;
        }
        return;
    }
    uint32_t counts[2048];
    /* Pass 1: bits 0..10 */
    memset(counts, 0, sizeof(counts));
    for (Py_ssize_t i = 0; i < n; i++) counts[a[i] & 2047u]++;
    uint32_t sum = 0;
    for (int b = 0; b < 2048; b++) { uint32_t c = counts[b]; counts[b] = sum; sum += c; }
    for (Py_ssize_t i = 0; i < n; i++) tmp[counts[a[i] & 2047u]++] = a[i];

    /* Pass 2: bits 11..21 */
    memset(counts, 0, sizeof(counts));
    for (Py_ssize_t i = 0; i < n; i++) counts[(tmp[i] >> 11) & 2047u]++;
    sum = 0;
    for (int b = 0; b < 2048; b++) { uint32_t c = counts[b]; counts[b] = sum; sum += c; }
    for (Py_ssize_t i = 0; i < n; i++) a[counts[(tmp[i] >> 11) & 2047u]++] = tmp[i];

    if (max_val >= (1u << 22)) {
        /* Pass 3: bits 22..31 */
        memset(counts, 0, 1024 * sizeof(uint32_t));
        for (Py_ssize_t i = 0; i < n; i++) counts[a[i] >> 22]++;
        sum = 0;
        for (int b = 0; b < 1024; b++) { uint32_t c = counts[b]; counts[b] = sum; sum += c; }
        for (Py_ssize_t i = 0; i < n; i++) tmp[counts[a[i] >> 22]++] = a[i];
        memcpy(a, tmp, (size_t)n * sizeof(uint32_t));
    }
}

static PyObject *py_encode_coords_rle_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "encode_coords_rle_fast expects (rows, cols, height, width)");
        return NULL;
    }
    Py_ssize_t height = PyLong_AsSsize_t(args[2]);
    if (height == -1 && PyErr_Occurred()) return NULL;
    Py_ssize_t width = PyLong_AsSsize_t(args[3]);
    if (width == -1 && PyErr_Occurred()) return NULL;
    if (height < 0 || width < 0) {
        PyErr_SetString(PyExc_ValueError, "height and width must be nonnegative");
        return NULL;
    }

    Py_buffer rb = {0}, cb = {0};
    int g_r = 0, g_c = 0;
    int64_t *lin = NULL;
    uint32_t *lin32 = NULL;
    uint32_t *tmp32 = NULL;
    int64_t *runs = NULL;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(args[0], &rb, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_r = 1;
    if (PyObject_GetBuffer(args[1], &cb, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_c = 1;

    if (rb.ndim != 1 || cb.ndim != 1 || rb.shape[0] != cb.shape[0] ||
        (rb.itemsize != 2 && rb.itemsize != 4 && rb.itemsize != 8) ||
        (cb.itemsize != 2 && cb.itemsize != 4 && cb.itemsize != 8)) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    Py_ssize_t n = rb.shape[0];
    PyObject *size_list = PyList_New(2);
    if (!size_list) goto cleanup;
    PyList_SET_ITEM(size_list, 0, PyLong_FromSsize_t(height));
    PyList_SET_ITEM(size_list, 1, PyLong_FromSsize_t(width));

    if (n == 0) {
        PyObject *counts_list = PyList_New(0);
        PyObject *d = PyDict_New();
        if (!counts_list || !d) {
            Py_XDECREF(size_list); Py_XDECREF(counts_list); Py_XDECREF(d);
            goto cleanup;
        }
        PyDict_SetItem(d, g_str_size, size_list);
        PyDict_SetItem(d, g_str_counts, counts_list);
        Py_DECREF(size_list); Py_DECREF(counts_list);
        ret = d;
        goto cleanup;
    }

    const int64_t h64 = (int64_t)height, w64 = (int64_t)width;
    const uint64_t total_px = (uint64_t)height * (uint64_t)width;
    const int use_u32 = (total_px > 0ULL && total_px <= 0xFFFFFFFFULL);

    runs = (int64_t *)malloc((size_t)n * 2 * sizeof(int64_t));
    if (use_u32) {
        lin32 = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
        if (n > 64) tmp32 = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
        if (!runs || !lin32 || (n > 64 && !tmp32)) { Py_DECREF(size_list); PyErr_NoMemory(); goto cleanup; }
    } else {
        lin = (int64_t *)malloc((size_t)n * sizeof(int64_t));
        if (!runs || !lin) { Py_DECREF(size_list); PyErr_NoMemory(); goto cleanup; }
    }

    int oob = 0;
    Py_ssize_t n_runs = 0;

    Py_BEGIN_ALLOW_THREADS
    if (use_u32) {
        const uint32_t w32 = (uint32_t)width;
        uint32_t max_v = 0;
        for (Py_ssize_t i = 0; i < n; i++) {
            int64_t r = (rb.itemsize == 2) ? (int64_t)((const int16_t *)rb.buf)[i]
                      : (rb.itemsize == 4) ? (int64_t)((const int32_t *)rb.buf)[i]
                      : ((const int64_t *)rb.buf)[i];
            int64_t c = (cb.itemsize == 2) ? (int64_t)((const int16_t *)cb.buf)[i]
                      : (cb.itemsize == 4) ? (int64_t)((const int32_t *)cb.buf)[i]
                      : ((const int64_t *)cb.buf)[i];
            if (r < 0 || r >= h64 || c < 0 || c >= w64) { oob = 1; break; }
            uint32_t lv = (uint32_t)r * w32 + (uint32_t)c;
            lin32[i] = lv;
            if (lv > max_v) max_v = lv;
        }
        if (!oob) {
            if (n > 1) radix_sort_u32(lin32, tmp32, n, max_v);
            uint32_t start = lin32[0];
            uint32_t prev = lin32[0];
            for (Py_ssize_t i = 1; i < n; i++) {
                uint32_t cur = lin32[i];
                if (cur == prev) continue;
                if (cur == prev + 1u) {
                    prev = cur;
                } else {
                    runs[2 * n_runs + 0] = (int64_t)start;
                    runs[2 * n_runs + 1] = (int64_t)(prev - start + 1u);
                    n_runs++;
                    start = cur;
                    prev = cur;
                }
            }
            runs[2 * n_runs + 0] = (int64_t)start;
            runs[2 * n_runs + 1] = (int64_t)(prev - start + 1u);
            n_runs++;
        }
    } else {
        for (Py_ssize_t i = 0; i < n; i++) {
            int64_t r = (rb.itemsize == 2) ? (int64_t)((const int16_t *)rb.buf)[i]
                      : (rb.itemsize == 4) ? (int64_t)((const int32_t *)rb.buf)[i]
                      : ((const int64_t *)rb.buf)[i];
            int64_t c = (cb.itemsize == 2) ? (int64_t)((const int16_t *)cb.buf)[i]
                      : (cb.itemsize == 4) ? (int64_t)((const int32_t *)cb.buf)[i]
                      : ((const int64_t *)cb.buf)[i];
            if (r < 0 || r >= h64 || c < 0 || c >= w64) { oob = 1; break; }
            lin[i] = r * w64 + c;
        }
        if (!oob) {
            if (n > 1) qsort(lin, (size_t)n, sizeof(int64_t), cmp_i64_asc);
            int64_t start = lin[0];
            int64_t prev = lin[0];
            for (Py_ssize_t i = 1; i < n; i++) {
                int64_t cur = lin[i];
                if (cur == prev) continue;
                if (cur == prev + 1) {
                    prev = cur;
                } else {
                    runs[2 * n_runs + 0] = start;
                    runs[2 * n_runs + 1] = prev - start + 1;
                    n_runs++;
                    start = cur;
                    prev = cur;
                }
            }
            runs[2 * n_runs + 0] = start;
            runs[2 * n_runs + 1] = prev - start + 1;
            n_runs++;
        }
    }
    Py_END_ALLOW_THREADS

    if (oob) {
        Py_DECREF(size_list);
        PyErr_SetString(PyExc_ValueError, "row/col coordinates out of image bounds");
        goto cleanup;
    }

    PyObject *counts_list = PyList_New(2 * n_runs);
    PyObject *d = PyDict_New();
    if (!counts_list || !d) {
        Py_XDECREF(size_list); Py_XDECREF(counts_list); Py_XDECREF(d);
        goto cleanup;
    }
    for (Py_ssize_t i = 0; i < 2 * n_runs; i++) {
        PyList_SET_ITEM(counts_list, i, PyLong_FromLongLong(runs[i]));
    }
    PyDict_SetItem(d, g_str_size, size_list);
    PyDict_SetItem(d, g_str_counts, counts_list);
    Py_DECREF(size_list); Py_DECREF(counts_list);
    ret = d;

cleanup:
    free(lin); free(lin32); free(tmp32); free(runs);
    if (g_r) PyBuffer_Release(&rb);
    if (g_c) PyBuffer_Release(&cb);
    return ret;
}

static PyObject *py_estm_quantize_vertices_fast(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "estm_quantize_vertices_fast expects (verts, min_quantization_m)");
        return NULL;
    }
    double min_q = PyFloat_AsDouble(args[1]);
    if (min_q == -1.0 && PyErr_Occurred()) return NULL;

    Py_buffer vb = {0};
    int g_v = 0;
    PyObject *ret = NULL;

    if (PyObject_GetBuffer(args[0], &vb, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_v = 1;
    if (vb.ndim != 2 || vb.shape[1] != 3 || (vb.itemsize != 4 && vb.itemsize != 8)) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    Py_ssize_t n = vb.shape[0];
    if (n == 0) {
        PyObject *q_arr = alloc_fast_2d(0, 3, 2, g_np_uint16, NULL);
        if (q_arr) ret = Py_BuildValue("(N(ddd)d)", q_arr, 0.0, 0.0, 0.0, min_q);
        goto cleanup;
    }

    int is_f32 = (vb.itemsize == 4);
    const float *vf32 = is_f32 ? (const float *)vb.buf : NULL;
    const double *vf64 = is_f32 ? NULL : (const double *)vb.buf;

    void *qp = NULL;
    PyObject *q_arr = alloc_fast_2d(n, 3, 2, g_np_uint16, &qp);
    if (!q_arr) goto cleanup;
    uint16_t *qo = (uint16_t *)qp;

    double min_x = HUGE_VAL, min_y = HUGE_VAL, min_z = HUGE_VAL;
    double max_x = -HUGE_VAL, max_y = -HUGE_VAL, max_z = -HUGE_VAL;
    double scale = min_q;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        double x = is_f32 ? (double)vf32[3 * i + 0] : vf64[3 * i + 0];
        double y = is_f32 ? (double)vf32[3 * i + 1] : vf64[3 * i + 1];
        double z = is_f32 ? (double)vf32[3 * i + 2] : vf64[3 * i + 2];
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
        if (z < min_z) min_z = z;
        if (z > max_z) max_z = z;
    }
    double sx = max_x - min_x, sy = max_y - min_y, sz = max_z - min_z;
    double span = sx > sy ? sx : sy;
    if (sz > span) span = sz;
    double s_span = span / 65535.0;
    scale = (s_span > min_q) ? s_span : min_q;
    double inv_s = 1.0 / scale;

    for (Py_ssize_t i = 0; i < n; i++) {
        double x = is_f32 ? (double)vf32[3 * i + 0] : vf64[3 * i + 0];
        double y = is_f32 ? (double)vf32[3 * i + 1] : vf64[3 * i + 1];
        double z = is_f32 ? (double)vf32[3 * i + 2] : vf64[3 * i + 2];
        double qx = rint((x - min_x) * inv_s);
        double qy = rint((y - min_y) * inv_s);
        double qz = rint((z - min_z) * inv_s);
        qo[3 * i + 0] = (uint16_t)(qx < 0.0 ? 0.0 : (qx > 65535.0 ? 65535.0 : qx));
        qo[3 * i + 1] = (uint16_t)(qy < 0.0 ? 0.0 : (qy > 65535.0 ? 65535.0 : qy));
        qo[3 * i + 2] = (uint16_t)(qz < 0.0 ? 0.0 : (qz > 65535.0 ? 65535.0 : qz));
    }
    Py_END_ALLOW_THREADS

    ret = Py_BuildValue("(N(ddd)d)", q_arr, min_x, min_y, min_z, scale);

cleanup:
    if (g_v) PyBuffer_Release(&vb);
    return ret;
}

/* ============================================================================
 * 19. v2.7.0 Kernel: Bounded 3D Voxel Slot Grid Single-Pass Delta Step
 * ============================================================================ */

typedef struct {
    Py_buffer pts_b;
    Py_buffer rgb_b;
    Py_buffer skip_b;
    int has_pts;
    int has_rgb;
    int has_skip;
    Py_ssize_t n;
    const float *pts;
    const uint8_t *rgb;
    const uint8_t *skip;
} VoxelCloudView;

static PyObject *py_voxel_slot_grid_step_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *clouds_o, *ws_min_o, *ws_max_o, *q_min_o, *q_span_o;
    PyObject *grid_o, *free_o, *alive_o, *gidx_o, *pos_o, *rgb_o, *lo_o;
    PyObject *sq_o, *sc_o, *ns_o, *stamp_o;
    double inv_v;
    int nx, ny, nz, free_top_in, step_stamp, retire_unobs;

    if (!PyArg_ParseTuple(
            args, "OOOdiiiOOOOiOOOOOOOOOii",
            &clouds_o, &ws_min_o, &ws_max_o, &inv_v, &nx, &ny, &nz,
            &q_min_o, &q_span_o,
            &grid_o, &free_o, &free_top_in,
            &alive_o, &gidx_o, &pos_o, &rgb_o, &lo_o,
            &sq_o, &sc_o, &ns_o, &stamp_o,
            &step_stamp, &retire_unobs)) {
        return NULL;
    }

    PyObject *clouds_seq = PySequence_Fast(clouds_o, "clouds must be a sequence");
    if (!clouds_seq) return NULL;
    Py_ssize_t n_clouds = PySequence_Fast_GET_SIZE(clouds_seq);

    Py_buffer b[15];
    int has_b[15] = {0};
    PyObject *objs[15] = {
        ws_min_o, ws_max_o, q_min_o, q_span_o,
        grid_o, free_o, alive_o, gidx_o, pos_o, rgb_o, lo_o,
        sq_o, sc_o, ns_o, stamp_o
    };
    int flags[15] = {
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
        PyBUF_WRITABLE | PyBUF_ND | PyBUF_C_CONTIGUOUS,
    };

    VoxelCloudView *cviews = NULL;
    uint16_t *upserted_buf = NULL;
    uint16_t *deleted_buf = NULL;
    PyObject *ret = NULL;

    for (int i = 0; i < 15; i++) {
        if (PyObject_GetBuffer(objs[i], &b[i], flags[i]) < 0) goto cleanup;
        has_b[i] = 1;
    }

    Py_ssize_t max_slots = b[5].shape[0];
    Py_ssize_t grid_len = (Py_ssize_t)nx * (Py_ssize_t)ny * (Py_ssize_t)nz;
    if (b[0].itemsize != 4 || b[1].itemsize != 4 || b[2].itemsize != 8 || b[3].itemsize != 8 ||
        b[4].ndim != 1 || b[4].itemsize != 4 || b[4].shape[0] < grid_len ||
        b[5].ndim != 1 || b[5].itemsize != 4 || max_slots <= 0 || max_slots > 65535 ||
        b[6].shape[0] != max_slots || b[6].itemsize != 1 ||
        b[7].shape[0] != max_slots || b[7].itemsize != 4 ||
        b[8].ndim != 2 || b[8].shape[0] != max_slots || b[8].shape[1] != 3 || b[8].itemsize != 4 ||
        b[9].ndim != 2 || b[9].shape[0] != max_slots || b[9].shape[1] != 3 || b[9].itemsize != 4 ||
        b[10].shape[0] != max_slots || b[10].itemsize != 2 ||
        b[11].ndim != 2 || b[11].shape[0] != max_slots || b[11].shape[1] != 3 || b[11].itemsize != 4 ||
        b[12].shape[0] != max_slots || b[12].itemsize != 2 ||
        b[13].shape[0] != max_slots || b[13].itemsize != 1 ||
        b[14].shape[0] != max_slots || b[14].itemsize != 4 ||
        free_top_in < 0 || free_top_in > (int)max_slots) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    if (n_clouds > 0) {
        cviews = (VoxelCloudView *)calloc((size_t)n_clouds, sizeof(VoxelCloudView));
        if (!cviews) { PyErr_NoMemory(); goto cleanup; }
        for (Py_ssize_t c = 0; c < n_clouds; c++) {
            PyObject *item = PySequence_Fast_GET_ITEM(clouds_seq, c);
            if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 3) {
                ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
            }
            PyObject *po = PyTuple_GET_ITEM(item, 0);
            PyObject *ro = PyTuple_GET_ITEM(item, 1);
            PyObject *so = PyTuple_GET_ITEM(item, 2);
            if (PyObject_GetBuffer(po, &cviews[c].pts_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
            cviews[c].has_pts = 1;
            if (PyObject_GetBuffer(ro, &cviews[c].rgb_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
            cviews[c].has_rgb = 1;
            if (cviews[c].pts_b.ndim != 2 || cviews[c].pts_b.shape[1] != 3 || cviews[c].pts_b.itemsize != 4 ||
                cviews[c].rgb_b.ndim != 2 || cviews[c].rgb_b.shape[1] != 3 || cviews[c].rgb_b.itemsize != 1 ||
                cviews[c].pts_b.shape[0] != cviews[c].rgb_b.shape[0]) {
                ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
            }
            cviews[c].n = cviews[c].pts_b.shape[0];
            cviews[c].pts = (const float *)cviews[c].pts_b.buf;
            cviews[c].rgb = (const uint8_t *)cviews[c].rgb_b.buf;
            if (so != Py_None) {
                if (PyObject_GetBuffer(so, &cviews[c].skip_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
                cviews[c].has_skip = 1;
                if (cviews[c].skip_b.ndim != 1 || cviews[c].skip_b.itemsize != 1 ||
                    cviews[c].skip_b.shape[0] != cviews[c].n) {
                    ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
                }
                cviews[c].skip = (const uint8_t *)cviews[c].skip_b.buf;
            }
        }
    }

    upserted_buf = (uint16_t *)malloc((size_t)max_slots * 5 * sizeof(uint16_t));
    deleted_buf = (uint16_t *)malloc((size_t)max_slots * sizeof(uint16_t));
    if (!upserted_buf || !deleted_buf) { PyErr_NoMemory(); goto cleanup; }

    const float *wmin = (const float *)b[0].buf;
    const float *wmax = (const float *)b[1].buf;
    const double *qmin = (const double *)b[2].buf;
    const double *qspan = (const double *)b[3].buf;
    int32_t *grid_to_slot = (int32_t *)b[4].buf;
    int32_t *free_slots = (int32_t *)b[5].buf;
    uint8_t *slot_alive = (uint8_t *)b[6].buf;
    int32_t *slot_grid_idx = (int32_t *)b[7].buf;
    float *slot_pos = (float *)b[8].buf;
    float *slot_rgb = (float *)b[9].buf;
    int16_t *slot_lo = (int16_t *)b[10].buf;
    int32_t *sent_qpos = (int32_t *)b[11].buf;
    uint16_t *sent_rgb565 = (uint16_t *)b[12].buf;
    uint8_t *never_sent = (uint8_t *)b[13].buf;
    int32_t *seen_stamp = (int32_t *)b[14].buf;

    const float wmin0 = wmin[0], wmin1 = wmin[1], wmin2 = wmin[2];
    const float wmax0 = wmax[0], wmax1 = wmax[1], wmax2 = wmax[2];
    const double qmin0 = qmin[0], qmin1 = qmin[1], qmin2 = qmin[2];
    const double qspan0 = qspan[0], qspan1 = qspan[1], qspan2 = qspan[2];
    const int32_t stamp = (int32_t)step_stamp;
    int free_top = free_top_in;
    Py_ssize_t n_upserted = 0;
    Py_ssize_t n_deleted = 0;
    Py_ssize_t n_alive = 0;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t c = 0; c < n_clouds; c++) {
        const Py_ssize_t cn = cviews[c].n;
        const float *pts = cviews[c].pts;
        const uint8_t *rgb = cviews[c].rgb;
        const uint8_t *skip = cviews[c].skip;
        for (Py_ssize_t i = 0; i < cn; i++) {
            if (skip && skip[i]) continue;
            float px = pts[3 * i + 0];
            float py = pts[3 * i + 1];
            float pz = pts[3 * i + 2];
            if (!(px >= wmin0 && px <= wmax0 && py >= wmin1 && py <= wmax1 && pz >= wmin2 && pz <= wmax2)) {
                continue;
            }
            int ix = (int)floor(((double)px - (double)wmin0) * inv_v);
            int iy = (int)floor(((double)py - (double)wmin1) * inv_v);
            int iz = (int)floor(((double)pz - (double)wmin2) * inv_v);
            if (ix < 0) ix = 0; else if (ix >= nx) ix = nx - 1;
            if (iy < 0) iy = 0; else if (iy >= ny) iy = ny - 1;
            if (iz < 0) iz = 0; else if (iz >= nz) iz = nz - 1;
            int32_t gidx = (int32_t)(((Py_ssize_t)ix * (Py_ssize_t)ny + (Py_ssize_t)iy) * (Py_ssize_t)nz + (Py_ssize_t)iz);
            int32_t slot = grid_to_slot[gidx];

            uint8_t r8 = rgb[3 * i + 0];
            uint8_t g8 = rgb[3 * i + 1];
            uint8_t b8 = rgb[3 * i + 2];

            if (slot >= 0) {
                if (seen_stamp[slot] == stamp) continue;
                seen_stamp[slot] = stamp;
                slot_pos[3 * slot + 0] = px;
                slot_pos[3 * slot + 1] = py;
                slot_pos[3 * slot + 2] = pz;
                slot_rgb[3 * slot + 0] = (float)r8;
                slot_rgb[3 * slot + 1] = (float)g8;
                slot_rgb[3 * slot + 2] = (float)b8;
                int lo = (int)slot_lo[slot] + 2;
                slot_lo[slot] = (int16_t)(lo > 7 ? 7 : lo);
            } else {
                if (free_top <= 0) continue;
                slot = free_slots[--free_top];
                grid_to_slot[gidx] = slot;
                slot_alive[slot] = 1;
                slot_grid_idx[slot] = gidx;
                slot_pos[3 * slot + 0] = px;
                slot_pos[3 * slot + 1] = py;
                slot_pos[3 * slot + 2] = pz;
                slot_rgb[3 * slot + 0] = (float)r8;
                slot_rgb[3 * slot + 1] = (float)g8;
                slot_rgb[3 * slot + 2] = (float)b8;
                slot_lo[slot] = 6;
                never_sent[slot] = 1;
                seen_stamp[slot] = stamp;
            }

            double qxd = rint((((double)px - qmin0) / qspan0) * 65535.0);
            double qyd = rint((((double)py - qmin1) / qspan1) * 65535.0);
            double qzd = rint((((double)pz - qmin2) / qspan2) * 65535.0);
            int32_t qx = (int32_t)(qxd < 0.0 ? 0.0 : (qxd > 65535.0 ? 65535.0 : qxd));
            int32_t qy = (int32_t)(qyd < 0.0 ? 0.0 : (qyd > 65535.0 ? 65535.0 : qyd));
            int32_t qz = (int32_t)(qzd < 0.0 ? 0.0 : (qzd > 65535.0 ? 65535.0 : qzd));
            uint16_t c565 = (uint16_t)((((uint16_t)r8 >> 3) << 11) | (((uint16_t)g8 >> 2) << 5) | ((uint16_t)b8 >> 3));

            if (never_sent[slot] ||
                qx != sent_qpos[3 * slot + 0] ||
                qy != sent_qpos[3 * slot + 1] ||
                qz != sent_qpos[3 * slot + 2] ||
                c565 != sent_rgb565[slot]) {
                sent_qpos[3 * slot + 0] = qx;
                sent_qpos[3 * slot + 1] = qy;
                sent_qpos[3 * slot + 2] = qz;
                sent_rgb565[slot] = c565;
                never_sent[slot] = 0;
                uint16_t *rec = upserted_buf + 5 * n_upserted;
                rec[0] = (uint16_t)slot;
                rec[1] = (uint16_t)qx;
                rec[2] = (uint16_t)qy;
                rec[3] = (uint16_t)qz;
                rec[4] = c565;
                n_upserted++;
            }
        }
    }

    for (Py_ssize_t s = 0; s < max_slots; s++) {
        if (!slot_alive[s]) continue;
        if (retire_unobs && seen_stamp[s] != stamp) {
            int32_t gi = slot_grid_idx[s];
            if (gi >= 0 && gi < grid_len) grid_to_slot[gi] = -1;
            slot_alive[s] = 0;
            slot_grid_idx[s] = -1;
            slot_lo[s] = 0;
            never_sent[s] = 1;
            free_slots[free_top++] = (int32_t)s;
            deleted_buf[n_deleted++] = (uint16_t)s;
        } else {
            n_alive++;
        }
    }
    Py_END_ALLOW_THREADS

    void *up_p = NULL, *del_p = NULL;
    PyObject *up_arr = alloc_fast_2d(n_upserted, 5, 2, g_np_uint16, &up_p);
    PyObject *del_arr = alloc_fast_1d(n_deleted, 2, g_np_uint16, &del_p);
    if (!up_arr || !del_arr) {
        Py_XDECREF(up_arr); Py_XDECREF(del_arr);
        goto cleanup;
    }
    if (n_upserted > 0) memcpy(up_p, upserted_buf, (size_t)n_upserted * 5 * sizeof(uint16_t));
    if (n_deleted > 0) memcpy(del_p, deleted_buf, (size_t)n_deleted * sizeof(uint16_t));
    ret = Py_BuildValue("(inNN)", free_top, n_alive, up_arr, del_arr);

cleanup:
    free(upserted_buf);
    free(deleted_buf);
    if (cviews) {
        for (Py_ssize_t c = 0; c < n_clouds; c++) {
            if (cviews[c].has_pts) PyBuffer_Release(&cviews[c].pts_b);
            if (cviews[c].has_rgb) PyBuffer_Release(&cviews[c].rgb_b);
            if (cviews[c].has_skip) PyBuffer_Release(&cviews[c].skip_b);
        }
        free(cviews);
    }
    for (int i = 0; i < 15; i++) {
        if (has_b[i]) PyBuffer_Release(&b[i]);
    }
    Py_DECREF(clouds_seq);
    return ret;
}

/* ============================================================================
 * 20. v2.8.0 Kernels: Quickselect Quantiles, 2D Height-Grid Clustering,
 *     & Azimuthal Outer-Rim Annulus Extraction
 * ============================================================================ */

static void quickselect_f64(double *a, Py_ssize_t lo, Py_ssize_t hi, Py_ssize_t k) {
    while (lo < hi) {
        Py_ssize_t mid = lo + ((hi - lo) >> 1);
        double v0 = a[lo], v1 = a[mid], v2 = a[hi];
        double pivot;
        if ((v0 <= v1 && v1 <= v2) || (v2 <= v1 && v1 <= v0)) pivot = v1;
        else if ((v1 <= v0 && v0 <= v2) || (v2 <= v0 && v0 <= v1)) pivot = v0;
        else pivot = v2;
        Py_ssize_t lt = lo, i = lo, gt = hi;
        while (i <= gt) {
            double ai = a[i];
            if (ai < pivot) {
                a[i] = a[lt];
                a[lt] = ai;
                lt++;
                i++;
            } else if (ai > pivot) {
                a[i] = a[gt];
                a[gt] = ai;
                gt--;
            } else {
                i++;
            }
        }
        if (k < lt) {
            hi = lt - 1;
        } else if (k > gt) {
            lo = gt + 1;
        } else {
            return;
        }
    }
}

static double percentile_linear_f64(double *a, Py_ssize_t n, double p) {
    if (n <= 0) return 0.0;
    if (n == 1) return a[0];
    double pos = (p * 0.01) * (double)(n - 1);
    Py_ssize_t lo = (Py_ssize_t)floor(pos);
    if (lo < 0) lo = 0;
    if (lo >= n) lo = n - 1;
    Py_ssize_t hi = (lo + 1 < n) ? (lo + 1) : (n - 1);
    double frac = pos - (double)lo;
    quickselect_f64(a, 0, n - 1, lo);
    double v_lo = a[lo];
    if (hi == lo || frac <= 0.0) return v_lo;
    double mn = a[lo + 1];
    for (Py_ssize_t j = lo + 2; j < n; j++) {
        if (a[j] < mn) mn = a[j];
    }
    return v_lo + (mn - v_lo) * frac;
}

static void percentile_linear_pair_f64(
    double *a, Py_ssize_t n, double p1, double p2, double *out1, double *out2)
{
    if (n <= 0) { *out1 = 0.0; *out2 = 0.0; return; }
    if (n == 1) { *out1 = a[0]; *out2 = a[0]; return; }
    double pos1 = (p1 * 0.01) * (double)(n - 1);
    Py_ssize_t lo1 = (Py_ssize_t)floor(pos1);
    if (lo1 < 0) lo1 = 0;
    if (lo1 >= n) lo1 = n - 1;
    Py_ssize_t hi1 = (lo1 + 1 < n) ? (lo1 + 1) : (n - 1);
    double frac1 = pos1 - (double)lo1;
    quickselect_f64(a, 0, n - 1, lo1);
    double v_lo1 = a[lo1];
    double v_hi1 = v_lo1;
    if (hi1 > lo1 && frac1 > 0.0) {
        double mn = a[lo1 + 1];
        for (Py_ssize_t j = lo1 + 2; j < n; j++) {
            if (a[j] < mn) mn = a[j];
        }
        v_hi1 = mn;
    }
    *out1 = v_lo1 + (v_hi1 - v_lo1) * frac1;

    double pos2 = (p2 * 0.01) * (double)(n - 1);
    Py_ssize_t lo2 = (Py_ssize_t)floor(pos2);
    if (lo2 < 0) lo2 = 0;
    if (lo2 >= n) lo2 = n - 1;
    Py_ssize_t hi2 = (lo2 + 1 < n) ? (lo2 + 1) : (n - 1);
    double frac2 = pos2 - (double)lo2;
    Py_ssize_t start2 = (lo2 >= lo1) ? lo1 : 0;
    quickselect_f64(a, start2, n - 1, lo2);
    double v_lo2 = a[lo2];
    double v_hi2 = v_lo2;
    if (hi2 > lo2 && frac2 > 0.0) {
        double mn = a[lo2 + 1];
        for (Py_ssize_t j = lo2 + 2; j < n; j++) {
            if (a[j] < mn) mn = a[j];
        }
        v_hi2 = mn;
    }
    *out2 = v_lo2 + (v_hi2 - v_lo2) * frac2;
}

typedef struct {
    Py_ssize_t start;
    Py_ssize_t count;
    int layer_id;
    double hp5, hp97, xlo, ylo, xhi, yhi, ztop;
} GridClusterRec;

static PyObject *py_cluster_height_grid_2d_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pts_o, *h_o, *exc_o;
    double x_lo, x_hi, y_lo, y_hi, h_lo, h_hi;
    double ox, oy, sx, sy, cell_m;
    Py_ssize_t min_pts;
    double mid_lo, mid_hi, hi_min, l0_max_h, l0_max_ay, l0_max_x;
    if (!PyArg_ParseTuple(
            args, "OOOdddddddddddndddddd",
            &pts_o, &h_o, &exc_o,
            &x_lo, &x_hi, &y_lo, &y_hi, &h_lo, &h_hi,
            &ox, &oy, &sx, &sy, &cell_m, &min_pts,
            &mid_lo, &mid_hi, &hi_min, &l0_max_h, &l0_max_ay, &l0_max_x)) {
        return NULL;
    }

    Py_buffer pts_b = {0}, h_b = {0}, exc_b = {0};
    int g_p = 0, g_h = 0, g_e = 0;
    const int has_exc = (exc_o != NULL && exc_o != Py_None);
    PyObject *ret = NULL;

    GridClusterRec stack_clusters[64];
    GridClusterRec *clusters = stack_clusters;
    Py_ssize_t cl_cap = 64;

    if (PyObject_GetBuffer(pts_o, &pts_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_p = 1;
    if (PyObject_GetBuffer(h_o, &h_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_h = 1;
    if (has_exc) {
        if (PyObject_GetBuffer(exc_o, &exc_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
        g_e = 1;
    }

    if (pts_b.ndim != 2 || pts_b.shape[1] != 3 || (pts_b.itemsize != 4 && pts_b.itemsize != 8) ||
        h_b.ndim != 1 || h_b.shape[0] != pts_b.shape[0] || (h_b.itemsize != 4 && h_b.itemsize != 8) ||
        (has_exc && (exc_b.ndim != 1 || exc_b.shape[0] != pts_b.shape[0] || exc_b.itemsize != 1)) ||
        pts_b.shape[0] > INT32_MAX || cell_m <= 0.0 || !isfinite(cell_m)) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    Py_ssize_t n = pts_b.shape[0];
    if (n < min_pts || min_pts < 1) {
        ret = PyList_New(0);
        goto cleanup;
    }

    int nx_c = (int)ceil(sx / cell_m) + 2;
    int ny_c = (int)ceil(sy / cell_m) + 2;
    if (nx_c <= 0 || ny_c <= 0 || (int64_t)nx_c * (int64_t)ny_c > 4000000LL) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }
    size_t gsize = (size_t)nx_c * (size_t)ny_c;

    const int pts_f32 = (pts_b.itemsize == 4);
    const int h_f32 = (h_b.itemsize == 4);
    const float *pf32 = pts_f32 ? (const float *)pts_b.buf : NULL;
    const double *pf64 = pts_f32 ? NULL : (const double *)pts_b.buf;
    const float *hf32 = h_f32 ? (const float *)h_b.buf : NULL;
    const double *hf64 = h_f32 ? NULL : (const double *)h_b.buf;
    const uint8_t *exc = has_exc ? (const uint8_t *)exc_b.buf : NULL;

    ThreadScratch *sc = get_scratch();
    size_t max_cc_alloc = gsize + 1;
    if (!sc ||
        ensure_p_idx(sc, (size_t)n * 3) < 0 ||
        ensure_marks(sc, (size_t)n + gsize * 5) < 0 ||
        ensure_grid(sc, gsize * 3 + 1) < 0 ||
        ensure_idx_tmp(sc, max_cc_alloc * 2) < 0 ||
        ensure_dbl_tmp(sc, (size_t)n) < 0) {
        PyErr_NoMemory(); goto cleanup;
    }

    int32_t *fg_idx = sc->p_idx;
    int32_t *fg_cell = fg_idx + n;
    int32_t *all_cl_idx = fg_cell + n;
    uint8_t *fg_layer = sc->marks;
    uint8_t *mid_occ = fg_layer + n;
    uint8_t *hi_occ = mid_occ + gsize;
    uint8_t *tall_seed = hi_occ + gsize;
    uint8_t *tall_grid = tall_seed + gsize;
    uint8_t *occ_l = tall_grid + gsize;
    int32_t *lbl_2d = sc->grid;
    int32_t *bfs_q = lbl_2d + gsize;
    int32_t *cc_to_q = bfs_q + gsize;
    Py_ssize_t *cc_counts = sc->idx_tmp;
    Py_ssize_t *q_write = cc_counts + max_cc_alloc;
    double *qbuf = sc->dbl_tmp;

    Py_ssize_t n_fg = 0;
    Py_ssize_t n_clusters = 0;
    Py_ssize_t total_out_pts = 0;
    int oom = 0;

    Py_BEGIN_ALLOW_THREADS
    memset(mid_occ, 0, gsize * 4); /* mid_occ, hi_occ, tall_seed, tall_grid */
    /* Pass 1: Filter foreground points, compute 2D grid cells, populate mid/hi bitmaps */
    for (Py_ssize_t i = 0; i < n; i++) {
        if (exc && exc[i]) continue;
        double hv = h_f32 ? (double)hf32[i] : hf64[i];
        if (!(hv >= h_lo && hv <= h_hi)) continue;
        double px = pts_f32 ? (double)pf32[3 * i + 0] : pf64[3 * i + 0];
        if (!(px >= x_lo && px <= x_hi)) continue;
        double py = pts_f32 ? (double)pf32[3 * i + 1] : pf64[3 * i + 1];
        if (!(py >= y_lo && py <= y_hi)) continue;
        double pz = pts_f32 ? (double)pf32[3 * i + 2] : pf64[3 * i + 2];
        if (!isfinite(pz)) continue;

        int gx = (int)floor((px - ox) / cell_m);
        int gy = (int)floor((py - oy) / cell_m);
        if (gx < 0) gx = 0; else if (gx >= nx_c) gx = nx_c - 1;
        if (gy < 0) gy = 0; else if (gy >= ny_c) gy = ny_c - 1;
        int32_t cell = (int32_t)(gx * ny_c + gy);

        fg_idx[n_fg] = (int32_t)i;
        fg_cell[n_fg] = cell;
        n_fg++;

        if (hv >= mid_lo && hv <= mid_hi) mid_occ[cell] = 1;
        if (hv > hi_min) hi_occ[cell] = 1;
    }

    if (n_fg >= min_pts) {
        /* Pass 2: Two 3x3 dilations on tiny L1-resident occupancy grid */
        for (int r = 0; r < nx_c; r++) {
            int r0 = r > 0 ? r - 1 : 0;
            int r1 = r + 1 < nx_c ? r + 1 : nx_c - 1;
            for (int c = 0; c < ny_c; c++) {
                int idx = r * ny_c + c;
                if (!mid_occ[idx]) continue;
                int c0 = c > 0 ? c - 1 : 0;
                int c1 = c + 1 < ny_c ? c + 1 : ny_c - 1;
                uint8_t hit = 0;
                for (int rr = r0; rr <= r1 && !hit; rr++) {
                    const uint8_t *row_hi = hi_occ + (size_t)rr * (size_t)ny_c;
                    for (int cc = c0; cc <= c1; cc++) {
                        if (row_hi[cc]) { hit = 1; break; }
                    }
                }
                tall_seed[idx] = hit;
            }
        }
        for (int r = 0; r < nx_c; r++) {
            int r0 = r > 0 ? r - 1 : 0;
            int r1 = r + 1 < nx_c ? r + 1 : nx_c - 1;
            for (int c = 0; c < ny_c; c++) {
                int c0 = c > 0 ? c - 1 : 0;
                int c1 = c + 1 < ny_c ? c + 1 : ny_c - 1;
                uint8_t hit = 0;
                for (int rr = r0; rr <= r1 && !hit; rr++) {
                    const uint8_t *row_ts = tall_seed + (size_t)rr * (size_t)ny_c;
                    for (int cc = c0; cc <= c1; cc++) {
                        if (row_ts[cc]) { hit = 1; break; }
                    }
                }
                tall_grid[r * ny_c + c] = hit;
            }
        }

        /* Pass 3: Assign layer 0 (lower/graspable) vs layer 1 (tall/equipment) */
        Py_ssize_t layer_counts[2] = {0, 0};
        for (Py_ssize_t k = 0; k < n_fg; k++) {
            int32_t gi = fg_idx[k];
            double hv = h_f32 ? (double)hf32[gi] : hf64[gi];
            double px = pts_f32 ? (double)pf32[3 * gi + 0] : pf64[3 * gi + 0];
            double py = pts_f32 ? (double)pf32[3 * gi + 1] : pf64[3 * gi + 1];
            uint8_t l1 = (tall_grid[fg_cell[k]] || hv > l0_max_h || fabs(py) > l0_max_ay || px > l0_max_x) ? 1 : 0;
            fg_layer[k] = l1;
            layer_counts[l1]++;
        }

        /* Pass 4: 8-connected BFS clustering per layer + linear quickselect quantiles */
        for (int layer_id = 0; layer_id < 2 && !oom; layer_id++) {
            if (layer_counts[layer_id] < min_pts) continue;
            memset(occ_l, 0, gsize);
            memset(lbl_2d, 0, gsize * sizeof(int32_t));
            for (Py_ssize_t k = 0; k < n_fg; k++) {
                if (fg_layer[k] == (uint8_t)layer_id) occ_l[fg_cell[k]] = 1;
            }
            int32_t n_cc = 0;
            for (int r = 0; r < nx_c; r++) {
                for (int c = 0; c < ny_c; c++) {
                    int32_t start_cell = (int32_t)(r * ny_c + c);
                    if (!occ_l[start_cell] || lbl_2d[start_cell] != 0) continue;
                    int32_t cid = ++n_cc;
                    lbl_2d[start_cell] = cid;
                    Py_ssize_t qh = 0, qt = 0;
                    bfs_q[qt++] = start_cell;
                    while (qh < qt) {
                        int32_t cur = bfs_q[qh++];
                        int cr = cur / ny_c;
                        int cc_col = cur - cr * ny_c;
                        int rr0 = cr > 0 ? cr - 1 : 0;
                        int rr1 = cr + 1 < nx_c ? cr + 1 : nx_c - 1;
                        int cc0 = cc_col > 0 ? cc_col - 1 : 0;
                        int cc1 = cc_col + 1 < ny_c ? cc_col + 1 : ny_c - 1;
                        for (int rr = rr0; rr <= rr1; rr++) {
                            int base_r = rr * ny_c;
                            for (int cc2 = cc0; cc2 <= cc1; cc2++) {
                                int32_t ni = (int32_t)(base_r + cc2);
                                if (occ_l[ni] && lbl_2d[ni] == 0) {
                                    lbl_2d[ni] = cid;
                                    bfs_q[qt++] = ni;
                                }
                            }
                        }
                    }
                }
            }
            if (n_cc <= 0) continue;
            memset(cc_counts, 0, (size_t)(n_cc + 1) * sizeof(Py_ssize_t));
            for (Py_ssize_t k = 0; k < n_fg; k++) {
                if (fg_layer[k] == (uint8_t)layer_id) {
                    cc_counts[lbl_2d[fg_cell[k]]]++;
                }
            }
            int32_t n_qual = 0;
            for (int32_t cid = 1; cid <= n_cc; cid++) {
                if (cc_counts[cid] >= min_pts) {
                    cc_to_q[cid] = n_qual;
                    q_write[n_qual] = total_out_pts;
                    if (n_clusters + n_qual >= cl_cap) {
                        Py_ssize_t new_cap = cl_cap * 2;
                        GridClusterRec *nc = (clusters == stack_clusters)
                            ? (GridClusterRec *)malloc((size_t)new_cap * sizeof(GridClusterRec))
                            : (GridClusterRec *)realloc(clusters, (size_t)new_cap * sizeof(GridClusterRec));
                        if (!nc) { oom = 1; break; }
                        if (clusters == stack_clusters) {
                            memcpy(nc, stack_clusters, (size_t)(n_clusters + n_qual) * sizeof(GridClusterRec));
                        }
                        clusters = nc;
                        cl_cap = new_cap;
                    }
                    GridClusterRec *rec = &clusters[n_clusters + n_qual];
                    rec->start = total_out_pts;
                    rec->count = cc_counts[cid];
                    rec->layer_id = layer_id;
                    total_out_pts += cc_counts[cid];
                    n_qual++;
                } else {
                    cc_to_q[cid] = -1;
                }
            }
            if (oom || n_qual == 0) continue;
            for (Py_ssize_t k = 0; k < n_fg; k++) {
                if (fg_layer[k] != (uint8_t)layer_id) continue;
                int32_t qid = cc_to_q[lbl_2d[fg_cell[k]]];
                if (qid >= 0) {
                    all_cl_idx[q_write[qid]++] = fg_idx[k];
                }
            }
            for (int32_t qid = 0; qid < n_qual; qid++) {
                GridClusterRec *rec = &clusters[n_clusters + qid];
                const int32_t *cidx = all_cl_idx + rec->start;
                Py_ssize_t cnt = rec->count;
                for (Py_ssize_t j = 0; j < cnt; j++) {
                    int32_t gi = cidx[j];
                    qbuf[j] = h_f32 ? (double)hf32[gi] : hf64[gi];
                }
                percentile_linear_pair_f64(qbuf, cnt, 5.0, 97.0, &rec->hp5, &rec->hp97);

                for (Py_ssize_t j = 0; j < cnt; j++) {
                    int32_t gi = cidx[j];
                    qbuf[j] = pts_f32 ? (double)pf32[3 * gi + 0] : pf64[3 * gi + 0];
                }
                percentile_linear_pair_f64(qbuf, cnt, 3.0, 97.0, &rec->xlo, &rec->xhi);

                for (Py_ssize_t j = 0; j < cnt; j++) {
                    int32_t gi = cidx[j];
                    qbuf[j] = pts_f32 ? (double)pf32[3 * gi + 1] : pf64[3 * gi + 1];
                }
                percentile_linear_pair_f64(qbuf, cnt, 3.0, 97.0, &rec->ylo, &rec->yhi);

                for (Py_ssize_t j = 0; j < cnt; j++) {
                    int32_t gi = cidx[j];
                    qbuf[j] = pts_f32 ? (double)pf32[3 * gi + 2] : pf64[3 * gi + 2];
                }
                rec->ztop = percentile_linear_f64(qbuf, cnt, 98.5);
            }
            n_clusters += n_qual;
        }
    }
    Py_END_ALLOW_THREADS

    if (oom) { PyErr_NoMemory(); goto cleanup; }

    PyObject *res_list = PyList_New(n_clusters);
    if (!res_list) goto cleanup;
    for (Py_ssize_t i = 0; i < n_clusters; i++) {
        const GridClusterRec *rec = &clusters[i];
        void *ip = NULL;
        PyObject *idx_arr = alloc_fast_1d(rec->count, 4, g_np_int32, &ip);
        if (!idx_arr) { Py_DECREF(res_list); goto cleanup; }
        memcpy(ip, all_cl_idx + rec->start, (size_t)rec->count * sizeof(int32_t));
        PyObject *tup = Py_BuildValue(
            "(Niddddddd)",
            idx_arr,
            rec->layer_id,
            rec->hp5,
            rec->hp97,
            rec->xlo,
            rec->ylo,
            rec->xhi,
            rec->yhi,
            rec->ztop);
        if (!tup) { Py_DECREF(res_list); goto cleanup; }
        PyList_SET_ITEM(res_list, i, tup);
    }
    ret = res_list;

cleanup:
    if (clusters != stack_clusters) free(clusters);
    if (g_p) PyBuffer_Release(&pts_b);
    if (g_h) PyBuffer_Release(&h_b);
    if (g_e) PyBuffer_Release(&exc_b);
    return ret;
}

static PyObject *py_extract_rim_annulus_fast(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pts_o;
    double band_below, min_rad, z_blend;
    int n_bins, min_rim;
    if (!PyArg_ParseTuple(
            args, "Odidid",
            &pts_o, &band_below, &n_bins, &min_rad, &min_rim, &z_blend)) {
        return NULL;
    }

    Py_buffer pts_b = {0};
    int g_p = 0;
    PyObject *ret = NULL;
    double *qbuf = NULL;
    Py_ssize_t *upper_idx = NULL;
    double *best_r = NULL;
    Py_ssize_t *best_i = NULL;

    if (PyObject_GetBuffer(pts_o, &pts_b, PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0) goto cleanup;
    g_p = 1;

    if (pts_b.ndim != 2 || pts_b.shape[1] != 3 ||
        (pts_b.itemsize != 4 && pts_b.itemsize != 8) ||
        n_bins < 1 || n_bins > 4096) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    Py_ssize_t n = pts_b.shape[0];
    const int is_f32 = (pts_b.itemsize == 4);
    const float *pf32 = is_f32 ? (const float *)pts_b.buf : NULL;
    const double *pf64 = is_f32 ? NULL : (const double *)pts_b.buf;

    if (n < 12) {
        double *out = NULL;
        PyObject *arr = alloc_fast_2d(n, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out);
        if (!arr) goto cleanup;
        if (n > 0) {
            if (is_f32) {
                for (Py_ssize_t i = 0; i < 3 * n; i++) out[i] = (double)pf32[i];
            } else {
                memcpy(out, pf64, (size_t)n * 3 * sizeof(double));
            }
        }
        ret = arr;
        goto cleanup;
    }

    qbuf = (double *)malloc((size_t)n * sizeof(double));
    upper_idx = (Py_ssize_t *)malloc((size_t)n * sizeof(Py_ssize_t));
    best_r = (double *)malloc((size_t)n_bins * sizeof(double));
    best_i = (Py_ssize_t *)malloc((size_t)n_bins * sizeof(Py_ssize_t));
    if (!qbuf || !upper_idx || !best_r || !best_i) {
        PyErr_NoMemory(); goto cleanup;
    }

    int non_finite = 0;
    Py_ssize_t n_upper = 0;
    Py_ssize_t n_rim = 0;
    double z_top = 0.0;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        double x = is_f32 ? (double)pf32[3 * i + 0] : pf64[3 * i + 0];
        double y = is_f32 ? (double)pf32[3 * i + 1] : pf64[3 * i + 1];
        double z = is_f32 ? (double)pf32[3 * i + 2] : pf64[3 * i + 2];
        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) { non_finite = 1; break; }
        qbuf[i] = z;
    }
    if (!non_finite) {
        z_top = percentile_linear_f64(qbuf, n, 98.5);
        double z_cut = z_top - band_below;
        for (Py_ssize_t i = 0; i < n; i++) {
            double z = is_f32 ? (double)pf32[3 * i + 2] : pf64[3 * i + 2];
            if (z >= z_cut) upper_idx[n_upper++] = i;
        }
        if (n_upper < 12) {
            for (Py_ssize_t i = 0; i < n; i++) upper_idx[i] = i;
            n_upper = n;
        }
        for (Py_ssize_t k = 0; k < n_upper; k++) {
            Py_ssize_t gi = upper_idx[k];
            qbuf[k] = is_f32 ? (double)pf32[3 * gi + 0] : pf64[3 * gi + 0];
        }
        double xlo, xhi, ylo, yhi;
        percentile_linear_pair_f64(qbuf, n_upper, 3.0, 97.0, &xlo, &xhi);
        for (Py_ssize_t k = 0; k < n_upper; k++) {
            Py_ssize_t gi = upper_idx[k];
            qbuf[k] = is_f32 ? (double)pf32[3 * gi + 1] : pf64[3 * gi + 1];
        }
        percentile_linear_pair_f64(qbuf, n_upper, 3.0, 97.0, &ylo, &yhi);

        double cx = 0.5 * (xlo + xhi);
        double cy = 0.5 * (ylo + yhi);
        for (int b = 0; b < n_bins; b++) {
            best_r[b] = -1.0;
            best_i[b] = -1;
        }
        const double pi_v = 3.14159265358979323846;
        const double inv_2pi = 1.0 / (2.0 * pi_v);
        const double nb_d = (double)n_bins;
        for (Py_ssize_t k = 0; k < n_upper; k++) {
            Py_ssize_t gi = upper_idx[k];
            double dx = (is_f32 ? (double)pf32[3 * gi + 0] : pf64[3 * gi + 0]) - cx;
            double dy = (is_f32 ? (double)pf32[3 * gi + 1] : pf64[3 * gi + 1]) - cy;
            double ang = atan2(dy, dx);
            double rad = sqrt(dx * dx + dy * dy);
            int b = (int)floor((ang + pi_v) * inv_2pi * nb_d);
            if (b < 0) b = 0;
            else if (b >= n_bins) b = n_bins - 1;
            if (rad > best_r[b]) {
                best_r[b] = rad;
                best_i[b] = gi;
            }
        }
        for (int b = 0; b < n_bins; b++) {
            if (best_i[b] >= 0 && best_r[b] >= min_rad) n_rim++;
        }
    }
    Py_END_ALLOW_THREADS

    if (non_finite) {
        ret = Py_NotImplemented; Py_INCREF(ret); goto cleanup;
    }

    if (n_rim < (Py_ssize_t)min_rim) {
        double *out = NULL;
        PyObject *arr = alloc_fast_2d(n_upper, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out);
        if (!arr) goto cleanup;
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t k = 0; k < n_upper; k++) {
            Py_ssize_t gi = upper_idx[k];
            out[3 * k + 0] = is_f32 ? (double)pf32[3 * gi + 0] : pf64[3 * gi + 0];
            out[3 * k + 1] = is_f32 ? (double)pf32[3 * gi + 1] : pf64[3 * gi + 1];
            out[3 * k + 2] = is_f32 ? (double)pf32[3 * gi + 2] : pf64[3 * gi + 2];
        }
        Py_END_ALLOW_THREADS
        ret = arr;
    } else {
        double *out = NULL;
        PyObject *arr = alloc_fast_2d(n_rim, 3, (Py_ssize_t)sizeof(double), g_np_float64, (void **)&out);
        if (!arr) goto cleanup;
        const double z_rem = 1.0 - z_blend;
        Py_BEGIN_ALLOW_THREADS
        Py_ssize_t w = 0;
        for (int b = 0; b < n_bins; b++) {
            Py_ssize_t gi = best_i[b];
            if (gi >= 0 && best_r[b] >= min_rad) {
                double z = is_f32 ? (double)pf32[3 * gi + 2] : pf64[3 * gi + 2];
                out[3 * w + 0] = is_f32 ? (double)pf32[3 * gi + 0] : pf64[3 * gi + 0];
                out[3 * w + 1] = is_f32 ? (double)pf32[3 * gi + 1] : pf64[3 * gi + 1];
                out[3 * w + 2] = z_blend * z_top + z_rem * z;
                w++;
            }
        }
        Py_END_ALLOW_THREADS
        ret = arr;
    }

cleanup:
    free(qbuf); free(upper_idx); free(best_r); free(best_i);
    if (g_p) PyBuffer_Release(&pts_b);
    return ret;
}

/* ============================================================================
 * Module Definition & Init
 * ============================================================================ */

static PyMethodDef NativeMethods[] = {
    {"simd_isa", py_simd_isa, METH_NOARGS, "Return runtime-dispatched SIMD ISA level ('avx512', 'avx2', or 'scalar')."},
    {"voxel_centroids", (PyCFunction)(void (*)(void))py_voxel_centroids, METH_FASTCALL, "Compute exact lexicographic per-voxel centroids (and optional RGB means)."},
    {"surfel_general_update", py_surfel_general_update, METH_VARARGS, "Execute columnar surfel map delta update (with optional RGB sums)."},
    {"snapshot_select", py_snapshot_select, METH_VARARGS, "Select top-k surfels matching owner priority (with optional fused RGB)."},
    {"snapshot_uniform_fast", (PyCFunction)(void (*)(void))py_snapshot_uniform_fast, METH_FASTCALL, "Fast uniform snapshot copy and divide (with optional fused RGB)."},
    {"select_existing_points_shadow_fast", (PyCFunction)(void (*)(void))py_select_existing_points_shadow, METH_FASTCALL, "Coarse 12mm shadow snapshot selector."},
    {"rle_mask_box_and_centroid", (PyCFunction)(void (*)(void))py_rle_mask_box_and_centroid, METH_FASTCALL, "Exact O(runs) RLE bounding box and centroid."},
    {"rle_area", py_rle_area, METH_O, "Exact O(runs) RLE area."},
    {"rle_intersection_area", (PyCFunction)(void (*)(void))py_rle_intersection_area, METH_FASTCALL, "Exact O(R_A+R_B) RLE intersection area."},
    {"rle_iou", (PyCFunction)(void (*)(void))py_rle_iou, METH_FASTCALL, "Exact O(R_A+R_B) RLE IoU."},
    {"rle_set_op", (PyCFunction)(void (*)(void))py_rle_set_op, METH_FASTCALL, "Exact O(R_A+R_B) RLE set intersection/union/difference."},
    {"encode_mask_rle_bytes", py_encode_mask_rle_bytes, METH_VARARGS, "Encode contiguous boolean mask bytes to RLE."},
    {"decode_mask_rle_into", (PyCFunction)(void (*)(void))py_decode_mask_rle_into, METH_FASTCALL, "Decode row-major RLE dict to boolean ndarray."},
    {"circle_consensus_2d", (PyCFunction)(void (*)(void))py_circle_consensus_2d, METH_FASTCALL, "Deterministic 3-point 2D circle consensus."},
    {"fit_circular_aperture_fast", (PyCFunction)(void (*)(void))py_fit_circular_aperture_fast, METH_FASTCALL, "End-to-end deterministic 3D circular aperture fitting."},
    {"aperture_sector_rim_filter", py_aperture_sector_rim_filter, METH_VARARGS, "Sector outer-rim quantile selection."},
    {"is_valid_se3", py_is_valid_se3, METH_VARARGS, "Validate 4x4 SE(3) matrix."},
    {"build_se3_landmark_jacobians_fast", py_build_se3_landmark_jacobians_fast, METH_VARARGS, "Populate whitened SE(3)-landmark Jacobians."},
    {"evaluate_schur_gate_fast", (PyCFunction)(void (*)(void))py_evaluate_schur_gate_fast, METH_FASTCALL, "Schur marginal pose observability gate kernel."},
    {"nn_min_dists_3d", (PyCFunction)(void (*)(void))py_nn_min_dists_3d, METH_FASTCALL, "3D nearest-neighbor Euclidean distances via median-split KD-tree."},
    {"transfer_rgb_knn_3d", (PyCFunction)(void (*)(void))py_transfer_rgb_knn_3d, METH_FASTCALL, "3D kNN inverse-distance RGB transfer via median-split KD-tree."},
    {"backproject_masked_depth", py_backproject_masked_depth, METH_VARARGS, "Backproject masked strided depth to 3D points."},
    {"splat_synthetic_rgbd", py_splat_synthetic_rgbd, METH_VARARGS, "2x2 z-buffered synthetic depth/mask/RGB splat."},
    {"planar_mesh_finish_fast", py_planar_mesh_finish_fast, METH_VARARGS, "Single-pass quad meshing, gating, and vertex compaction."},
    {"planar_tile_quads_fast", py_planar_tile_quads_fast, METH_VARARGS, "Extract gated global-slot triangles for a single spatial tile."},
    {"planar_compact_triangles_fast", py_planar_compact_triangles_fast, METH_VARARGS, "Compact concatenated global-slot triangles into dense mesh arrays."},
    {"planar_cell_aggregate_fast", py_planar_cell_aggregate_fast, METH_VARARGS, "Hash-aggregate weighted planar cell centroids and RGB."},
    {"planar_project_and_aggregate_fast", py_planar_project_and_aggregate_fast, METH_VARARGS, "Single-pass plane gating, physics weighting, optional mask, and L2-grid cell aggregation."},
    {"planar_free_space_carve_fast", py_planar_free_space_carve_fast, METH_VARARGS, "Single-pass camera free-space carving with 3x3 neighborhood verification."},
    {"planar_ht_lookup_fast", py_planar_ht_lookup_fast, METH_VARARGS, "Batch open-addressing int64 slot lookup."},
    {"planar_ht_rebuild_fast", py_planar_ht_rebuild_fast, METH_VARARGS, "Rebuild open-addressing int64 slot hash table from active slots."},
    {"planar_window_accumulate_fast", py_planar_window_accumulate_fast, METH_VARARGS, "Fused SoA window accumulation for persistent planar cells."},
    {"planar_compact_obs_fast", py_planar_compact_obs_fast, METH_VARARGS, "Fused C11 compaction of an aged-out observation into archived cell baselines."},
    {"planar_mesh_snapshot_ht_fast", py_planar_mesh_snapshot_ht_fast, METH_VARARGS, "Single-pass hash-table planar mesh extraction and compaction."},
    {"fit_plane_ransac_fast", (PyCFunction)(void (*)(void))py_fit_plane_ransac_fast, METH_FASTCALL, "Deterministic seeded RANSAC plane consensus with exact early-exit pruning."},
    {"fit_cylinder_axis_fast", py_fit_cylinder_axis_fast, METH_VARARGS, "Azimuthal outer-envelope IRLS Gauss-Newton cylinder axis refinement."},
    {"encode_coords_rle_fast", (PyCFunction)(void (*)(void))py_encode_coords_rle_fast, METH_FASTCALL, "Sparse O(K) radix-sorted coordinate-to-RLE mask encoder."},
    {"estm_quantize_vertices_fast", (PyCFunction)(void (*)(void))py_estm_quantize_vertices_fast, METH_FASTCALL, "Single-pass ESTM uint16 vertex bounding-box quantizer."},
    {"voxel_slot_grid_step_fast", py_voxel_slot_grid_step_fast, METH_VARARGS, "Single-pass bounded 3D voxel slot dedup, quantization, and delta emission."},
    {"cluster_height_grid_2d_fast", py_cluster_height_grid_2d_fast, METH_VARARGS, "Single-pass 2D height-grid foreground filtering, air-gap decoupling, 8-connected BFS, and linear quantiles."},
    {"extract_rim_annulus_fast", py_extract_rim_annulus_fast, METH_VARARGS, "Single-pass azimuthal outer-perimeter rim annulus extraction with linear quantiles."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nativemodule = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "EvidenceScene C11/SIMD deterministic computational kernels.",
    -1,
    NativeMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__native(void) {
#if ES_HAVE_X86_RUNTIME_SIMD
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl")) {
        g_simd_level = 2;
    } else if (__builtin_cpu_supports("avx2")) {
        g_simd_level = 1;
    } else {
        g_simd_level = 0;
    }
#endif
    if (PyType_Ready(&PooledBufferType) < 0) return NULL;
    PyObject *m = PyModule_Create(&nativemodule);
    if (!m) return NULL;

    PyObject *np_mod = PyImport_ImportModule("numpy");
    if (!np_mod) { Py_DECREF(m); return NULL; }

    g_np_empty = PyObject_GetAttrString(np_mod, "empty");
    g_np_ndarray = PyObject_GetAttrString(np_mod, "ndarray");
    g_np_int64 = PyObject_GetAttrString(np_mod, "int64");
    g_np_int32 = PyObject_GetAttrString(np_mod, "int32");
    g_np_uint16 = PyObject_GetAttrString(np_mod, "uint16");
    g_np_uint8 = PyObject_GetAttrString(np_mod, "uint8");
    g_np_float64 = PyObject_GetAttrString(np_mod, "float64");
    g_np_float32 = PyObject_GetAttrString(np_mod, "float32");
    g_np_intp = PyObject_GetAttrString(np_mod, "intp");
    g_np_bool = PyObject_GetAttrString(np_mod, "bool_");
    Py_DECREF(np_mod);

    if (!g_np_empty || !g_np_ndarray || !g_np_int64 || !g_np_int32 || !g_np_uint16 || !g_np_uint8 ||
        !g_np_float64 || !g_np_float32 || !g_np_intp || !g_np_bool) {
        Py_XDECREF(g_np_empty); Py_XDECREF(g_np_ndarray); Py_XDECREF(g_np_int64);
        Py_XDECREF(g_np_int32); Py_XDECREF(g_np_uint16); Py_XDECREF(g_np_uint8);
        Py_XDECREF(g_np_float64); Py_XDECREF(g_np_float32);
        Py_XDECREF(g_np_intp); Py_XDECREF(g_np_bool);
        Py_DECREF(m);
        return NULL;
    }

    g_str_size = PyUnicode_InternFromString("size");
    g_str_counts = PyUnicode_InternFromString("counts");
    g_str_owner_priority = PyUnicode_InternFromString("owner_priority");
    g_str_coarse_shadow = PyUnicode_InternFromString("coarse_shadow");
    g_str_within_budget = PyUnicode_InternFromString("within_budget");
    g_str_insufficient_coarse = PyUnicode_InternFromString("insufficient_coarse_support");

    if (!g_str_size || !g_str_counts || !g_str_owner_priority ||
        !g_str_coarse_shadow || !g_str_within_budget || !g_str_insufficient_coarse) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
