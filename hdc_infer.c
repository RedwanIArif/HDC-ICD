/* hdc_infer.c -- see hdc_infer.h.
 *
 * Feature definitions mirror features.py (columns 0-7 and 23-29, the FFT-free set), in float32:
 *   0 log_mad  1 zcr  2 line_length  3 hjorth_mobility  4 hjorth_complexity  5 kurtosis  6 skewness
 *   7 crest_factor  8 ac_peak  9 ac_lag_hz  10 n_peaks  11 rate_hz  12 rr_mean  13 rr_std  14 rr_cv
 * Medians are exact order statistics found by bisection over the float bit patterns (no sorted copy,
 * so no second 5 KB buffer). Peak picking reproduces scipy.signal.find_peaks(height=, distance=37).
 * Autocorrelation is computed directly over lags 25..249 (identical in exact arithmetic to the FFT form).
 */
#include <math.h>
#include <string.h>
#include "hdc_infer.h"

#define FS        250.0f
#define AC_LO     25
#define AC_HI     250            /* exclusive */
#define PK_DIST   37             /* int(0.15 * 250) */
#define MAXPK     (HDC_T / 2)

#if HDC_PROFILE && !defined(HDC_NOW)
#define HDC_NOW() (*(volatile uint32_t *)0xE0001004UL)   /* Cortex-M DWT->CYCCNT (enabled by dwt_init) */
#endif

#if HDC_NFEAT != 15
#error "hdc_infer.c computes the 15 FFT-free features; regenerate the model with that feature set"
#endif

/* ------------------------------------------------------------------ exact medians without a copy */
static inline uint32_t fkey(float f) { uint32_t u; memcpy(&u, &f, 4); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
static inline float funkey(uint32_t k) { uint32_t u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k; float f; memcpy(&f, &u, 4); return f; }

/* k-th smallest (0-based) of v_i = |x_i - c| when absmode, else x_i */
static float kth(const float *x, int n, int k, int absmode, float c)
{
    uint32_t lo = 0u, hi = 0xffffffffu;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            float v = absmode ? fabsf(x[i] - c) : x[i];
            cnt += (fkey(v) <= mid);
        }
        if (cnt >= k + 1) hi = mid; else lo = mid + 1u;
    }
    return funkey(lo);
}

static float median(const float *x, int n, int absmode, float c)
{
    if (n & 1) return kth(x, n, n / 2, absmode, c);
    return 0.5f * (kth(x, n, n / 2 - 1, absmode, c) + kth(x, n, n / 2, absmode, c));
}

/* ------------------------------------------------------------------ features */
static uint16_t g_pk[MAXPK];
static uint16_t g_ord[MAXPK];
static uint8_t  g_keep[MAXPK];

static int find_peaks_abs(const float *xs, int n, float thr)
{
    /* local maxima of |xs| (scipy _local_maxima_1d), keeping those with height >= thr */
    int m = 0, i = 1, imax = n - 1;
    while (i < imax) {
        float ai = fabsf(xs[i]);
        if (fabsf(xs[i - 1]) < ai) {
            int ia = i + 1;
            while (ia < imax && fabsf(xs[ia]) == ai) ia++;
            if (fabsf(xs[ia]) < ai) {
                int mid = (i + ia - 1) / 2;
                if (ai >= thr && m < MAXPK) g_pk[m++] = (uint16_t)mid;
                i = ia;
            }
        }
        i++;
    }
    /* scipy _select_by_peak_distance: highest first, suppress neighbours closer than PK_DIST */
    for (int a = 0; a < m; a++) { g_ord[a] = (uint16_t)a; g_keep[a] = 1; }
    for (int a = 1; a < m; a++) {                        /* stable insertion sort by height, ascending */
        uint16_t t = g_ord[a]; float ht = fabsf(xs[g_pk[t]]); int b = a - 1;
        while (b >= 0 && fabsf(xs[g_pk[g_ord[b]]]) > ht) { g_ord[b + 1] = g_ord[b]; b--; }
        g_ord[b + 1] = t;
    }
    for (int r = m - 1; r >= 0; r--) {
        int j = g_ord[r];
        if (!g_keep[j]) continue;
        for (int k = j - 1; k >= 0 && (int)g_pk[j] - (int)g_pk[k] < PK_DIST; k--) g_keep[k] = 0;
        for (int k = j + 1; k < m && (int)g_pk[k] - (int)g_pk[j] < PK_DIST; k++) g_keep[k] = 0;
    }
    int K = 0;
    for (int a = 0; a < m; a++) if (g_keep[a]) g_pk[K++] = g_pk[a];   /* compact, still in position order */
    return K;
}

static void extract_features(float *x, float f[HDC_NFEAT])
{
    const int T = HDC_T;
    float med = median(x, T, 0, 0.0f);
    float mad = median(x, T, 1, med);
    if (mad < 1e-6f) mad = 1e-6f;
    for (int i = 0; i < T; i++) x[i] = (x[i] - med) / mad;          /* xs, in place */
    const float *xs = x;

    /* means for the two-pass standard deviations */
    float m0 = 0, m1 = 0, m2 = 0;
    for (int i = 0; i < T; i++) m0 += xs[i];
    m0 /= T;
    for (int i = 0; i < T - 1; i++) m1 += xs[i + 1] - xs[i];
    m1 /= (T - 1);
    for (int i = 0; i < T - 2; i++) m2 += (xs[i + 2] - xs[i + 1]) - (xs[i + 1] - xs[i]);
    m2 /= (T - 2);

    float v0 = 0, v1 = 0, v2 = 0, ll = 0, amax = 0;
    int zc = 0;
    for (int i = 0; i < T; i++) {
        float d = xs[i] - m0; v0 += d * d;
        float a = fabsf(xs[i]); if (a > amax) amax = a;
        if (i > 0) zc += (signbit(xs[i]) != signbit(xs[i - 1]));
    }
    for (int i = 0; i < T - 1; i++) { float d1 = xs[i + 1] - xs[i]; ll += fabsf(d1); float e = d1 - m1; v1 += e * e; }
    for (int i = 0; i < T - 2; i++) {
        float d2 = (xs[i + 2] - xs[i + 1]) - (xs[i + 1] - xs[i]); float e = d2 - m2; v2 += e * e;
    }
    float s0 = sqrtf(v0 / T) + 1e-9f, s1 = sqrtf(v1 / (T - 1)) + 1e-9f, s2 = sqrtf(v2 / (T - 2)) + 1e-9f;

    float k4 = 0, k3 = 0;
    for (int i = 0; i < T; i++) { float z = (xs[i] - m0) / s0; float z2 = z * z; k3 += z2 * z; k4 += z2 * z2; }

    f[0] = log10f(mad + 1e-9f);
    f[1] = (float)zc / (T - 1);
    f[2] = ll / (T - 1);
    f[3] = s1 / s0;
    f[4] = (s2 / s1) / (s1 / s0);
    f[5] = k4 / T;
    f[6] = k3 / T;
    f[7] = amax / s0;

    /* autocorrelation of xz = xs - mean, normalised by lag 0, peak over lags 25..249 */
    float r0 = 0;
    for (int i = 0; i < T; i++) { float d = xs[i] - m0; r0 += d * d; }
    float best = -INFINITY; int best_lag = AC_LO;
    for (int lag = AC_LO; lag < AC_HI; lag++) {
        float r = 0;
        for (int i = 0; i + lag < T; i++) r += (xs[i] - m0) * (xs[i + lag] - m0);
        float ac = r / (r0 + 1e-12f);
        if (ac > best) { best = ac; best_lag = lag; }                  /* first maximum, like np.argmax */
    }
    f[8] = best;
    f[9] = FS / (float)best_lag;

    /* beat intervals */
    int K = find_peaks_abs(xs, T, 0.4f * amax);
    f[10] = (float)K;
    f[11] = (float)K / ((float)T / FS);
    if (K >= 3) {
        float mean = 0, var = 0;
        for (int a = 0; a < K - 1; a++) mean += (float)(g_pk[a + 1] - g_pk[a]) / FS;
        mean /= (K - 1);
        for (int a = 0; a < K - 1; a++) { float d = (float)(g_pk[a + 1] - g_pk[a]) / FS - mean; var += d * d; }
        float sd = sqrtf(var / (K - 1));
        f[12] = mean; f[13] = sd; f[14] = sd / (mean + 1e-9f);
    } else {
        f[12] = f[13] = f[14] = 0.0f;
    }
    for (int i = 0; i < HDC_NFEAT; i++) if (!isfinite(f[i])) f[i] = 0.0f;
}

/* ------------------------------------------------------------------ encoding and classification */
static inline uint32_t level_word(int k, int w)
{
#if HDC_REBUILD_LEVELS
    uint32_t gt = 0u, eq = 0xffffffffu;                  /* bit-sliced (switch level > k) */
    for (int b = HDC_LEVEL_BITS - 1; b >= 0; b--) {
        uint32_t sb = hdc_switch_planes[b][w];
        if ((k >> b) & 1) eq &= sb; else { gt |= eq & sb; eq &= ~sb; }
    }
    uint32_t m = ~gt;                                     /* level k has reached the switch: take B */
    return (hdc_level_a[w] & ~m) | (hdc_level_b[w] & m);
#else
    return hdc_levels[k][w];
#endif
}

static uint32_t g_query[HDC_WORDS];

void hdc_predict(float *x, hdc_result_t *out)
{
#if HDC_PROFILE
    uint32_t t0 = HDC_NOW();
#endif
    float f[HDC_NFEAT];
    extract_features(x, f);
#if HDC_PROFILE
    uint32_t t1 = HDC_NOW();
#endif

    int lev[HDC_NFEAT];
    for (int i = 0; i < HDC_NFEAT; i++) {
        float v = (f[i] - hdc_scale_lo[i]) / (hdc_scale_hi[i] - hdc_scale_lo[i]);
        if (!(v > 0.0f)) v = 0.0f;                       /* also maps NaN to 0, like a clip */
        if (v > 1.0f) v = 1.0f;
        lev[i] = (int)(v * (float)(HDC_NLEVELS - 1));
        out->levels[i] = (uint8_t)lev[i];
        out->feat[i] = f[i];
    }

    /* bundle: per element, count how many of the bound vectors (key XOR level) have bit 1 (= -1);
       the sum of +-1 is <= 0 exactly when that count >= ceil(n/2). Bit-sliced 5-bit counters. */
    const int thr = (HDC_NFEAT + 1) / 2;
    int h0 = 0, h1 = 0;
    for (int w = 0; w < HDC_WORDS; w++) {
        uint32_t c[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < HDC_NFEAT; i++) {
            uint32_t carry = hdc_keys[i][w] ^ level_word(lev[i], w);
            for (int t = 0; t < 5 && carry; t++) { uint32_t nc = c[t] & carry; c[t] ^= carry; carry = nc; }
        }
        uint32_t gt = 0u, eq = 0xffffffffu;               /* count >= thr, bit-sliced */
        for (int b = 4; b >= 0; b--) {
            if ((thr >> b) & 1) eq &= c[b]; else { gt |= eq & c[b]; eq &= ~c[b]; }
        }
        uint32_t q = gt | eq;
        g_query[w] = q;
#if !HDC_PROFILE
        h0 += __builtin_popcount(q ^ hdc_proto[0][w]);
        h1 += __builtin_popcount(q ^ hdc_proto[1][w]);
#endif
    }
#if HDC_PROFILE                                          /* same result, classification timed on its own */
    uint32_t t2 = HDC_NOW();
    for (int w = 0; w < HDC_WORDS; w++) {
        h0 += __builtin_popcount(g_query[w] ^ hdc_proto[0][w]);
        h1 += __builtin_popcount(g_query[w] ^ hdc_proto[1][w]);
    }
#endif
    out->h0 = (int16_t)h0;
    out->h1 = (int16_t)h1;
    out->pred = (uint8_t)((h0 - h1) > hdc_threshold);
#if HDC_PROFILE
    uint32_t t3 = HDC_NOW();
    out->cyc_feat = t1 - t0;
    out->cyc_enc = t2 - t1;
    out->cyc_cls = t3 - t2;
#endif
}
