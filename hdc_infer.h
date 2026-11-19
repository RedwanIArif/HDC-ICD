/* hdc_infer.h -- HDC ventricular-arrhythmia detector: features, binary encoding and classification.
 * Portable C99, single precision only (no double arithmetic), no dynamic memory.
 * Static RAM: one 1,250-float segment buffer is provided by the caller and overwritten in place.
 */
#ifndef HDC_INFER_H
#define HDC_INFER_H

#include <stdint.h>
#include "hdc_model.h"

#define HDC_T   1250          /* samples per segment (5 s at 250 Hz) */

/* 1: also report how many CPU cycles each stage takes (feature extraction, encoding, classification).
   Predictions are identical. Set it here or with -DHDC_PROFILE=1; use host_eval.py --profile with it. */
#ifndef HDC_PROFILE
#define HDC_PROFILE 0
#endif

typedef struct {
    uint8_t pred;                  /* 1 = VA, 0 = non-VA */
    int16_t h0;                    /* Hamming distance to the non-VA prototype */
    int16_t h1;                    /* Hamming distance to the VA prototype */
    uint8_t levels[HDC_NFEAT];     /* level index (0..63) of each feature, for checking against Python */
    float   feat[HDC_NFEAT];       /* the 15 raw features, for debugging */
#if HDC_PROFILE
    uint32_t cyc_feat;             /* cycles: feature extraction */
    uint32_t cyc_enc;              /* cycles: scaling, level lookup, binding and bundling */
    uint32_t cyc_cls;              /* cycles: Hamming distances and threshold */
#endif
} hdc_result_t;

/* Classify one segment. `x` holds HDC_T raw samples and is overwritten (scaled in place). */
void hdc_predict(float *x, hdc_result_t *out);

#endif
