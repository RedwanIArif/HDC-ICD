/* host_test.c -- desktop check of hdc_infer.c (used by c_check.py; not needed on the board).
 * Reads N x 1250 float32 segments from argv[1], writes N records (pred, h0, h1, 15 levels) to argv[2].
 */
#include <stdio.h>
#include <stdlib.h>
#include "hdc_infer.h"

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s segments.bin out.bin\n", argv[0]); return 1; }
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    if (!fi || !fo) { perror("open"); return 1; }
    static float seg[HDC_T];
    while (fread(seg, sizeof(float), HDC_T, fi) == HDC_T) {
        hdc_result_t r;
        hdc_predict(seg, &r);
        int16_t rec[3 + HDC_NFEAT] = {r.pred, r.h0, r.h1};
        for (int i = 0; i < HDC_NFEAT; i++) rec[3 + i] = r.levels[i];
        fwrite(rec, sizeof(int16_t), 3 + HDC_NFEAT, fo);
    }
    fclose(fi); fclose(fo);
    return 0;
}
