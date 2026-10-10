#include "yolo.h"
#include <stdlib.h>

typedef struct {
    float s;
    int i;
} scorepair;

static int cmpDesc(const void* a, const void* b) {
    const scorepair* pa = (const scorepair*)a;
    const scorepair* pb = (const scorepair*)b;
    if (pa->s < pb->s) return 1;
    if (pa->s > pb->s) return -1;
    return 0;
}

int yoloDecode(const float* pred, int channels, int anchors, int boxes, int classes,
               int topk, yolo_detection* dets, int maxDets) {
    (void)channels;
    int nc = classes;
    int k1 = anchors < topk ? anchors : topk;
    scorepair* stage1 = (scorepair*)malloc(sizeof(scorepair) * (size_t)anchors);
    for (int a = 0; a < anchors; a++) {
        float best = pred[(size_t)boxes * anchors + a];
        for (int c = 1; c < nc; c++) {
            float v = pred[(size_t)(boxes + c) * anchors + a];
            if (v > best) best = v;
        }
        stage1[a].s = best;
        stage1[a].i = a;
    }
    qsort(stage1, (size_t)anchors, sizeof(scorepair), cmpDesc);

    int npair = k1 * nc;
    scorepair* stage2 = (scorepair*)malloc(sizeof(scorepair) * (size_t)npair);
    for (int i = 0; i < k1; i++) {
        int a = stage1[i].i;
        for (int c = 0; c < nc; c++) {
            stage2[i * nc + c].s = pred[(size_t)(boxes + c) * anchors + a];
            stage2[i * nc + c].i = i * nc + c;
        }
    }
    qsort(stage2, (size_t)npair, sizeof(scorepair), cmpDesc);

    int k2 = npair < topk ? npair : topk;
    if (k2 > maxDets) k2 = maxDets;
    for (int j = 0; j < k2; j++) {
        int flat = stage2[j].i;
        int i = flat / nc;
        int c = flat % nc;
        int a = stage1[i].i;
        dets[j].x1 = pred[(size_t)0 * anchors + a];
        dets[j].y1 = pred[(size_t)1 * anchors + a];
        dets[j].x2 = pred[(size_t)2 * anchors + a];
        dets[j].y2 = pred[(size_t)3 * anchors + a];
        dets[j].score = stage2[j].s;
        dets[j].cls = c;
    }
    free(stage1);
    free(stage2);
    return k2;
}
