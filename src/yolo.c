#include "yolo.h"
#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct yolo_model {
    yolo_graph g;
    float* outHost;
    int outElems;
};

static char g_lastError[512] = "";

int yolo_open(const char* onnx_path, const char* shader_dir, yolo_model** out) {
    yolo_model* m = (yolo_model*)calloc(1, sizeof(yolo_model));
    if (m == NULL) {
        snprintf(g_lastError, sizeof(g_lastError), "out of memory");
        return -1;
    }
    if (graphLoad(&m->g, onnx_path, shader_dir) != 0) {
        snprintf(g_lastError, sizeof(g_lastError), "failed to load model from %s", onnx_path);
        free(m);
        return -1;
    }
    m->outElems = m->g.outChannels * m->g.outAnchors;
    m->outHost = (float*)malloc(sizeof(float) * (size_t)m->outElems);
    if (m->outHost == NULL) {
        snprintf(g_lastError, sizeof(g_lastError), "out of memory for output buffer");
        graphFree(&m->g);
        free(m);
        return -1;
    }
    *out = m;
    return 0;
}

int yolo_infer(yolo_model* m, const float* rgb, yolo_detection* dets, int max_dets, int* n_dets) {
    graphRun(&m->g, rgb);
    graphReadOutput(&m->g, m->outHost);
    int n = yoloDecode(m->outHost, m->g.outChannels, m->g.outAnchors, m->g.outBoxes,
                       m->g.outClasses, m->g.outTopk, dets, max_dets);
    *n_dets = n;
    return 0;
}

void yolo_close(yolo_model* m) {
    if (m == NULL) return;
    graphFree(&m->g);
    free(m->outHost);
    free(m);
}

const char* yolo_last_error(void) {
    return g_lastError;
}
