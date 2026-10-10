#ifndef compile_h
#define compile_h

#include "dispatch.h"

typedef struct graph_res {
    char* name;
    int isWeight;
    int64_t size;
    int64_t off;
    int64_t nbytes;
} graph_res;

typedef struct graph_op {
    char shader[128];
    int buffers[MAX_OP_BUFFERS];
    int bufferCount;
    int pushConstants[MAX_PUSH_CONSTANTS];
    int pushConstantCount;
    int dispatchX;
    int dispatchY;
    int dispatchZ;
    int layer;
} graph_op;

typedef struct graph_plan {
    graph_res* res;
    int resCount;
    unsigned char* wblob;
    int64_t wblobLen;
    graph_op* ops;
    int opCount;
    char** tensorNames;
    int* tensorRes;
    int* tensorOff;
    int tensorCount;
    int inputRes;
    int outputRes;
    int outChannels;
    int outAnchors;
    int outBoxes;
    int outClasses;
    int outTopk;
} graph_plan;

int compileOnnx(const char* onnxPath, graph_plan* plan);
void freePlan(graph_plan* plan);
const char* compileLastError(void);

#endif
