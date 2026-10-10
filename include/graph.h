#ifndef graph_h
#define graph_h

#include "dispatch.h"
#include "json.h"

typedef struct yolo_graph {
    session sess;
    buffer* buffers;
    char** bufferNames;
    int bufferCount;
    buffer* weights;
    char** weightNames;
    int weightCount;
    operation* ops;
    int opCount;
    int inputIndex;
    int outputIndex;
    int outChannels;
    int outAnchors;
    int outBoxes;
    int outClasses;
    int outTopk;
    VkCommandPool uploadPool;
    VkCommandBuffer uploadCmd;
    char** onnxNames;
    int* onnxBuf;
    int* onnxOff;
    int onnxCount;
} yolo_graph;

int graphLoad(yolo_graph* g, const char* modelDir, const char* shaderDir);
void graphFree(yolo_graph* g);
void graphRun(yolo_graph* g, const float* input);
void graphRunLogged(yolo_graph* g, const float* input, const char* phase, int token);
void graphReadOutput(yolo_graph* g, float* out);
int graphReadTensor(yolo_graph* g, const char* onnxName, float* out, int* count);

#endif
