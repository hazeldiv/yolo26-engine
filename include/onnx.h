#ifndef onnx_h
#define onnx_h

#include <stdint.h>

typedef struct onnx_attr {
    char* name;
    int64_t i;
    float f;
    char* s;
    int64_t* ints;
    int nInts;
    float* floats;
    int nFloats;
} onnx_attr;

typedef struct onnx_node {
    char* name;
    char* opType;
    char** inputs;
    int nInputs;
    char** outputs;
    int nOutputs;
    onnx_attr* attrs;
    int nAttrs;
} onnx_node;

typedef struct onnx_tensor {
    char* name;
    int dataType;
    int64_t* dims;
    int nDims;
    int64_t count;
    float* fdata;
    int64_t* idata;
} onnx_tensor;

typedef struct onnx_model {
    onnx_node* nodes;
    int nNodes;
    onnx_tensor* inits;
    int nInit;
    char* inputName;
    int64_t* inputDims;
    int nInputDims;
} onnx_model;

int onnxLoad(const char* path, onnx_model* m);
void onnxFree(onnx_model* m);
const onnx_attr* onnxAttr(const onnx_node* n, const char* name);
const onnx_tensor* onnxInit(const onnx_model* m, const char* name);

#endif
