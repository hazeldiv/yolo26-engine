#include "onnx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const unsigned char* p;
    const unsigned char* end;
} pb_t;

static uint64_t pbVarint(pb_t* c) {
    uint64_t v = 0;
    int shift = 0;
    while (c->p < c->end) {
        unsigned char b = *c->p++;
        v |= (uint64_t)(b & 0x7f) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return v;
}

static int pbTag(pb_t* c, int* field, int* wire) {
    if (c->p >= c->end) return 0;
    uint64_t key = pbVarint(c);
    *field = (int)(key >> 3);
    *wire = (int)(key & 7);
    return 1;
}

static const unsigned char* pbBytes(pb_t* c, size_t* len) {
    uint64_t n = pbVarint(c);
    const unsigned char* d = c->p;
    if (c->p + n > c->end) n = (uint64_t)(c->end - c->p);
    c->p += n;
    *len = (size_t)n;
    return d;
}

static char* pbString(pb_t* c) {
    size_t n;
    const unsigned char* d = pbBytes(c, &n);
    char* s = (char*)malloc(n + 1);
    memcpy(s, d, n);
    s[n] = 0;
    return s;
}

static void pbSkip(pb_t* c, int wire) {
    if (wire == 0) {
        pbVarint(c);
    } else if (wire == 2) {
        size_t n;
        pbBytes(c, &n);
    } else if (wire == 5) {
        c->p += 4;
    } else if (wire == 1) {
        c->p += 8;
    }
}

static void pbRepeatedInt64(pb_t* c, int64_t** out, int* count) {
    while (c->p < c->end) {
        *out = (int64_t*)realloc(*out, ((size_t)*count + 1) * sizeof(int64_t));
        (*out)[(*count)++] = (int64_t)pbVarint(c);
    }
}

static void pbRepeatedFloat(pb_t* c, float** out, int* count) {
    size_t n;
    const unsigned char* d = pbBytes(c, &n);
    int cnt = (int)(n / 4);
    *out = (float*)realloc(*out, ((size_t)*count + cnt) * sizeof(float));
    memcpy(*out + *count, d, (size_t)cnt * 4);
    *count += cnt;
}

static void parseTensor(pb_t* c, onnx_tensor* t) {
    memset(t, 0, sizeof(*t));
    int64_t* dims = NULL;
    int nDims = 0;
    float* fdata = NULL;
    int nf = 0;
    int64_t* idata = NULL;
    int ni = 0;
    const unsigned char* raw = NULL;
    size_t rawLen = 0;

    while (c->p < c->end) {
        int f, w;
        if (!pbTag(c, &f, &w)) break;
        if (f == 1 && w == 2) {
            size_t n;
            const unsigned char* d = pbBytes(c, &n);
            pb_t sub = {d, d + n};
            pbRepeatedInt64(&sub, &dims, &nDims);
        } else if (f == 1 && w == 0) {
            dims = (int64_t*)realloc(dims, ((size_t)nDims + 1) * sizeof(int64_t));
            dims[nDims++] = (int64_t)pbVarint(c);
        } else if (f == 2 && w == 0) {
            t->dataType = (int)pbVarint(c);
        } else if (f == 4 && w == 2) {
            pbRepeatedFloat(c, &fdata, &nf);
        } else if (f == 4 && w == 5) {
            float v;
            memcpy(&v, c->p, 4);
            c->p += 4;
            fdata = (float*)realloc(fdata, ((size_t)nf + 1) * sizeof(float));
            fdata[nf++] = v;
        } else if (f == 7 && w == 2) {
            size_t n;
            const unsigned char* d = pbBytes(c, &n);
            pb_t sub = {d, d + n};
            pbRepeatedInt64(&sub, &idata, &ni);
        } else if (f == 7 && w == 0) {
            idata = (int64_t*)realloc(idata, ((size_t)ni + 1) * sizeof(int64_t));
            idata[ni++] = (int64_t)pbVarint(c);
        } else if (f == 8 && w == 2) {
            t->name = pbString(c);
        } else if (f == 9 && w == 2) {
            raw = pbBytes(c, &rawLen);
        } else {
            pbSkip(c, w);
        }
    }

    t->dims = dims;
    t->nDims = nDims;
    int64_t count = 1;
    for (int i = 0; i < nDims; i++) count *= dims[i];
    t->count = count;

    if (t->dataType == 1) {
        t->fdata = (float*)calloc((size_t)count, sizeof(float));
        if (nf > 0) memcpy(t->fdata, fdata, (size_t)nf * 4);
        else if (raw != NULL) memcpy(t->fdata, raw, (size_t)count * 4);
    } else if (t->dataType == 7) {
        t->idata = (int64_t*)calloc((size_t)count, sizeof(int64_t));
        if (ni > 0) memcpy(t->idata, idata, (size_t)ni * 8);
        else if (raw != NULL) memcpy(t->idata, raw, (size_t)count * 8);
    }
    free(fdata);
    free(idata);
}

static void parseAttr(pb_t* c, onnx_attr* a) {
    memset(a, 0, sizeof(*a));
    while (c->p < c->end) {
        int f, w;
        if (!pbTag(c, &f, &w)) break;
        if (f == 1 && w == 2) {
            a->name = pbString(c);
        } else if (f == 2 && w == 5) {
            memcpy(&a->f, c->p, 4);
            c->p += 4;
        } else if (f == 3 && w == 0) {
            a->i = (int64_t)pbVarint(c);
        } else if (f == 4 && w == 2) {
            size_t n;
            const unsigned char* d = pbBytes(c, &n);
            a->s = (char*)malloc(n + 1);
            memcpy(a->s, d, n);
            a->s[n] = 0;
        } else if (f == 7 && w == 2) {
            pbRepeatedFloat(c, &a->floats, &a->nFloats);
        } else if (f == 7 && w == 5) {
            float v;
            memcpy(&v, c->p, 4);
            c->p += 4;
            a->floats = (float*)realloc(a->floats, ((size_t)a->nFloats + 1) * sizeof(float));
            a->floats[a->nFloats++] = v;
        } else if (f == 8 && w == 2) {
            size_t n;
            const unsigned char* d = pbBytes(c, &n);
            pb_t sub = {d, d + n};
            pbRepeatedInt64(&sub, &a->ints, &a->nInts);
        } else if (f == 8 && w == 0) {
            a->ints = (int64_t*)realloc(a->ints, ((size_t)a->nInts + 1) * sizeof(int64_t));
            a->ints[a->nInts++] = (int64_t)pbVarint(c);
        } else {
            pbSkip(c, w);
        }
    }
}

static void parseNode(pb_t* c, onnx_node* n) {
    memset(n, 0, sizeof(*n));
    while (c->p < c->end) {
        int f, w;
        if (!pbTag(c, &f, &w)) break;
        if (f == 1 && w == 2) {
            n->inputs = (char**)realloc(n->inputs, ((size_t)n->nInputs + 1) * sizeof(char*));
            n->inputs[n->nInputs++] = pbString(c);
        } else if (f == 2 && w == 2) {
            n->outputs = (char**)realloc(n->outputs, ((size_t)n->nOutputs + 1) * sizeof(char*));
            n->outputs[n->nOutputs++] = pbString(c);
        } else if (f == 3 && w == 2) {
            n->name = pbString(c);
        } else if (f == 4 && w == 2) {
            n->opType = pbString(c);
        } else if (f == 5 && w == 2) {
            size_t len;
            const unsigned char* d = pbBytes(c, &len);
            pb_t sub = {d, d + len};
            n->attrs = (onnx_attr*)realloc(n->attrs, ((size_t)n->nAttrs + 1) * sizeof(onnx_attr));
            parseAttr(&sub, &n->attrs[n->nAttrs++]);
        } else {
            pbSkip(c, w);
        }
    }
}

static void parseShape(pb_t* c, int64_t** dims, int* nDims) {
    while (c->p < c->end) {
        int f, w;
        if (!pbTag(c, &f, &w)) break;
        if (f == 1 && w == 2) {
            size_t len;
            const unsigned char* d = pbBytes(c, &len);
            pb_t sub = {d, d + len};
            int64_t v = -1;
            while (sub.p < sub.end) {
                int ff, ww;
                if (!pbTag(&sub, &ff, &ww)) break;
                if (ff == 1 && ww == 0) v = (int64_t)pbVarint(&sub);
                else pbSkip(&sub, ww);
            }
            *dims = (int64_t*)realloc(*dims, ((size_t)*nDims + 1) * sizeof(int64_t));
            (*dims)[(*nDims)++] = v;
        } else {
            pbSkip(c, w);
        }
    }
}

static void parseType(pb_t* c, int64_t** dims, int* nDims) {
    while (c->p < c->end) {
        int f, w;
        if (!pbTag(c, &f, &w)) break;
        if (f == 1 && w == 2) {
            size_t len;
            const unsigned char* d = pbBytes(c, &len);
            pb_t sub = {d, d + len};
            while (sub.p < sub.end) {
                int ff, ww;
                if (!pbTag(&sub, &ff, &ww)) break;
                if (ff == 2 && ww == 2) {
                    size_t l2;
                    const unsigned char* d2 = pbBytes(&sub, &l2);
                    pb_t s2 = {d2, d2 + l2};
                    parseShape(&s2, dims, nDims);
                } else {
                    pbSkip(&sub, ww);
                }
            }
        } else {
            pbSkip(c, w);
        }
    }
}

static void parseValueInfo(pb_t* c, onnx_model* m) {
    char* name = NULL;
    int64_t* dims = NULL;
    int nDims = 0;
    while (c->p < c->end) {
        int f, w;
        if (!pbTag(c, &f, &w)) break;
        if (f == 1 && w == 2) {
            name = pbString(c);
        } else if (f == 2 && w == 2) {
            size_t len;
            const unsigned char* d = pbBytes(c, &len);
            pb_t sub = {d, d + len};
            parseType(&sub, &dims, &nDims);
        } else {
            pbSkip(c, w);
        }
    }
    if (m->inputName == NULL && name != NULL) {
        m->inputName = name;
        m->inputDims = dims;
        m->nInputDims = nDims;
    } else {
        free(name);
        free(dims);
    }
}

static void parseGraph(pb_t* c, onnx_model* m) {
    while (c->p < c->end) {
        int f, w;
        if (!pbTag(c, &f, &w)) break;
        if (f == 1 && w == 2) {
            size_t len;
            const unsigned char* d = pbBytes(c, &len);
            pb_t sub = {d, d + len};
            m->nodes = (onnx_node*)realloc(m->nodes, ((size_t)m->nNodes + 1) * sizeof(onnx_node));
            parseNode(&sub, &m->nodes[m->nNodes++]);
        } else if (f == 5 && w == 2) {
            size_t len;
            const unsigned char* d = pbBytes(c, &len);
            pb_t sub = {d, d + len};
            m->inits = (onnx_tensor*)realloc(m->inits, ((size_t)m->nInit + 1) * sizeof(onnx_tensor));
            parseTensor(&sub, &m->inits[m->nInit++]);
        } else if (f == 11 && w == 2) {
            size_t len;
            const unsigned char* d = pbBytes(c, &len);
            pb_t sub = {d, d + len};
            parseValueInfo(&sub, m);
        } else {
            pbSkip(c, w);
        }
    }
}

int onnxLoad(const char* path, onnx_model* m) {
    memset(m, 0, sizeof(*m));
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* buf = (unsigned char*)malloc((size_t)len);
    if (buf == NULL || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);

    pb_t c = {buf, buf + len};
    while (c.p < c.end) {
        int fld, w;
        if (!pbTag(&c, &fld, &w)) break;
        if (fld == 7 && w == 2) {
            size_t l;
            const unsigned char* d = pbBytes(&c, &l);
            pb_t sub = {d, d + l};
            parseGraph(&sub, m);
        } else {
            pbSkip(&c, w);
        }
    }
    free(buf);
    return 0;
}

const onnx_attr* onnxAttr(const onnx_node* n, const char* name) {
    for (int i = 0; i < n->nAttrs; i++) {
        if (strcmp(n->attrs[i].name, name) == 0) return &n->attrs[i];
    }
    return NULL;
}

const onnx_tensor* onnxInit(const onnx_model* m, const char* name) {
    for (int i = 0; i < m->nInit; i++) {
        if (strcmp(m->inits[i].name, name) == 0) return &m->inits[i];
    }
    return NULL;
}

static void freeAttr(onnx_attr* a) {
    free(a->name);
    free(a->s);
    free(a->ints);
    free(a->floats);
}

void onnxFree(onnx_model* m) {
    for (int i = 0; i < m->nNodes; i++) {
        onnx_node* n = &m->nodes[i];
        free(n->name);
        free(n->opType);
        for (int j = 0; j < n->nInputs; j++) free(n->inputs[j]);
        for (int j = 0; j < n->nOutputs; j++) free(n->outputs[j]);
        for (int j = 0; j < n->nAttrs; j++) freeAttr(&n->attrs[j]);
        free(n->inputs);
        free(n->outputs);
        free(n->attrs);
    }
    for (int i = 0; i < m->nInit; i++) {
        free(m->inits[i].name);
        free(m->inits[i].dims);
        free(m->inits[i].fdata);
        free(m->inits[i].idata);
    }
    free(m->nodes);
    free(m->inits);
    free(m->inputName);
    free(m->inputDims);
    memset(m, 0, sizeof(*m));
}
