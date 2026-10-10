#include "compile.h"
#include "onnx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DIMS 8

typedef struct {
    int n;
    int64_t d[MAX_DIMS];
} shp;

typedef struct {
    char* name;
    shp s;
} shpEnt;

typedef struct {
    char* name;
    int res;
    int off;
} slotEnt;

typedef struct {
    char* name;
    int node;
} prodEnt;

typedef struct {
    char* name;
} nameEnt;

typedef struct {
    char* tensor;
    char* alias;
} siluEnt;

typedef struct {
    char* name;
    int res;
} constEnt;

typedef struct {
    int node;
    int softmax;
} triggerEnt;

typedef struct {
    onnx_model* onm;
    graph_plan* plan;
    shpEnt* shapes;
    int nShape;
    slotEnt* slots;
    int nSlot;
    prodEnt* prods;
    int nProd;
    nameEnt* skip;
    int nSkip;
    nameEnt* handled;
    int nHandled;
    siluEnt* silu;
    int nSilu;
    constEnt* consts;
    int nConst;
    int zeroRes;
    int im2colRes;
    int padRes;
    int nAct;
    int failed;
    char err[256];
    int64_t wblobCap;
} builder;

static char g_error[256] = "";

const char* compileLastError(void) {
    return g_error;
}

static void fail(builder* b, const char* msg) {
    if (!b->failed) {
        b->failed = 1;
        snprintf(b->err, sizeof(b->err), "%s", msg);
    }
}

static int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static int64_t ceil16(int64_t x) {
    return (x + 15) / 16 * 16;
}

static int64_t ceil32(int64_t x) {
    return (x + 31) / 32 * 32;
}

static int64_t prodAll(shp s) {
    int64_t r = 1;
    for (int i = 0; i < s.n; i++) r *= s.d[i];
    return r;
}

static int64_t prodFrom(shp s, int k) {
    int64_t r = 1;
    for (int i = k; i < s.n; i++) r *= s.d[i];
    return r;
}

static unsigned short floatToHalf(float f) {
    unsigned int x;
    memcpy(&x, &f, 4);
    unsigned int sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xff);
    unsigned int mant = x & 0x7fffff;
    if (exp == 255) {
        if (mant) return (unsigned short)(sign | 0x7e00);
        return (unsigned short)(sign | 0x7c00);
    }
    int e = exp - 127 + 15;
    if (e >= 31) return (unsigned short)(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (unsigned short)sign;
        mant |= 0x800000;
        int shift = 14 - e;
        unsigned int m = mant >> shift;
        unsigned int rem = mant & ((1u << shift) - 1);
        unsigned int half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1))) m++;
        return (unsigned short)(sign | m);
    }
    unsigned int m = mant >> 13;
    unsigned int rem = mant & 0x1fff;
    unsigned short r = (unsigned short)(sign | ((unsigned int)e << 10) | m);
    if (rem > 0x1000 || (rem == 0x1000 && (m & 1))) r++;
    return r;
}

static int pushRes(graph_plan* p, const char* name, int isWeight, int64_t size, int64_t off, int64_t nbytes) {
    p->res = (graph_res*)realloc(p->res, ((size_t)p->resCount + 1) * sizeof(graph_res));
    graph_res* r = &p->res[p->resCount];
    r->name = strdup(name);
    r->isWeight = isWeight;
    r->size = size;
    r->off = off;
    r->nbytes = nbytes;
    return p->resCount++;
}

static void setShape(builder* b, const char* name, shp s) {
    for (int i = 0; i < b->nShape; i++) {
        if (strcmp(b->shapes[i].name, name) == 0) {
            b->shapes[i].s = s;
            return;
        }
    }
    b->shapes = (shpEnt*)realloc(b->shapes, ((size_t)b->nShape + 1) * sizeof(shpEnt));
    b->shapes[b->nShape].name = (char*)name;
    b->shapes[b->nShape].s = s;
    b->nShape++;
}

static shp* shapeOf(builder* b, const char* name) {
    if (name == NULL || name[0] == 0) return NULL;
    for (int i = 0; i < b->nShape; i++) {
        if (strcmp(b->shapes[i].name, name) == 0) return &b->shapes[i].s;
    }
    return NULL;
}

static int prodNode(builder* b, const char* name) {
    for (int i = 0; i < b->nProd; i++) {
        if (strcmp(b->prods[i].name, name) == 0) return b->prods[i].node;
    }
    return -1;
}

static const onnx_node* nodeAt(builder* b, int idx) {
    if (idx < 0) return NULL;
    return &b->onm->nodes[idx];
}

static int nameIn(nameEnt* list, int n, const char* name) {
    for (int i = 0; i < n; i++) {
        if (strcmp(list[i].name, name) == 0) return 1;
    }
    return 0;
}

static void addName(nameEnt** list, int* n, const char* name) {
    if (nameIn(*list, *n, name)) return;
    *list = (nameEnt*)realloc(*list, ((size_t)*n + 1) * sizeof(nameEnt));
    (*list)[*n].name = (char*)name;
    (*n)++;
}

static int isSkip(builder* b, const char* name) {
    return nameIn(b->skip, b->nSkip, name);
}

static int isHandled(builder* b, const char* name) {
    return nameIn(b->handled, b->nHandled, name);
}

static const char* siluAlias(builder* b, const char* out) {
    for (int i = 0; i < b->nSilu; i++) {
        if (strcmp(b->silu[i].tensor, out) == 0) return b->silu[i].alias;
    }
    return NULL;
}

static int isSilu(builder* b, const char* out) {
    return siluAlias(b, out) != NULL;
}

static void setSlot(builder* b, const char* name, int res, int off) {
    for (int i = 0; i < b->nSlot; i++) {
        if (strcmp(b->slots[i].name, name) == 0) {
            b->slots[i].res = res;
            b->slots[i].off = off;
            return;
        }
    }
    b->slots = (slotEnt*)realloc(b->slots, ((size_t)b->nSlot + 1) * sizeof(slotEnt));
    b->slots[b->nSlot].name = (char*)name;
    b->slots[b->nSlot].res = res;
    b->slots[b->nSlot].off = off;
    b->nSlot++;
}

static slotEnt* slotOf(builder* b, const char* name) {
    for (int i = 0; i < b->nSlot; i++) {
        if (strcmp(b->slots[i].name, name) == 0) return &b->slots[i];
    }
    return NULL;
}

static int newBuffer(builder* b, int64_t size) {
    char name[32];
    snprintf(name, sizeof(name), "t%d", b->nAct);
    b->nAct++;
    return pushRes(b->plan, name, 0, size, 0, 0);
}

static int addWeight(builder* b, const char* name, const void* data, int64_t nbytes) {
    graph_plan* p = b->plan;
    if (p->wblobLen + nbytes > b->wblobCap) {
        int64_t cap = b->wblobCap ? b->wblobCap : (int64_t)1 << 20;
        while (cap < p->wblobLen + nbytes) cap *= 2;
        p->wblob = (unsigned char*)realloc(p->wblob, (size_t)cap);
        b->wblobCap = cap;
    }
    memcpy(p->wblob + p->wblobLen, data, (size_t)nbytes);
    int64_t off = p->wblobLen;
    p->wblobLen += nbytes;
    return pushRes(p, name, 1, 0, off, nbytes);
}

static int weightFp32(builder* b, const char* name, const float* data, int64_t count) {
    return addWeight(b, name, data, count * 4);
}

static int weightFp16Raw(builder* b, const char* name, const float* data, int64_t count) {
    unsigned short* h = (unsigned short*)malloc((size_t)count * 2);
    for (int64_t i = 0; i < count; i++) h[i] = floatToHalf(data[i]);
    int r = addWeight(b, name, h, count * 2);
    free(h);
    return r;
}

static int weightBlock16(builder* b, const char* name, const float* W2, int K, int Cout, int Kpad, int Npad) {
    unsigned short* out = (unsigned short*)malloc((size_t)Kpad * Npad * 2);
    for (int g = 0; g < Kpad / 8; g++) {
        for (int nn = 0; nn < Npad; nn++) {
            for (int j = 0; j < 8; j++) {
                int i = g * 8 + j;
                float v = (i < K && nn < Cout) ? W2[(int64_t)i * Cout + nn] : 0.0f;
                out[((int64_t)g * Npad + nn) * 8 + j] = floatToHalf(v);
            }
        }
    }
    int r = addWeight(b, name, out, (int64_t)Kpad * Npad * 2);
    free(out);
    return r;
}

static int constBuffer(builder* b, const char* name) {
    for (int i = 0; i < b->nConst; i++) {
        if (strcmp(b->consts[i].name, name) == 0) return b->consts[i].res;
    }
    const onnx_tensor* t = onnxInit(b->onm, name);
    if (t == NULL || t->fdata == NULL) {
        fail(b, "missing constant initializer");
        return -1;
    }
    char buf[200];
    snprintf(buf, sizeof(buf), "%s_c", name);
    int r = weightFp32(b, buf, t->fdata, t->count);
    b->consts = (constEnt*)realloc(b->consts, ((size_t)b->nConst + 1) * sizeof(constEnt));
    b->consts[b->nConst].name = (char*)name;
    b->consts[b->nConst].res = r;
    b->nConst++;
    return r;
}

static int ensureZero(builder* b) {
    if (b->zeroRes < 0) {
        float* z = (float*)calloc(8192, 4);
        b->zeroRes = weightFp32(b, "wzero", z, 8192);
        free(z);
    }
    return b->zeroRes;
}

static void opEmit(builder* b, const char* shader, const int* bufs, int nBuf,
                   const int* pc, int nPc, int dx, int dy, int dz, int layer) {
    graph_plan* p = b->plan;
    p->ops = (graph_op*)realloc(p->ops, ((size_t)p->opCount + 1) * sizeof(graph_op));
    graph_op* o = &p->ops[p->opCount++];
    snprintf(o->shader, sizeof(o->shader), "%s.spv", shader);
    for (int i = 0; i < nBuf; i++) o->buffers[i] = bufs[i];
    o->bufferCount = nBuf;
    for (int i = 0; i < nPc; i++) o->pushConstants[i] = pc[i];
    o->pushConstantCount = nPc;
    o->dispatchX = dx;
    o->dispatchY = dy;
    o->dispatchZ = dz;
    o->layer = layer;
}

static int64_t attrInt(const onnx_node* n, const char* name, int64_t def) {
    const onnx_attr* a = onnxAttr(n, name);
    if (a == NULL) return def;
    if (a->nInts > 0) return a->ints[0];
    return a->i;
}

static shp broadcast(shp a, shp b) {
    int n = a.n > b.n ? a.n : b.n;
    shp r;
    r.n = n;
    for (int i = 0; i < n; i++) {
        int64_t da = (i < n - a.n) ? 1 : a.d[i - (n - a.n)];
        int64_t db = (i < n - b.n) ? 1 : b.d[i - (n - b.n)];
        r.d[i] = da > db ? da : db;
    }
    return r;
}

static shp inferConv(builder* b, const onnx_node* n) {
    shp r;
    memset(&r, 0, sizeof(r));
    shp* xs = shapeOf(b, n->inputs[0]);
    const onnx_tensor* w = onnxInit(b->onm, n->inputs[1]);
    if (xs == NULL || w == NULL) return r;
    const onnx_attr* ks = onnxAttr(n, "kernel_shape");
    const onnx_attr* st = onnxAttr(n, "strides");
    const onnx_attr* pd = onnxAttr(n, "pads");
    const onnx_attr* dl = onnxAttr(n, "dilations");
    int64_t kh = ks ? ks->ints[0] : w->dims[2];
    int64_t kw = ks ? ks->ints[1] : w->dims[3];
    int64_t sh = st ? st->ints[0] : 1;
    int64_t sw = st ? st->ints[1] : 1;
    int64_t pt = pd ? pd->ints[0] : 0;
    int64_t pl = pd ? pd->ints[1] : 0;
    int64_t pb = pd ? pd->ints[2] : 0;
    int64_t pr = pd ? pd->ints[3] : 0;
    int64_t dh = dl ? dl->ints[0] : 1;
    int64_t dw = dl ? dl->ints[1] : 1;
    r.n = 4;
    r.d[0] = xs->d[0];
    r.d[1] = w->dims[0];
    r.d[2] = (xs->d[2] + pt + pb - dh * (kh - 1) - 1) / sh + 1;
    r.d[3] = (xs->d[3] + pl + pr - dw * (kw - 1) - 1) / sw + 1;
    return r;
}

static shp inferPool(builder* b, const onnx_node* n) {
    shp r;
    memset(&r, 0, sizeof(r));
    shp* xs = shapeOf(b, n->inputs[0]);
    if (xs == NULL) return r;
    const onnx_attr* ks = onnxAttr(n, "kernel_shape");
    const onnx_attr* st = onnxAttr(n, "strides");
    const onnx_attr* pd = onnxAttr(n, "pads");
    const onnx_attr* dl = onnxAttr(n, "dilations");
    int64_t kh = ks->ints[0];
    int64_t kw = ks->ints[1];
    int64_t sh = st ? st->ints[0] : 1;
    int64_t sw = st ? st->ints[1] : 1;
    int64_t pt = pd ? pd->ints[0] : 0;
    int64_t pl = pd ? pd->ints[1] : 0;
    int64_t pb = pd ? pd->ints[2] : 0;
    int64_t pr = pd ? pd->ints[3] : 0;
    int64_t dh = dl ? dl->ints[0] : 1;
    int64_t dw = dl ? dl->ints[1] : 1;
    r.n = 4;
    r.d[0] = xs->d[0];
    r.d[1] = xs->d[1];
    r.d[2] = (xs->d[2] + pt + pb - dh * (kh - 1) - 1) / sh + 1;
    r.d[3] = (xs->d[3] + pl + pr - dw * (kw - 1) - 1) / sw + 1;
    return r;
}

static void inferNode(builder* b, const onnx_node* n) {
    const char* t = n->opType;
    shp r;
    memset(&r, 0, sizeof(r));

    if (strcmp(t, "Conv") == 0) {
        shp s = inferConv(b, n);
        if (s.n == 4) setShape(b, n->outputs[0], s);
    } else if (strcmp(t, "MaxPool") == 0) {
        shp s = inferPool(b, n);
        if (s.n == 4) setShape(b, n->outputs[0], s);
    } else if (strcmp(t, "Resize") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        if (xs == NULL) return;
        const onnx_tensor* sizes = n->nInputs >= 4 ? onnxInit(b->onm, n->inputs[3]) : NULL;
        const onnx_tensor* scales = n->nInputs >= 3 ? onnxInit(b->onm, n->inputs[2]) : NULL;
        shp out = *xs;
        if (sizes && sizes->idata) {
            out.n = (int)sizes->count;
            for (int i = 0; i < out.n; i++) out.d[i] = sizes->idata[i];
        } else if (scales && scales->fdata) {
            for (int i = 0; i < out.n; i++) out.d[i] = (int64_t)floor(xs->d[i] * scales->fdata[i]);
        } else {
            return;
        }
        setShape(b, n->outputs[0], out);
    } else if (strcmp(t, "Concat") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        if (xs == NULL) return;
        shp out = *xs;
        int axis = (int)attrInt(n, "axis", 0);
        if (axis < 0) axis += out.n;
        for (int i = 1; i < n->nInputs; i++) {
            shp* s = shapeOf(b, n->inputs[i]);
            if (s == NULL) return;
            out.d[axis] += s->d[axis];
        }
        setShape(b, n->outputs[0], out);
    } else if (strcmp(t, "Split") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        if (xs == NULL) return;
        int axis = (int)attrInt(n, "axis", 0);
        if (axis < 0) axis += xs->n;
        const onnx_tensor* sp = n->nInputs >= 2 ? onnxInit(b->onm, n->inputs[1]) : NULL;
        for (int o = 0; o < n->nOutputs; o++) {
            shp out = *xs;
            out.d[axis] = (sp && sp->idata) ? sp->idata[o] : xs->d[axis] / n->nOutputs;
            setShape(b, n->outputs[o], out);
        }
    } else if (strcmp(t, "Reshape") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        const onnx_tensor* sh = onnxInit(b->onm, n->inputs[1]);
        if (xs == NULL || sh == NULL || sh->idata == NULL) return;
        shp out;
        out.n = (int)sh->count;
        int negIdx = -1;
        for (int i = 0; i < out.n; i++) {
            int64_t v = sh->idata[i];
            if (v == 0 && attrInt(n, "allowzero", 0) == 0) out.d[i] = xs->d[i];
            else if (v == -1) {
                out.d[i] = -1;
                negIdx = i;
            } else out.d[i] = v;
        }
        if (negIdx >= 0) {
            int64_t known = 1;
            for (int i = 0; i < out.n; i++) if (i != negIdx) known *= out.d[i];
            out.d[negIdx] = prodAll(*xs) / known;
        }
        setShape(b, n->outputs[0], out);
    } else if (strcmp(t, "Transpose") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        if (xs == NULL) return;
        const onnx_attr* pm = onnxAttr(n, "perm");
        shp out;
        out.n = xs->n;
        for (int i = 0; i < out.n; i++) {
            int p = pm ? (int)pm->ints[i] : (xs->n - 1 - i);
            out.d[i] = xs->d[p];
        }
        setShape(b, n->outputs[0], out);
    } else if (strcmp(t, "MatMul") == 0) {
        shp* as = shapeOf(b, n->inputs[0]);
        shp* bs = shapeOf(b, n->inputs[1]);
        if (as == NULL || bs == NULL || as->n < 2 || bs->n < 2) return;
        int an = as->n - 2;
        int bn = bs->n - 2;
        int nb = an > bn ? an : bn;
        shp out;
        out.n = nb + 2;
        for (int i = 0; i < nb; i++) {
            int64_t da = (i < nb - an) ? 1 : as->d[i - (nb - an)];
            int64_t db = (i < nb - bn) ? 1 : bs->d[i - (nb - bn)];
            out.d[i] = da > db ? da : db;
        }
        out.d[nb] = as->d[as->n - 2];
        out.d[nb + 1] = bs->d[bs->n - 1];
        setShape(b, n->outputs[0], out);
    } else if (strcmp(t, "Softmax") == 0 || strcmp(t, "Sigmoid") == 0 || strcmp(t, "Cast") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        if (xs == NULL) return;
        setShape(b, n->outputs[0], *xs);
    } else if (strcmp(t, "Add") == 0 || strcmp(t, "Sub") == 0 || strcmp(t, "Mul") == 0 ||
               strcmp(t, "Div") == 0 || strcmp(t, "Mod") == 0) {
        shp* as = shapeOf(b, n->inputs[0]);
        shp* bs = shapeOf(b, n->inputs[1]);
        if (as == NULL || bs == NULL) return;
        setShape(b, n->outputs[0], broadcast(*as, *bs));
    } else if (strcmp(t, "Slice") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        const onnx_tensor* st = onnxInit(b->onm, n->inputs[1]);
        const onnx_tensor* en = n->nInputs >= 3 ? onnxInit(b->onm, n->inputs[2]) : NULL;
        if (xs == NULL || st == NULL || en == NULL) return;
        const onnx_tensor* ax = n->nInputs >= 4 ? onnxInit(b->onm, n->inputs[3]) : NULL;
        const onnx_tensor* sp = n->nInputs >= 5 ? onnxInit(b->onm, n->inputs[4]) : NULL;
        shp out = *xs;
        for (int i = 0; i < (int)st->count; i++) {
            int a = (ax && ax->idata) ? (int)ax->idata[i] : i;
            if (a < 0) a += out.n;
            int64_t start = st->idata[i];
            int64_t end = en->idata[i];
            int64_t step = (sp && sp->idata) ? sp->idata[i] : 1;
            int64_t dim = out.d[a];
            if (start < 0) start += dim;
            if (end < 0) end += dim;
            if (start < 0) start = 0;
            if (end > dim) end = dim;
            if (start > dim) start = dim;
            int64_t n2 = step > 0 ? (end - start + step - 1) / step : 0;
            out.d[a] = n2 > 0 ? n2 : 0;
        }
        setShape(b, n->outputs[0], out);
    } else if (strcmp(t, "Unsqueeze") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        const onnx_tensor* ax = n->nInputs >= 2 ? onnxInit(b->onm, n->inputs[1]) : NULL;
        if (xs == NULL || ax == NULL || ax->idata == NULL) return;
        int naxes = (int)ax->count;
        shp out;
        out.n = xs->n + naxes;
        int64_t axes[MAX_DIMS];
        for (int i = 0; i < naxes; i++) axes[i] = ax->idata[i];
        for (int i = 0; i < naxes; i++) {
            for (int j = i + 1; j < naxes; j++) {
                if (axes[j] < axes[i]) {
                    int64_t tmp = axes[i];
                    axes[i] = axes[j];
                    axes[j] = tmp;
                }
            }
        }
        for (int i = 0; i < naxes; i++) if (axes[i] < 0) axes[i] += out.n;
        int si = 0;
        for (int i = 0; i < out.n; i++) {
            int isAxis = 0;
            for (int j = 0; j < naxes; j++) if (axes[j] == i) isAxis = 1;
            out.d[i] = isAxis ? 1 : xs->d[si++];
        }
        setShape(b, n->outputs[0], out);
    } else if (strcmp(t, "Expand") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        const onnx_tensor* sh = n->nInputs >= 2 ? onnxInit(b->onm, n->inputs[1]) : NULL;
        if (xs == NULL || sh == NULL || sh->idata == NULL) return;
        shp tgt;
        tgt.n = (int)sh->count;
        for (int i = 0; i < tgt.n; i++) tgt.d[i] = sh->idata[i];
        setShape(b, n->outputs[0], broadcast(*xs, tgt));
    } else if (strcmp(t, "GatherElements") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        if (xs == NULL) return;
        setShape(b, n->outputs[0], *xs);
    } else if (strcmp(t, "Flatten") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        if (xs == NULL) return;
        int axis = (int)attrInt(n, "axis", 1);
        if (axis < 0) axis += xs->n;
        shp out;
        out.n = 2;
        out.d[0] = 1;
        out.d[1] = 1;
        for (int i = 0; i < axis; i++) out.d[0] *= xs->d[i];
        for (int i = axis; i < xs->n; i++) out.d[1] *= xs->d[i];
        setShape(b, n->outputs[0], out);
    } else if (strcmp(t, "TopK") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        const onnx_tensor* k = onnxInit(b->onm, n->inputs[1]);
        if (xs == NULL || k == NULL || k->idata == NULL) return;
        int axis = (int)attrInt(n, "axis", -1);
        if (axis < 0) axis += xs->n;
        shp out = *xs;
        out.d[axis] = k->idata[0];
        for (int o = 0; o < n->nOutputs; o++) setShape(b, n->outputs[o], out);
    } else if (strcmp(t, "ReduceMax") == 0) {
        shp* xs = shapeOf(b, n->inputs[0]);
        if (xs == NULL) return;
        const onnx_attr* ax = onnxAttr(n, "axes");
        int keep = (int)attrInt(n, "keepdims", 1);
        int red[MAX_DIMS];
        memset(red, 0, sizeof(red));
        if (ax) {
            for (int i = 0; i < ax->nInts; i++) {
                int a = (int)ax->ints[i];
                if (a < 0) a += xs->n;
                red[a] = 1;
            }
        }
        shp out;
        out.n = 0;
        for (int i = 0; i < xs->n; i++) {
            if (red[i]) {
                if (keep) out.d[out.n++] = 1;
            } else {
                out.d[out.n++] = xs->d[i];
            }
        }
        setShape(b, n->outputs[0], out);
    }
}

static void buildProds(builder* b) {
    onnx_model* m = b->onm;
    for (int i = 0; i < m->nNodes; i++) {
        for (int j = 0; j < m->nodes[i].nOutputs; j++) {
            b->prods = (prodEnt*)realloc(b->prods, ((size_t)b->nProd + 1) * sizeof(prodEnt));
            b->prods[b->nProd].name = m->nodes[i].outputs[j];
            b->prods[b->nProd].node = i;
            b->nProd++;
        }
    }
}

static void inferShapes(builder* b) {
    onnx_model* m = b->onm;
    if (m->inputName != NULL) {
        shp s;
        s.n = m->nInputDims;
        for (int i = 0; i < s.n; i++) s.d[i] = m->inputDims[i];
        setShape(b, m->inputName, s);
    }
    for (int i = 0; i < m->nInit; i++) {
        shp s;
        s.n = m->inits[i].nDims;
        for (int j = 0; j < s.n; j++) s.d[j] = m->inits[i].dims[j];
        setShape(b, m->inits[i].name, s);
    }
    for (int i = 0; i < m->nNodes; i++) inferNode(b, &m->nodes[i]);
}

static const onnx_node* consumer(builder* b, const char* tensor, const char* opType) {
    for (int i = 0; i < b->onm->nNodes; i++) {
        const onnx_node* n = &b->onm->nodes[i];
        if (strcmp(n->opType, opType) != 0) continue;
        for (int j = 0; j < n->nInputs; j++) {
            if (strcmp(n->inputs[j], tensor) == 0) return n;
        }
    }
    return NULL;
}

static void detectSilu(builder* b) {
    onnx_model* m = b->onm;
    for (int i = 0; i < m->nNodes; i++) {
        const onnx_node* n = &m->nodes[i];
        if (strcmp(n->opType, "Conv") != 0) continue;
        const char* x = n->outputs[0];
        const onnx_node* sig = NULL;
        for (int j = 0; j < m->nNodes; j++) {
            const onnx_node* c = &m->nodes[j];
            if (strcmp(c->opType, "Sigmoid") == 0 && c->nInputs > 0 && strcmp(c->inputs[0], x) == 0) sig = c;
        }
        if (sig == NULL) continue;
        for (int j = 0; j < m->nNodes; j++) {
            const onnx_node* c = &m->nodes[j];
            if (strcmp(c->opType, "Mul") != 0 || c->nInputs != 2) continue;
            int a = strcmp(c->inputs[0], x) == 0 && strcmp(c->inputs[1], sig->outputs[0]) == 0;
            int bb = strcmp(c->inputs[1], x) == 0 && strcmp(c->inputs[0], sig->outputs[0]) == 0;
            if (a || bb) {
                b->silu = (siluEnt*)realloc(b->silu, ((size_t)b->nSilu + 1) * sizeof(siluEnt));
                b->silu[b->nSilu].tensor = (char*)x;
                b->silu[b->nSilu].alias = c->outputs[0];
                b->nSilu++;
                addName(&b->skip, &b->nSkip, sig->name);
                addName(&b->skip, &b->nSkip, c->name);
            }
        }
    }
}

static const char* findPred(builder* b) {
    onnx_model* m = b->onm;
    const onnx_node* rm = NULL;
    for (int i = 0; i < m->nNodes; i++) {
        if (strcmp(m->nodes[i].opType, "ReduceMax") == 0) {
            rm = &m->nodes[i];
            break;
        }
    }
    if (rm == NULL) return NULL;
    const onnx_node* sp = nodeAt(b, prodNode(b, rm->inputs[0]));
    if (sp == NULL) return NULL;
    const onnx_node* tr = nodeAt(b, prodNode(b, sp->inputs[0]));
    if (tr == NULL) return NULL;
    return tr->inputs[0];
}

static void collectVisit(builder* b, const char* t, nameEnt** seen, int* nSeen, int** order, int* nOrder) {
    if (t == NULL || t[0] == 0) return;
    if (onnxInit(b->onm, t) != NULL) return;
    if (nameIn(*seen, *nSeen, t)) return;
    addName(seen, nSeen, t);
    int ni = prodNode(b, t);
    if (ni < 0) return;
    const onnx_node* n = &b->onm->nodes[ni];
    for (int i = 0; i < n->nInputs; i++) collectVisit(b, n->inputs[i], seen, nSeen, order, nOrder);
    *order = (int*)realloc(*order, ((size_t)*nOrder + 1) * sizeof(int));
    (*order)[(*nOrder)++] = ni;
}

static void collect(builder* b, const char* pred, int** order, int* nOrder) {
    nameEnt* seen = NULL;
    int nSeen = 0;
    *order = NULL;
    *nOrder = 0;
    collectVisit(b, pred, &seen, &nSeen, order, nOrder);
    free(seen);
}

static void findAttention(builder* b, int* order, int nOrder, triggerEnt** triggers, int* nTrig) {
    *triggers = NULL;
    *nTrig = 0;
    for (int i = 0; i < nOrder; i++) {
        const onnx_node* sm = &b->onm->nodes[order[i]];
        if (strcmp(sm->opType, "Softmax") != 0) continue;
        const onnx_node* m1 = nodeAt(b, prodNode(b, sm->inputs[0]));
        if (m1 == NULL) continue;
        const onnx_node* transq = nodeAt(b, prodNode(b, m1->inputs[0]));
        if (transq == NULL) continue;
        const onnx_node* qmul = nodeAt(b, prodNode(b, transq->inputs[0]));
        if (qmul == NULL) continue;
        const onnx_node* split_q = nodeAt(b, prodNode(b, qmul->inputs[0]));
        if (split_q == NULL) continue;
        int rq = prodNode(b, split_q->inputs[0]);
        if (rq < 0) continue;
        *triggers = (triggerEnt*)realloc(*triggers, ((size_t)*nTrig + 1) * sizeof(triggerEnt));
        (*triggers)[*nTrig].node = rq;
        (*triggers)[*nTrig].softmax = order[i];
        (*nTrig)++;
    }
}

static int triggerFor(triggerEnt* triggers, int nTrig, int node) {
    for (int i = 0; i < nTrig; i++) {
        if (triggers[i].node == node) return triggers[i].softmax;
    }
    return -1;
}

static void emitConv(builder* b, const onnx_node* n, int layer) {
    const char* x = n->inputs[0];
    const char* wname = n->inputs[1];
    const char* bname = n->inputs[2];
    const char* out = n->outputs[0];
    shp* osp = shapeOf(b, out);
    shp* xsp = shapeOf(b, x);
    const onnx_tensor* wt = onnxInit(b->onm, wname);
    const onnx_tensor* bt = onnxInit(b->onm, bname);
    if (osp == NULL || xsp == NULL || wt == NULL || bt == NULL) {
        fail(b, "conv missing shape or weight");
        return;
    }
    int Cout = (int)osp->d[1], H = (int)osp->d[2], W = (int)osp->d[3];
    int Cin = (int)xsp->d[1], Hin = (int)xsp->d[2], Win = (int)xsp->d[3];
    int k = (int)attrInt(n, "kernel_shape", 3);
    int stride = (int)attrInt(n, "strides", 1);
    int pad = (int)attrInt(n, "pads", 0);
    int group = (int)attrInt(n, "group", 1);
    int M = H * W;
    int flags = isSilu(b, out) ? 1 : 0;
    int obuf = newBuffer(b, (int64_t)M * Cout);
    setSlot(b, out, obuf, 0);
    const char* alias = siluAlias(b, out);
    if (alias != NULL) setSlot(b, alias, obuf, 0);

    slotEnt* xs = slotOf(b, x);
    if (xs == NULL) {
        fail(b, "conv input slot missing");
        return;
    }

    if (group == 1) {
        int K = Cin * k * k;
        int Kpad = (int)ceil16(K);
        int Npad = (int)ceil32(Cout);
        float* W2 = (float*)malloc((size_t)K * Cout * 4);
        if (k == 1) {
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < Cout; j++) W2[(int64_t)i * Cout + j] = wt->fdata[(int64_t)j * K + i];
            }
        } else {
            for (int co = 0; co < Cout; co++) {
                for (int ci = 0; ci < Cin; ci++) {
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {
                            int i = (ky * k + kx) * Cin + ci;
                            W2[(int64_t)i * Cout + co] = wt->fdata[((int64_t)co * Cin + ci) * k * k + ky * k + kx];
                        }
                    }
                }
            }
        }
        char wb[220];
        snprintf(wb, sizeof(wb), "%s_b16", wname);
        int wbuf = weightBlock16(b, wb, W2, K, Cout, Kpad, Npad);
        free(W2);
        float* bias = (float*)calloc((size_t)Npad, 4);
        memcpy(bias, bt->fdata, (size_t)Cout * 4);
        char bb[220];
        snprintf(bb, sizeof(bb), "%s_pad", bname);
        int bbuf = weightFp32(b, bb, bias, Npad);
        free(bias);

        if (k == 1) {
            int pc[] = {M, Kpad, Npad, M, Cout, 0, xs->off, Hin * Win, flags};
            int bufs[] = {xs->res, wbuf, obuf, bbuf};
            opEmit(b, "CONV1x1-Q16", bufs, 4, pc, 9, Npad / 32, M / 16, 1, layer);
        } else if (Cin % 16 == 0) {
            int Hp = Hin + 2 * pad;
            int Wp = Win + 2 * pad;
            int pc1[] = {Cin, Hin, Win, pad, xs->off, 0};
            int bufs1[] = {xs->res, b->padRes};
            opEmit(b, "Pad", bufs1, 2, pc1, 6, (int)ceildiv((int64_t)Cin * Hp * Wp, 256), 1, 1, layer);
            int pc2[] = {M, Kpad, Npad, M, Cout, 0, 0, Cin, Hp, Wp, W, stride, flags};
            int bufs2[] = {b->padRes, wbuf, obuf, bbuf};
            opEmit(b, "CONV3x3-Q16", bufs2, 4, pc2, 13, Npad / 32, M / 16, 1, layer);
        } else {
            int pc1[] = {Win, Hin, Cin, xs->off, Kpad, stride, pad, W, H};
            int bufs1[] = {xs->res, b->im2colRes};
            opEmit(b, "Im2col", bufs1, 2, pc1, 9, (int)ceildiv((int64_t)M * (Kpad / 4), 256), 1, 1, layer);
            int pc2[] = {M, Kpad, Npad, M, Cout, 0, 0, Kpad, flags};
            int bufs2[] = {b->im2colRes, wbuf, obuf, bbuf};
            opEmit(b, "CONV-GEMM-Q16", bufs2, 4, pc2, 9, Npad / 32, M / 16, 1, layer);
        }
    } else {
        char wb[220];
        snprintf(wb, sizeof(wb), "%s_dw", wname);
        int wbuf = weightFp16Raw(b, wb, wt->fdata, (int64_t)Cout * k * k);
        char bb[220];
        snprintf(bb, sizeof(bb), "%s_dw", bname);
        int bbuf = weightFp32(b, bb, bt->fdata, bt->count);
        int Hout = (Hin + 2 * pad - k) / stride + 1;
        int Wout = (Win + 2 * pad - k) / stride + 1;
        int pc[] = {Cout, Hin, Win, k, stride, pad, xs->off, 0, flags};
        int bufs[] = {xs->res, wbuf, bbuf, obuf};
        opEmit(b, "DWConv", bufs, 4, pc, 9, (int)ceildiv((int64_t)Cout * Hout * Wout, 256), 1, 1, layer);
    }
}

static void emitMaxpool(builder* b, const onnx_node* n, int layer) {
    const char* x = n->inputs[0];
    const char* out = n->outputs[0];
    shp* osp = shapeOf(b, out);
    shp* xsp = shapeOf(b, x);
    slotEnt* xs = slotOf(b, x);
    if (osp == NULL || xsp == NULL || xs == NULL) {
        fail(b, "maxpool missing shape");
        return;
    }
    int C = (int)osp->d[1], H = (int)osp->d[2], W = (int)osp->d[3];
    int Hin = (int)xsp->d[2], Win = (int)xsp->d[3];
    int k = (int)attrInt(n, "kernel_shape", 1);
    int stride = (int)attrInt(n, "strides", 1);
    int pad = (int)attrInt(n, "pads", 0);
    int obuf = newBuffer(b, (int64_t)C * H * W);
    setSlot(b, out, obuf, 0);
    int pc[] = {C, Hin, Win, k, stride, pad, xs->off, 0};
    int bufs[] = {xs->res, obuf};
    opEmit(b, "MaxPool", bufs, 2, pc, 8, (int)ceildiv((int64_t)C * H * W, 256), 1, 1, layer);
}

static void emitResize(builder* b, const onnx_node* n, int layer) {
    const char* x = n->inputs[0];
    const char* out = n->outputs[0];
    shp* osp = shapeOf(b, out);
    shp* xsp = shapeOf(b, x);
    slotEnt* xs = slotOf(b, x);
    if (osp == NULL || xsp == NULL || xs == NULL) {
        fail(b, "resize missing shape");
        return;
    }
    int C = (int)osp->d[1], H = (int)osp->d[2], W = (int)osp->d[3];
    int Hin = (int)xsp->d[2], Win = (int)xsp->d[3];
    int obuf = newBuffer(b, (int64_t)C * H * W);
    setSlot(b, out, obuf, 0);
    int pc[] = {C, Hin, Win, xs->off, 0};
    int bufs[] = {xs->res, obuf};
    opEmit(b, "Upsample2x", bufs, 2, pc, 5, (int)ceildiv((int64_t)C * H * W, 256), 1, 1, layer);
}

static void emitConcat(builder* b, const onnx_node* n, int layer) {
    const char* out = n->outputs[0];
    shp* osp = shapeOf(b, out);
    if (osp == NULL) {
        fail(b, "concat missing shape");
        return;
    }
    int64_t total = prodAll(*osp);
    int buf = newBuffer(b, total);
    setSlot(b, out, buf, 0);
    int zero = ensureZero(b);
    int axis = (int)attrInt(n, "axis", 0);
    if (axis < 0) axis += osp->n;

    if (axis == 1) {
        int64_t run = 0;
        for (int i = 0; i < n->nInputs; i++) {
            shp* s = shapeOf(b, n->inputs[i]);
            slotEnt* sl = slotOf(b, n->inputs[i]);
            if (s == NULL || sl == NULL) {
                fail(b, "concat input missing");
                return;
            }
            int64_t cnt = prodAll(*s);
            int pc[] = {(int)cnt, sl->off, 0, (int)run, 5, 0, 0, 0};
            int bufs[] = {sl->res, zero, buf};
            opEmit(b, "Elementwise", bufs, 3, pc, 8, (int)ceildiv(cnt, 256), 1, 1, layer);
            run += cnt;
        }
    } else {
        int C = (int)osp->d[1];
        int64_t Ltotal = prodFrom(*osp, 2);
        int srcBuf[16], srcOff[16], srcL[16];
        int64_t srcRun[16];
        int ns = 0;
        int64_t run = 0;
        for (int i = 0; i < n->nInputs; i++) {
            shp* s = shapeOf(b, n->inputs[i]);
            slotEnt* sl = slotOf(b, n->inputs[i]);
            if (s == NULL || sl == NULL) {
                fail(b, "concat input missing");
                return;
            }
            srcBuf[ns] = sl->res;
            srcOff[ns] = sl->off;
            srcRun[ns] = run;
            srcL[ns] = (int)prodFrom(*s, 2);
            run += srcL[ns];
            ns++;
        }
        if (ns <= 3) {
            while (ns < 3) {
                srcBuf[ns] = buf;
                srcOff[ns] = 0;
                srcRun[ns] = Ltotal;
                srcL[ns] = 0;
                ns++;
            }
            int pc[16];
            int npc = 0;
            pc[npc++] = C;
            pc[npc++] = (int)Ltotal;
            int bufs[4];
            int nb = 0;
            for (int i = 0; i < ns; i++) {
                pc[npc++] = (int)srcRun[i];
                pc[npc++] = srcL[i];
                pc[npc++] = srcOff[i];
                bufs[nb++] = srcBuf[i];
            }
            bufs[nb++] = buf;
            opEmit(b, "ConcatA", bufs, nb, pc, npc, (int)ceildiv((int64_t)C * Ltotal, 256), 1, 1, layer);
        } else {
            for (int c = 0; c < C; c++) {
                for (int i = 0; i < ns; i++) {
                    int pc[] = {srcL[i], srcOff[i] + c * srcL[i], 0, (int)(c * Ltotal + srcRun[i]), 5, 0, 0, 0};
                    int bufs[] = {srcBuf[i], zero, buf};
                    opEmit(b, "Elementwise", bufs, 3, pc, 8, (int)ceildiv(srcL[i], 256), 1, 1, layer);
                }
            }
        }
    }
}

static void emitElementwise(builder* b, const onnx_node* n, int layer) {
    const char* out = n->outputs[0];
    shp* osp = shapeOf(b, out);
    if (osp == NULL) {
        fail(b, "elementwise missing shape");
        return;
    }
    int64_t cnt = prodAll(*osp);
    int obuf = newBuffer(b, cnt);
    setSlot(b, out, obuf, 0);

    const char* aName = n->inputs[0];
    const char* bName = n->inputs[1];
    int abuf, bbuf, aoff, boff;
    int64_t asz, bsz;
    const onnx_tensor* ai = onnxInit(b->onm, aName);
    if (ai != NULL) {
        abuf = constBuffer(b, aName);
        aoff = 0;
        asz = ai->count;
    } else {
        slotEnt* sl = slotOf(b, aName);
        if (sl == NULL) {
            fail(b, "elementwise input missing");
            return;
        }
        abuf = sl->res;
        aoff = sl->off;
        asz = cnt;
    }
    const onnx_tensor* bi = onnxInit(b->onm, bName);
    if (bi != NULL) {
        bbuf = constBuffer(b, bName);
        boff = 0;
        bsz = bi->count;
    } else {
        slotEnt* sl = slotOf(b, bName);
        if (sl == NULL) {
            fail(b, "elementwise input missing");
            return;
        }
        bbuf = sl->res;
        boff = sl->off;
        bsz = cnt;
    }
    int op = strcmp(n->opType, "Add") == 0 ? 0 : (strcmp(n->opType, "Sub") == 0 ? 1 : 2);
    int bperiod = bsz < cnt ? (int)bsz : 0;
    int aperiod = asz < cnt ? (int)asz : 0;
    int pc[] = {(int)cnt, aoff, boff, 0, op, bperiod, aperiod, 0};
    int bufs[] = {abuf, bbuf, obuf};
    opEmit(b, "Elementwise", bufs, 3, pc, 8, (int)ceildiv(cnt, 256), 1, 1, layer);
}

static void emitSigmoid(builder* b, const onnx_node* n, int layer) {
    const char* x = n->inputs[0];
    const char* out = n->outputs[0];
    shp* osp = shapeOf(b, out);
    slotEnt* xs = slotOf(b, x);
    if (osp == NULL || xs == NULL) {
        fail(b, "sigmoid missing shape");
        return;
    }
    int64_t cnt = prodAll(*osp);
    int obuf = newBuffer(b, cnt);
    setSlot(b, out, obuf, 0);
    int zero = ensureZero(b);
    int pc[] = {(int)cnt, xs->off, 0, 0, 3, 0, 0, 0};
    int bufs[] = {xs->res, zero, obuf};
    opEmit(b, "Elementwise", bufs, 3, pc, 8, (int)ceildiv(cnt, 256), 1, 1, layer);
}

static void emitSlice(builder* b, const onnx_node* n, int layer) {
    (void)layer;
    const char* x = n->inputs[0];
    const char* out = n->outputs[0];
    shp* xsp = shapeOf(b, x);
    const onnx_tensor* st = onnxInit(b->onm, n->inputs[1]);
    slotEnt* xs = slotOf(b, x);
    if (xsp == NULL || st == NULL || xs == NULL) {
        fail(b, "slice missing shape");
        return;
    }
    const onnx_tensor* ax = n->nInputs > 3 ? onnxInit(b->onm, n->inputs[3]) : NULL;
    int64_t off = 0;
    for (int i = 0; i < (int)st->count; i++) {
        int64_t s = st->idata[i];
        int64_t a = (ax && ax->idata) ? ax->idata[i] : 0;
        if (a < 0) a += xsp->n;
        int64_t stride = 1;
        for (int q = (int)a + 1; q < xsp->n; q++) stride *= xsp->d[q];
        off += s * stride;
    }
    setSlot(b, out, xs->res, xs->off + (int)off);
}

static void emitReshape(builder* b, const onnx_node* n) {
    slotEnt* xs = slotOf(b, n->inputs[0]);
    if (xs == NULL) {
        fail(b, "reshape input missing");
        return;
    }
    setSlot(b, n->outputs[0], xs->res, xs->off);
}

static void emitSplit(builder* b, const onnx_node* n) {
    const char* x = n->inputs[0];
    shp* xsp = shapeOf(b, x);
    slotEnt* xs = slotOf(b, x);
    if (xsp == NULL || xs == NULL) {
        fail(b, "split missing shape");
        return;
    }
    int64_t spatial = prodFrom(*xsp, 2);
    int64_t run = 0;
    for (int i = 0; i < n->nOutputs; i++) {
        shp* os = shapeOf(b, n->outputs[i]);
        setSlot(b, n->outputs[i], xs->res, xs->off + (int)run * (int)spatial);
        run += os->d[1];
    }
}

static void emitAttention(builder* b, const onnx_node* softmax, int layer) {
    const onnx_node* m1 = nodeAt(b, prodNode(b, softmax->inputs[0]));
    const onnx_node* transq = nodeAt(b, prodNode(b, m1->inputs[0]));
    const onnx_node* qmul = nodeAt(b, prodNode(b, transq->inputs[0]));
    const char* q_raw = qmul->inputs[0];
    const onnx_tensor* scaleT = onnxInit(b->onm, qmul->inputs[1]);
    const onnx_node* split_q = nodeAt(b, prodNode(b, q_raw));
    const onnx_node* reshape_qkv = nodeAt(b, prodNode(b, split_q->inputs[0]));
    const char* qkv = reshape_qkv->inputs[0];

    const onnx_node* t1 = consumer(b, softmax->outputs[0], "Transpose");
    const onnx_node* m2 = consumer(b, t1->outputs[0], "MatMul");
    const char* v_name = m2->inputs[0];
    const onnx_node* reshape_out = consumer(b, m2->outputs[0], "Reshape");
    const char* out_name = reshape_out->outputs[0];
    const onnx_node* reshape_v = consumer(b, v_name, "Reshape");
    const char* vcat_name = reshape_v->outputs[0];

    shp* qsh = shapeOf(b, split_q->outputs[0]);
    shp* vsh = shapeOf(b, v_name);
    slotEnt* qkvSlot = slotOf(b, qkv);
    if (qsh == NULL || vsh == NULL || qkvSlot == NULL || scaleT == NULL) {
        fail(b, "attention missing shape");
        return;
    }
    int heads = (int)qsh->d[1], qdim = (int)qsh->d[2], P = (int)qsh->d[3];
    int vdim = (int)vsh->d[2];
    int D = qdim + qdim + vdim;

    int qkv_buf = qkvSlot->res;
    int qkv_off = qkvSlot->off;
    int outbuf = newBuffer(b, (int64_t)heads * vdim * P);
    setSlot(b, out_name, outbuf, 0);
    int vcat = newBuffer(b, (int64_t)heads * vdim * P);
    setSlot(b, vcat_name, vcat, 0);
    int qs = newBuffer(b, (int64_t)qdim * P);
    int qT = newBuffer(b, (int64_t)P * qdim);
    int scores = newBuffer(b, (int64_t)P * P);
    int scoresT = newBuffer(b, (int64_t)P * P);
    int zero = ensureZero(b);
    int sbits;
    memcpy(&sbits, &scaleT->fdata[0], 4);

    for (int h = 0; h < heads; h++) {
        int qbase = qkv_off + h * D * P;
        int kbase = qbase + qdim * P;
        int vbase = qbase + (qdim + qdim) * P;

        int pc1[] = {qdim * P, qbase, 0, 0, 4, 0, 0, sbits};
        int bufs1[] = {qkv_buf, zero, qs};
        opEmit(b, "Elementwise", bufs1, 3, pc1, 8, (int)ceildiv((int64_t)qdim * P, 256), 1, 1, layer);

        int pc2[] = {qdim, P, 0, 0};
        int bufs2[] = {qs, qT};
        opEmit(b, "Transpose2D", bufs2, 2, pc2, 4, (int)ceildiv(P, 16), (int)ceildiv(qdim, 16), 1, layer);

        int pc3[] = {P, P, qdim, qdim, P, P, P, 0, 0, kbase, 0};
        int bufs3[] = {qT, qkv_buf, scores, zero};
        opEmit(b, "CONV-GEMM-FF32", bufs3, 4, pc3, 11, (int)ceildiv(P, 16), (int)ceildiv(P, 16), 1, layer);

        int pc4[] = {P, P, 0};
        int bufs4[] = {scores};
        opEmit(b, "Softmax", bufs4, 1, pc4, 3, P, 1, 1, layer);

        int pc5[] = {P, P, 0, 0};
        int bufs5[] = {scores, scoresT};
        opEmit(b, "Transpose2D", bufs5, 2, pc5, 4, (int)ceildiv(P, 16), (int)ceildiv(P, 16), 1, layer);

        int pc6[] = {vdim, P, P, P, P, P, P, h * vdim * P, vbase, 0, 0};
        int bufs6[] = {qkv_buf, scoresT, outbuf, zero};
        opEmit(b, "CONV-GEMM-FF32", bufs6, 4, pc6, 11, (int)ceildiv(P, 16), (int)ceildiv(vdim, 16), 1, layer);

        int pc7[] = {vdim * P, vbase, 0, h * vdim * P, 5, 0, 0, 0};
        int bufs7[] = {qkv_buf, zero, vcat};
        opEmit(b, "Elementwise", bufs7, 3, pc7, 8, (int)ceildiv((int64_t)vdim * P, 256), 1, 1, layer);
    }

    addName(&b->handled, &b->nHandled, softmax->name);
    addName(&b->handled, &b->nHandled, m1->name);
    addName(&b->handled, &b->nHandled, transq->name);
    addName(&b->handled, &b->nHandled, qmul->name);
    addName(&b->handled, &b->nHandled, split_q->name);
    addName(&b->handled, &b->nHandled, reshape_qkv->name);
    addName(&b->handled, &b->nHandled, t1->name);
    addName(&b->handled, &b->nHandled, m2->name);
    addName(&b->handled, &b->nHandled, reshape_out->name);
    addName(&b->handled, &b->nHandled, reshape_v->name);
}

static void build(builder* b) {
    graph_plan* p = b->plan;
    detectSilu(b);
    const char* pred = findPred(b);
    if (pred == NULL) {
        fail(b, "cannot locate prediction tensor");
        return;
    }
    int* order = NULL;
    int nOrder = 0;
    collect(b, pred, &order, &nOrder);

    triggerEnt* triggers = NULL;
    int nTrig = 0;
    findAttention(b, order, nOrder, &triggers, &nTrig);

    int64_t maxIm2col = 0;
    int64_t maxPad = 0;
    for (int i = 0; i < nOrder; i++) {
        const onnx_node* n = &b->onm->nodes[order[i]];
        if (strcmp(n->opType, "Conv") != 0) continue;
        int kh = (int)attrInt(n, "kernel_shape", 3);
        int group = (int)attrInt(n, "group", 1);
        if (kh != 3 || group != 1) continue;
        shp* xsp = shapeOf(b, n->inputs[0]);
        shp* osp = shapeOf(b, n->outputs[0]);
        if (xsp == NULL || osp == NULL) continue;
        int pad = (int)attrInt(n, "pads", 0);
        int64_t cin = xsp->d[1], hin = xsp->d[2], win = xsp->d[3];
        if (cin % 16 == 0) {
            int64_t v = cin * (hin + 2 * pad) * (win + 2 * pad);
            if (v > maxPad) maxPad = v;
        } else {
            int64_t v = ceil16(cin * 9) * osp->d[2] * osp->d[3];
            if (v > maxIm2col) maxIm2col = v;
        }
    }
    b->im2colRes = newBuffer(b, maxIm2col > 1 ? maxIm2col : 1);
    b->padRes = newBuffer(b, maxPad > 1 ? maxPad : 1);

    const char* images = b->onm->inputName;
    shp* ish = shapeOf(b, images);
    if (ish == NULL) {
        fail(b, "missing input shape");
        return;
    }
    int ibuf = newBuffer(b, prodAll(*ish));
    setSlot(b, images, ibuf, 0);

    for (int idx = 0; idx < nOrder; idx++) {
        int ni = order[idx];
        const onnx_node* n = &b->onm->nodes[ni];
        if (isHandled(b, n->name) || isSkip(b, n->name)) continue;
        int sm = triggerFor(triggers, nTrig, ni);
        if (sm >= 0) {
            emitAttention(b, &b->onm->nodes[sm], idx);
            if (b->failed) return;
            continue;
        }
        const char* t = n->opType;
        if (strcmp(t, "Conv") == 0) emitConv(b, n, idx);
        else if (strcmp(t, "MaxPool") == 0) emitMaxpool(b, n, idx);
        else if (strcmp(t, "Resize") == 0) emitResize(b, n, idx);
        else if (strcmp(t, "Concat") == 0) emitConcat(b, n, idx);
        else if (strcmp(t, "Add") == 0 || strcmp(t, "Sub") == 0 || strcmp(t, "Mul") == 0) emitElementwise(b, n, idx);
        else if (strcmp(t, "Sigmoid") == 0) emitSigmoid(b, n, idx);
        else if (strcmp(t, "Slice") == 0) emitSlice(b, n, idx);
        else if (strcmp(t, "Reshape") == 0) emitReshape(b, n);
        else if (strcmp(t, "Split") == 0) emitSplit(b, n);
        if (b->failed) return;
    }
    free(order);
    free(triggers);

    shp* psh = shapeOf(b, pred);
    slotEnt* pslot = slotOf(b, pred);
    if (psh == NULL || pslot == NULL) {
        fail(b, "missing prediction output");
        return;
    }
    p->inputRes = ibuf;
    p->outputRes = pslot->res;
    p->outChannels = (int)psh->d[1];
    p->outAnchors = (int)psh->d[2];
    p->outBoxes = 4;
    p->outClasses = (int)psh->d[1] - 4;
    p->outTopk = 300;

    p->tensorCount = b->nSlot;
    p->tensorNames = (char**)calloc((size_t)b->nSlot, sizeof(char*));
    p->tensorRes = (int*)calloc((size_t)b->nSlot, sizeof(int));
    p->tensorOff = (int*)calloc((size_t)b->nSlot, sizeof(int));
    for (int i = 0; i < b->nSlot; i++) {
        p->tensorNames[i] = strdup(b->slots[i].name);
        p->tensorRes[i] = b->slots[i].res;
        p->tensorOff[i] = b->slots[i].off;
    }
}

int compileOnnx(const char* onnxPath, graph_plan* plan) {
    memset(plan, 0, sizeof(*plan));
    g_error[0] = 0;
    onnx_model m;
    if (onnxLoad(onnxPath, &m) != 0) {
        snprintf(g_error, sizeof(g_error), "cannot read %s", onnxPath);
        return -1;
    }
    builder b;
    memset(&b, 0, sizeof(b));
    b.onm = &m;
    b.plan = plan;
    b.zeroRes = -1;
    buildProds(&b);
    inferShapes(&b);
    build(&b);
    onnxFree(&m);
    free(b.shapes);
    free(b.slots);
    free(b.prods);
    free(b.skip);
    free(b.handled);
    free(b.silu);
    free(b.consts);
    if (b.failed) {
        snprintf(g_error, sizeof(g_error), "%s", b.err);
        freePlan(plan);
        return -1;
    }
    return 0;
}

void freePlan(graph_plan* plan) {
    for (int i = 0; i < plan->resCount; i++) free(plan->res[i].name);
    for (int i = 0; i < plan->tensorCount; i++) free(plan->tensorNames[i]);
    free(plan->res);
    free(plan->wblob);
    free(plan->ops);
    free(plan->tensorNames);
    free(plan->tensorRes);
    free(plan->tensorOff);
    memset(plan, 0, sizeof(*plan));
}
