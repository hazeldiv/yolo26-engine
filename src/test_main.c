#include "yolo.h"
#include "graph.h"
#include "compile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static double nowMs(void) {
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static void fillRandom(float* p, int n) {
    unsigned int s = 12345u;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (float)((s >> 8) & 0xFFFF) / 65535.0f;
    }
}

static void writeFloats(const char* path, const float* p, int n) {
    FILE* f = fopen(path, "wb");
    if (f == NULL) return;
    fwrite(p, sizeof(float), (size_t)n, f);
    fclose(f);
}

static void dumpPlan(const char* onnxPath, const char* outFile) {
    graph_plan plan;
    if (compileOnnx(onnxPath, &plan) != 0) {
        fprintf(stderr, "compile failed: %s\n", compileLastError());
        exit(1);
    }
    FILE* f = fopen(outFile, "w");
    for (int i = 0; i < plan.resCount; i++) {
        if (!plan.res[i].isWeight) fprintf(f, "BUF %s %lld\n", plan.res[i].name, (long long)plan.res[i].size);
    }
    for (int i = 0; i < plan.resCount; i++) {
        if (plan.res[i].isWeight) fprintf(f, "WGT %s %lld\n", plan.res[i].name, (long long)plan.res[i].nbytes);
    }
    for (int i = 0; i < plan.opCount; i++) {
        graph_op* o = &plan.ops[i];
        fprintf(f, "OP %s", o->shader);
        for (int b = 0; b < o->bufferCount; b++) fprintf(f, " %s", plan.res[o->buffers[b]].name);
        fprintf(f, " |");
        for (int p = 0; p < o->pushConstantCount; p++) fprintf(f, " %d", o->pushConstants[p]);
        fprintf(f, " | %d %d %d\n", o->dispatchX, o->dispatchY, o->dispatchZ);
    }
    fprintf(f, "OUT %s %d %d %d %d %d\n", plan.res[plan.outputRes].name, plan.outChannels,
            plan.outAnchors, plan.outBoxes, plan.outClasses, plan.outTopk);
    fclose(f);
    char wbin[600];
    snprintf(wbin, sizeof(wbin), "%s.wbin", outFile);
    FILE* wf = fopen(wbin, "wb");
    if (wf != NULL) {
        fwrite(plan.wblob, 1, (size_t)plan.wblobLen, wf);
        fclose(wf);
    }
    freePlan(&plan);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test.exe <model.onnx> [--shader <dir>] [--bench N] [--input <raw>] [--dump <prefix>]\n");
        return 1;
    }
    const char* modelDir = argv[1];
    const char* shaderDir = "shader";
    const char* inputFile = NULL;
    const char* dumpPrefix = NULL;
    const char* dumpTensor = NULL;
    const char* dumpPlanFile = NULL;
    int bench = 0;
    int timing = 0;
    int meminfo = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--shader") == 0 && i + 1 < argc) shaderDir = argv[++i];
        else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) bench = atoi(argv[++i]);
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) inputFile = argv[++i];
        else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dumpPrefix = argv[++i];
        else if (strcmp(argv[i], "--dump-tensor") == 0 && i + 1 < argc) dumpTensor = argv[++i];
        else if (strcmp(argv[i], "--dump-plan") == 0 && i + 1 < argc) dumpPlanFile = argv[++i];
        else if (strcmp(argv[i], "--timing") == 0) timing = 1;
        else if (strcmp(argv[i], "--meminfo") == 0) meminfo = 1;
    }

    if (dumpPlanFile != NULL) {
        dumpPlan(modelDir, dumpPlanFile);
        return 0;
    }

    yolo_graph g;
    if (graphLoad(&g, modelDir, shaderDir) != 0) {
        fprintf(stderr, "graphLoad failed\n");
        return 1;
    }
    if (meminfo) {
        dumpMemoryInfo(g.sess.dev.physicalDevice);
    }

    int inElems = 3 * 640 * 640;
    float* input = (float*)malloc(sizeof(float) * (size_t)inElems);
    if (inputFile != NULL) {
        FILE* f = fopen(inputFile, "rb");
        if (f == NULL) {
            fprintf(stderr, "cannot open input %s\n", inputFile);
            return 1;
        }
        if (fread(input, sizeof(float), (size_t)inElems, f) != (size_t)inElems) {
            fprintf(stderr, "input file too small\n");
            fclose(f);
            return 1;
        }
        fclose(f);
    } else {
        fillRandom(input, inElems);
    }

    graphRun(&g, input);

    if (timing) {
        setTimingEnabled(1);
        graphRunLogged(&g, input, "frame", 0);
        closeTimingLog();
    }

    int outElems = g.outChannels * g.outAnchors;
    float* raw = (float*)malloc(sizeof(float) * (size_t)outElems);
    graphReadOutput(&g, raw);

    if (dumpPrefix != NULL) {
        char p[512];
        snprintf(p, sizeof(p), "%s_in.f32", dumpPrefix);
        writeFloats(p, input, inElems);
        snprintf(p, sizeof(p), "%s_out.f32", dumpPrefix);
        writeFloats(p, raw, outElems);
    }

    if (dumpTensor != NULL) {
        int cap = 32 * 1024 * 1024;
        float* tb = (float*)malloc(sizeof(float) * (size_t)cap);
        int cnt = 0;
        if (graphReadTensor(&g, dumpTensor, tb, &cnt) == 0) {
            char fn[600];
            snprintf(fn, sizeof(fn), "%s.f32", dumpTensor);
            for (char* c = fn; *c; c++) {
                if (*c == '/' || *c == '\\' || *c == ':') *c = '_';
            }
            writeFloats(fn, tb, cnt);
            printf("dumped tensor %s (%d elems) -> %s\n", dumpTensor, cnt, fn);
        } else {
            fprintf(stderr, "tensor not found: %s\n", dumpTensor);
        }
        free(tb);
    }

    yolo_detection dets[300];
    int n = yoloDecode(raw, g.outChannels, g.outAnchors, g.outBoxes, g.outClasses,
                       g.outTopk, dets, 300);
    printf("detections: %d\n", n);
    for (int i = 0; i < n && i < 10; i++) {
        printf("  [%2d] %.2f %.2f %.2f %.2f score=%.4f cls=%d\n",
               i, dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2, dets[i].score, dets[i].cls);
    }

    if (bench > 0) {
        double t0 = nowMs();
        for (int i = 0; i < bench; i++) {
            graphRun(&g, input);
        }
        double t1 = nowMs();
        double total = t1 - t0;
        double avg = total / bench;
        printf("bench: %d frames, total %.2f ms, avg %.3f ms, %.1f FPS\n", bench, total, avg, 1000.0 / avg);

        yolo_detection bdets[300];
        int bn = 0;
        double t2 = nowMs();
        for (int i = 0; i < bench; i++) {
            graphRun(&g, input);
            graphReadOutput(&g, raw);
            bn = yoloDecode(raw, g.outChannels, g.outAnchors, g.outBoxes, g.outClasses,
                            g.outTopk, bdets, 300);
        }
        double t3 = nowMs();
        double full = t3 - t2;
        printf("bench-full: %d frames, total %.2f ms, avg %.3f ms, %.1f FPS (dets=%d)\n",
               bench, full, full / bench, 1000.0 / (full / bench), bn);
    }

    free(raw);
    free(input);
    graphFree(&g);
    return 0;
}
