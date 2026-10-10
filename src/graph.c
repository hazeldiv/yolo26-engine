#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static buffer* findBuffer(yolo_graph* g, const char* name) {
    for (int i = 0; i < g->bufferCount; i++) {
        if (strcmp(g->bufferNames[i], name) == 0) return &g->buffers[i];
    }
    for (int i = 0; i < g->weightCount; i++) {
        if (strcmp(g->weightNames[i], name) == 0) return &g->weights[i];
    }
    return NULL;
}

static unsigned char* readFile(const char* path, long* sizeOut) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* buf = (unsigned char*)malloc((size_t)len);
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *sizeOut = len;
    return buf;
}

int graphLoad(yolo_graph* g, const char* modelDir, const char* shaderDir) {
    memset(g, 0, sizeof(*g));
    char path[512];
    snprintf(path, sizeof(path), "%s/graph.json", modelDir);
    json_value* root = json_parse_file(path);
    if (root == NULL) {
        fprintf(stderr, "Error: cannot parse %s\n", path);
        return -1;
    }

    g->sess = createSession();
    if (shaderDir != NULL) setShaderRootDir(shaderDir);

    json_value* bufs = json_get(root, "buffers");
    g->bufferCount = bufs->count;
    g->buffers = (buffer*)calloc(g->bufferCount, sizeof(buffer));
    g->bufferNames = (char**)calloc(g->bufferCount, sizeof(char*));

    json_value* outp = json_get(root, "output");
    const char* outName = json_get_str(outp, "buffer", "t0");

    char wpath[512];
    snprintf(wpath, sizeof(wpath), "%s/weights.bin", modelDir);
    long wsize = 0;
    unsigned char* wblob = readFile(wpath, &wsize);

    json_value* weights = json_get(root, "weights");
    g->weightCount = weights->count;
    g->weights = (buffer*)calloc(g->weightCount, sizeof(buffer));
    g->weightNames = (char**)calloc(g->weightCount, sizeof(char*));

    for (int i = 0; i < g->bufferCount; i++) {
        json_value* b = &bufs->items[i];
        const char* name = json_get_str(b, "name", "t");
        int size = json_get_int(b, "size", 0);
        g->bufferNames[i] = strdup(name);
        int64_t bytes = (int64_t)size * 4;
        if (bytes <= 0) bytes = 4;
        void* zero = calloc(1, (size_t)bytes);
        int memType = (strcmp(name, outName) == 0) ? MEMORY_RAM_CACHED : MEMORY_VRAM;
        g->buffers[i] = createBufferNamed(g->sess.dev.device, g->sess.dev.physicalDevice,
                                          zero, bytes, memType, name);
        free(zero);
    }

    for (int i = 0; i < g->weightCount; i++) {
        json_value* w = &weights->items[i];
        const char* name = json_get_str(w, "name", "w");
        long off = (long)json_get_num(w, "offset", 0);
        long nbytes = (long)json_get_num(w, "nbytes", 0);
        g->weightNames[i] = strdup(name);
        g->weights[i] = createBufferNamed(g->sess.dev.device, g->sess.dev.physicalDevice,
                                          wblob + off, nbytes, MEMORY_VRAM, name);
    }

    createTransferAndCopy(g->sess.dev.device, g->sess.dev.queue, g->buffers, g->bufferCount);
    createTransferAndCopy(g->sess.dev.device, g->sess.dev.queue, g->weights, g->weightCount);

    json_value* in = json_get(root, "input");
    const char* inName = json_get_str(in, "buffer", "t0");
    g->inputIndex = -1;
    for (int i = 0; i < g->bufferCount; i++) {
        if (strcmp(g->bufferNames[i], inName) == 0) g->inputIndex = i;
        if (g->inputIndex != i) releaseStaging(g->sess.dev.device, &g->buffers[i]);
    }
    for (int i = 0; i < g->weightCount; i++) {
        releaseStaging(g->sess.dev.device, &g->weights[i]);
    }

    json_value* outp2 = json_get(root, "output");
    g->outputIndex = -1;
    for (int i = 0; i < g->bufferCount; i++) {
        if (strcmp(g->bufferNames[i], outName) == 0) g->outputIndex = i;
    }
    g->outChannels = json_get_int(outp2, "channels", 0);
    g->outAnchors = json_get_int(outp2, "anchors", 0);
    g->outBoxes = json_get_int(outp2, "boxes", 4);
    g->outClasses = json_get_int(outp2, "classes", 0);
    g->outTopk = json_get_int(outp2, "topk", 300);

    json_value* names = json_get(root, "names");
    if (names != NULL && names->type == JSON_OBJECT) {
        g->onnxCount = names->count;
        g->onnxNames = (char**)calloc(g->onnxCount, sizeof(char*));
        g->onnxBuf = (int*)calloc(g->onnxCount, sizeof(int));
        g->onnxOff = (int*)calloc(g->onnxCount, sizeof(int));
        for (int i = 0; i < g->onnxCount; i++) {
            g->onnxNames[i] = strdup(names->keys[i]);
            json_value* v = &names->values[i];
            const char* bn = v->items[0].string;
            g->onnxBuf[i] = -1;
            for (int b = 0; b < g->bufferCount; b++) {
                if (strcmp(g->bufferNames[b], bn) == 0) g->onnxBuf[i] = b;
            }
            g->onnxOff[i] = (int)v->items[1].number;
        }
    }

    json_value* ops = json_get(root, "ops");
    g->opCount = ops->count;
    g->ops = (operation*)calloc(g->opCount, sizeof(operation));
    for (int i = 0; i < g->opCount; i++) {
        json_value* o = &ops->items[i];
        operation* op = &g->ops[i];
        snprintf(op->shader, sizeof(op->shader), "%s", json_get_str(o, "shader", ""));
        op->layer = json_get_int(o, "layer", -1);
        json_value* bl = json_get(o, "buffers");
        op->bufferCount = bl->count;
        for (int b = 0; b < bl->count; b++) {
            buffer* buf = findBuffer(g, bl->items[b].string);
            if (buf == NULL) {
                fprintf(stderr, "Error: op %d references unknown buffer '%s'\n", i, bl->items[b].string);
                return -1;
            }
            op->buffers[b] = *buf;
        }
        json_value* pc = json_get(o, "pc");
        op->pushConstantCount = pc->count;
        for (int p = 0; p < pc->count; p++) {
            op->pushConstants[p] = (int)pc->items[p].number;
        }
        json_value* dp = json_get(o, "dispatch");
        op->dispatchX = (int)dp->items[0].number;
        op->dispatchY = (int)dp->items[1].number;
        op->dispatchZ = (int)dp->items[2].number;
    }

    VkCommandPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = 0;
    vkCreateCommandPool(g->sess.dev.device, &poolInfo, NULL, &g->uploadPool);
    VkCommandBufferAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = g->uploadPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(g->sess.dev.device, &ai, &g->uploadCmd);

    json_free(root);
    free(wblob);
    return 0;
}

void graphRun(yolo_graph* g, const float* input) {
    buffer* in = &g->buffers[g->inputIndex];
    writeStaging(g->sess.dev.device, in, input, in->size);

    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(g->uploadCmd, 0);
    vkBeginCommandBuffer(g->uploadCmd, &bi);
    VkBufferCopy region = {0};
    region.size = in->size;
    vkCmdCopyBuffer(g->uploadCmd, in->stagingBuffer, in->buffer, 1, &region);
    vkEndCommandBuffer(g->uploadCmd);
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g->uploadCmd;
    vkQueueSubmit(g->sess.dev.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(g->sess.dev.queue);

    execute(g->sess, g->ops, g->opCount);
}

void graphRunLogged(yolo_graph* g, const float* input, const char* phase, int token) {
    buffer* in = &g->buffers[g->inputIndex];
    writeStaging(g->sess.dev.device, in, input, in->size);
    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(g->uploadCmd, 0);
    vkBeginCommandBuffer(g->uploadCmd, &bi);
    VkBufferCopy region = {0};
    region.size = in->size;
    vkCmdCopyBuffer(g->uploadCmd, in->stagingBuffer, in->buffer, 1, &region);
    vkEndCommandBuffer(g->uploadCmd);
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g->uploadCmd;
    vkQueueSubmit(g->sess.dev.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(g->sess.dev.queue);
    executeLogged(g->sess, g->ops, g->opCount, phase, token);
}

void graphReadOutput(yolo_graph* g, float* out) {
    readBuffer(g->sess.dev.device, g->sess.dev.physicalDevice, g->sess.dev.queue,
               g->buffers[g->outputIndex], out);
}

int graphReadTensor(yolo_graph* g, const char* onnxName, float* out, int* count) {
    for (int i = 0; i < g->onnxCount; i++) {
        if (strcmp(g->onnxNames[i], onnxName) == 0) {
            int b = g->onnxBuf[i];
            if (b < 0) return -1;
            readBuffer(g->sess.dev.device, g->sess.dev.physicalDevice, g->sess.dev.queue,
                       g->buffers[b], out);
            *count = (int)(g->buffers[b].size / 4) - g->onnxOff[i];
            return 0;
        }
    }
    return -1;
}

void graphFree(yolo_graph* g) {
    if (g->uploadPool != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(g->sess.dev.device);
        vkDestroyCommandPool(g->sess.dev.device, g->uploadPool, NULL);
    }
    for (int i = 0; i < g->bufferCount; i++) {
        destroyBuffer(g->sess.dev.device, g->buffers[i]);
        free(g->bufferNames[i]);
    }
    for (int i = 0; i < g->weightCount; i++) {
        destroyBuffer(g->sess.dev.device, g->weights[i]);
        free(g->weightNames[i]);
    }
    free(g->buffers);
    free(g->bufferNames);
    free(g->weights);
    free(g->weightNames);
    free(g->ops);
    for (int i = 0; i < g->onnxCount; i++) free(g->onnxNames[i]);
    free(g->onnxNames);
    free(g->onnxBuf);
    free(g->onnxOff);
    destroySession(g->sess);
    memset(g, 0, sizeof(*g));
}
