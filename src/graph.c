#include "graph.h"
#include "compile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int graphLoad(yolo_graph* g, const char* onnxPath, const char* shaderDir) {
    memset(g, 0, sizeof(*g));

    graph_plan plan;
    if (compileOnnx(onnxPath, &plan) != 0) {
        fprintf(stderr, "Error: %s\n", compileLastError());
        return -1;
    }

    g->sess = createSession();
    if (shaderDir != NULL) setShaderRootDir(shaderDir);

    int nAct = 0;
    int nWgt = 0;
    for (int i = 0; i < plan.resCount; i++) {
        if (plan.res[i].isWeight) nWgt++;
        else nAct++;
    }
    g->bufferCount = nAct;
    g->weightCount = nWgt;
    g->buffers = (buffer*)calloc((size_t)nAct, sizeof(buffer));
    g->bufferNames = (char**)calloc((size_t)nAct, sizeof(char*));
    g->weights = (buffer*)calloc((size_t)nWgt, sizeof(buffer));
    g->weightNames = (char**)calloc((size_t)nWgt, sizeof(char*));

    int* resKind = (int*)malloc(sizeof(int) * (size_t)plan.resCount);
    int* resIndex = (int*)malloc(sizeof(int) * (size_t)plan.resCount);
    int ai = 0;
    int wi = 0;
    for (int i = 0; i < plan.resCount; i++) {
        graph_res* r = &plan.res[i];
        if (r->isWeight) {
            g->weightNames[wi] = strdup(r->name);
            g->weights[wi] = createBufferNamed(g->sess.dev.device, g->sess.dev.physicalDevice,
                                               plan.wblob + r->off, r->nbytes, MEMORY_VRAM, r->name);
            resKind[i] = 1;
            resIndex[i] = wi;
            wi++;
        } else {
            int64_t bytes = r->size * 4;
            if (bytes <= 0) bytes = 4;
            void* zero = calloc(1, (size_t)bytes);
            int memType = (i == plan.outputRes) ? MEMORY_RAM_CACHED : MEMORY_VRAM;
            g->bufferNames[ai] = strdup(r->name);
            g->buffers[ai] = createBufferNamed(g->sess.dev.device, g->sess.dev.physicalDevice,
                                               zero, bytes, memType, r->name);
            free(zero);
            resKind[i] = 0;
            resIndex[i] = ai;
            ai++;
        }
    }

    createTransferAndCopy(g->sess.dev.device, g->sess.dev.queue, g->buffers, g->bufferCount);
    createTransferAndCopy(g->sess.dev.device, g->sess.dev.queue, g->weights, g->weightCount);

    g->inputIndex = resIndex[plan.inputRes];
    g->outputIndex = resIndex[plan.outputRes];
    for (int i = 0; i < g->bufferCount; i++) {
        if (i != g->inputIndex) releaseStaging(g->sess.dev.device, &g->buffers[i]);
    }
    for (int i = 0; i < g->weightCount; i++) {
        releaseStaging(g->sess.dev.device, &g->weights[i]);
    }

    g->outChannels = plan.outChannels;
    g->outAnchors = plan.outAnchors;
    g->outBoxes = plan.outBoxes;
    g->outClasses = plan.outClasses;
    g->outTopk = plan.outTopk;

    g->onnxCount = plan.tensorCount;
    g->onnxNames = (char**)calloc((size_t)plan.tensorCount, sizeof(char*));
    g->onnxBuf = (int*)calloc((size_t)plan.tensorCount, sizeof(int));
    g->onnxOff = (int*)calloc((size_t)plan.tensorCount, sizeof(int));
    for (int i = 0; i < plan.tensorCount; i++) {
        g->onnxNames[i] = strdup(plan.tensorNames[i]);
        g->onnxBuf[i] = resIndex[plan.tensorRes[i]];
        g->onnxOff[i] = plan.tensorOff[i];
    }

    g->opCount = plan.opCount;
    g->ops = (operation*)calloc((size_t)plan.opCount, sizeof(operation));
    for (int i = 0; i < plan.opCount; i++) {
        graph_op* o = &plan.ops[i];
        operation* op = &g->ops[i];
        snprintf(op->shader, sizeof(op->shader), "%s", o->shader);
        op->layer = o->layer;
        op->bufferCount = o->bufferCount;
        for (int b = 0; b < o->bufferCount; b++) {
            int ri = o->buffers[b];
            op->buffers[b] = (resKind[ri] == 0) ? g->buffers[resIndex[ri]] : g->weights[resIndex[ri]];
        }
        op->pushConstantCount = o->pushConstantCount;
        for (int p = 0; p < o->pushConstantCount; p++) op->pushConstants[p] = o->pushConstants[p];
        op->dispatchX = o->dispatchX;
        op->dispatchY = o->dispatchY;
        op->dispatchZ = o->dispatchZ;
    }

    free(resKind);
    free(resIndex);
    freePlan(&plan);

    VkCommandPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = 0;
    vkCreateCommandPool(g->sess.dev.device, &poolInfo, NULL, &g->uploadPool);
    VkCommandBufferAllocateInfo ai2 = {0};
    ai2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai2.commandPool = g->uploadPool;
    ai2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai2.commandBufferCount = 1;
    vkAllocateCommandBuffers(g->sess.dev.device, &ai2, &g->uploadCmd);

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
