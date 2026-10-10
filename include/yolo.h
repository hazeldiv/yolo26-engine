#ifndef yolo_h
#define yolo_h

#ifdef _WIN32
#define YOLO_API __declspec(dllexport)
#else
#define YOLO_API
#endif

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int cls;
} yolo_detection;

typedef struct yolo_model yolo_model;

YOLO_API int yolo_open(const char* model_dir, const char* shader_dir, yolo_model** out);
YOLO_API int yolo_infer(yolo_model* m, const float* rgb, yolo_detection* dets, int max_dets, int* n_dets);
YOLO_API void yolo_close(yolo_model* m);
YOLO_API const char* yolo_last_error(void);

int yoloDecode(const float* pred, int channels, int anchors, int boxes, int classes,
               int topk, yolo_detection* dets, int maxDets);

#endif
