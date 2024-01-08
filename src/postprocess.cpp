#include "postprocess.hpp"
#include <algorithm>
#include <math.h>

static inline float fast_exp(float x)
{
    // return exp(x);
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}

/****** detr ****/
DETR::DETR()
{
}

DETR::~DETR()
{
}

int DETR::GetDETRDetectResult(float **pBlob, std::vector<float> &DetectiontRects)
{
    float *pred_logits = (float *)pBlob[0];
    float *pred_boxes = (float *)pBlob[1];

    float softmaxsum = 0.0;
    float softmaxmax = 0.0;
    int softmaxindex = 0;
    float x_c = 0, y_c = 0, w = 0, h = 0;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;

    for (int i = 0; i < MaxObjectNum; i++)
    {
        softmaxsum = 0;
        for (int c = 0; c < ClassNum; c++)
        {
            pred_logits[i * ClassNum + c] = fast_exp(pred_logits[i * ClassNum + c]);
            softmaxsum += pred_logits[i * ClassNum + c];
        }

        for (int c = 0; c < ClassNum; c++)
        {
            pred_logits[i * ClassNum + c] /= softmaxsum;
        }

        softmaxmax = 0.0;
        softmaxindex = 0;
        for (int c = 0; c < ClassNum; c++)
        {
            if (c == 0)
            {
                softmaxmax = pred_logits[i * ClassNum + c];
                softmaxindex = c;
            }

            else
            {
                if (softmaxmax < pred_logits[i * ClassNum + c])
                {
                    softmaxmax = pred_logits[i * ClassNum + c];
                    softmaxindex = c;
                }
            }
        }
        // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在vector<float>中
        if (softmaxmax > ObjectThresh && softmaxindex == 1)
        {
            x_c = pred_boxes[i * 4 + 0];
            y_c = pred_boxes[i * 4 + 1];
            w = pred_boxes[i * 4 + 2];
            h = pred_boxes[i * 4 + 3];

            xmin = x_c - 0.5 * w;
            ymin = y_c - 0.5 * h;
            xmax = x_c + 0.5 * w;
            ymax = y_c + 0.5 * h;

            DetectiontRects.push_back(softmaxindex);
            DetectiontRects.push_back(softmaxmax);
            DetectiontRects.push_back(xmin);
            DetectiontRects.push_back(ymin);
            DetectiontRects.push_back(xmax);
            DetectiontRects.push_back(ymax);
        }
    }

    return 1;
}