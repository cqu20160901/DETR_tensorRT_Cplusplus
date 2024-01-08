#ifndef _POSTPROCESS_H_
#define _POSTPROCESS_H_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>

// DETR
class DETR
{
public:
    DETR();
    ~DETR();

    int GetDETRDetectResult(float **pBlob, std::vector<float> &DetectiontRects);

private:
    int MaxObjectNum = 100;
    int ClassNum = 2 + 1; // 模型输出是（background, person, empty）实际是一类检测, 训练时由于类别写成了3，实际是2
    float ObjectThresh = 0.5;
};

#endif