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
    int ClassNum = 2 + 1; // ģ������ǣ�background, person, empty��ʵ����һ����, ѵ��ʱ�������д����3��ʵ����2
    float ObjectThresh = 0.5;
};

#endif