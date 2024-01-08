#include "src/CNN.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    std::string OnnxFile = "/zhangqian/workspaces1/TensorRT/detr_trt_Cplusplus/models/detr_r50_person_sim_change.onnx";
    std::string SaveTrtFilePath = "/zhangqian/workspaces1/TensorRT/detr_trt_Cplusplus/models/detr_r50_person_sim_change.trt";
    cv::Mat SrcImage = cv::imread("/zhangqian/workspaces1/TensorRT/detr_trt_Cplusplus/images/test.jpg");

    int img_width = SrcImage.cols;
    int img_height = SrcImage.rows;

    CNN DETR(OnnxFile, SaveTrtFilePath, 1, 3, 640, 640, 3); // 1, 3, 640, 640, 3 前四个为模型输入的NCWH, 3为模型输出叶子节点的个数+1，（本示例中的onnx模型输出有2个叶子节点，再+1=3）
    DETR.ModelInit();
    DETR.Inference(SrcImage);

    for (int i = 0; i < DETR.DetectiontRects_.size(); i += 6)
    {
        int classId = int(DETR.DetectiontRects_[i + 0]);
        float conf = DETR.DetectiontRects_[i + 1];
        int xmin = int(DETR.DetectiontRects_[i + 2] * float(img_width) + 0.5);
        int ymin = int(DETR.DetectiontRects_[i + 3] * float(img_height) + 0.5);
        int xmax = int(DETR.DetectiontRects_[i + 4] * float(img_width) + 0.5);
        int ymax = int(DETR.DetectiontRects_[i + 5] * float(img_height) + 0.5);

        char text1[256];
        sprintf(text1, "%d:%.2f", classId, conf);
        rectangle(SrcImage, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
        putText(SrcImage, text1, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    imwrite("/zhangqian/workspaces1/TensorRT/detr_trt_Cplusplus/images/result.jpg", SrcImage);

    printf("== obj: %d \n", int(float(DETR.DetectiontRects_.size()) / 6.0));

    return 0;
}
