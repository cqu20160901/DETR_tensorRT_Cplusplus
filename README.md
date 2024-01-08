# detr tensorRT 的 C++ 部署

本示例中，包含完整的代码、模型、测试图片、测试结果。

TensorRT版本：TensorRT-8.2.1.8，（最早使用TensorRT-7.1.3.4 运行报错，使用TensorRT-8.2.1.8运行结果正常）。

由于模型较大无法直接上传，onnx和tensorrt 模型文件，[模型存储链接](https://github.com/cqu20160901/DETR_tensorRT_Cplusplus/releases)。


## 建议先看

[解决tesorrt 推理输出结果全为0的问题参考](https://blog.csdn.net/zhangqian_1/article/details/135453388)

## 编译

修改 CMakeLists.txt 对应的TensorRT位置

![image](https://github.com/cqu20160901/DETR_tensorRT_Cplusplus/assets/22290931/227ae810-8a01-49fc-82fa-5ae3a659b68f)


```powershell
cd DETR_tensorRT_Cplusplus
mkdir build
cd build
cmake ..
make
```

## 运行

```powershell
# 运行时如果.trt模型存在则直接加载，若不存会自动先将onnx转换成 trt 模型，并存在给定的位置，然后运行推理。
cd build
./detr_trt
```

## 测试效果

onnx 测试效果

![test_onnx_result](https://github.com/cqu20160901/DETR_tensorRT_Cplusplus/assets/22290931/f309fc50-df2d-4d34-b13d-f05a6cb3dddf)


tensorRT 测试效果

![result](https://github.com/cqu20160901/DETR_tensorRT_Cplusplus/assets/22290931/05ddc58c-8e9e-4890-9c92-067ce6b8451d)


tensorRT 时耗

使用的显卡 Tesla V100、cuda_11.0

![image](https://github.com/cqu20160901/DETR_tensorRT_Cplusplus/assets/22290931/ebc87337-e9ae-4d37-b8d0-0dc9db14f7af)


## 替换模型说明

1）导出的onnx模型建议simplify后，修改Gather层后再转trt模型。

2）注意修改后处理相关 postprocess.hpp 中相关的类别参数。

修改相关的路径

```cpp
    std::string OnnxFile = "/zhangqian/workspaces1/TensorRT/detr_trt_Cplusplus/models/detr_r50_person_sim_change.onnx";
    std::string SaveTrtFilePath = "/zhangqian/workspaces1/TensorRT/detr_trt_Cplusplus/models/detr_r50_person_sim_change.trt";
    cv::Mat SrcImage = cv::imread("/zhangqian/workspaces1/TensorRT/detr_trt_Cplusplus/images/test.jpg");

    int img_width = SrcImage.cols;
    int img_height = SrcImage.rows;

    CNN DETR(OnnxFile, SaveTrtFilePath, 1, 3, 640, 640, 3);  // 1, 3, 640, 640, 3 前四个为模型输入的NCWH, 3为模型输出叶子节点的个数+1，（本示例中的onnx模型输出有2个叶子节点，再+1=7）
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


```

## 特别说明

本示例只是用来测试流程，模型效果并不保证，且代码整理的布局合理性没有做过多的考虑。本示例提供的模型只检测行人，由于训练的时类别写成了3，因此模型输出结果只有第二类是有效的。

## 相关链接

[python tensorrt 部署](https://github.com/cqu20160901/DETR_onnx_tensorRT)

[解决tesorrt 推理输出结果全为0的问题参考](https://blog.csdn.net/zhangqian_1/article/details/135453388)
