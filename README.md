# 装甲板识别-----onnxruntime推理

本代码为上海交通大学2021年开源模型的onnxruntime推理

参考开源

[Harry-hhj/CVRM2021-sjtu: 上海交通大学 RoboMaster 2021赛季 视觉代码 (github.com)](https://github.com/Harry-hhj/CVRM2021-sjtu)

[Spphire/YOLOarmor-2022final (github.com)](https://github.com/Spphire/YOLOarmor-2022final)



video视频文件请访问网盘链接下载

链接：https://pan.baidu.com/s/1QykXf3QvKQdGDIeRvCxdzw?pwd=0000 
提取码：0000 

模型文件

链接：https://pan.baidu.com/s/1qtwrBaqnu3CHkI8d8YpqGA?pwd=0000 
提取码：0000 



安装所需依赖

```
pip install onnxruntime
pip install numpy
pip install opencv-python
pip install torch
#这样安装pytorch为cpu版本，gpu版本可从pytorch进行安装
```

运行代码

```
python onnx_infer.py
#代码默认cpu推理，更改gpu推理可参考网上相关资料进行修改，因本人电脑没有显卡，没进行相关测试
```

