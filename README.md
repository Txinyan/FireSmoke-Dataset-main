# 基于改进YOLOv5的火灾烟雾检测

## 📊 性能对比
| 模型 | 精确率(P) | 召回率(R) | F1分数 | AP50 | 帧率(FPS) |
|------|-----------|-----------|--------|------|-----------|
| YOLOv5s (基线) | 63.3% | 81.6% | 71.0% | 79.6% | 154 |
| YOLOv5_DS | **70.1%** | **82.6%** | **74.9%** | **81.2%** | 152 |


# 训练YOLOv5_DS模型
python train.py --img 640 --batch 16 --epochs 100 \
                --data data/firesmoke.yaml \
                --weights yolov5s.pt \
                --cfg models/yolov5_ds.yaml


📁 数据集 | Dataset
中文版本
FireSmoke数据集包含12,032张高质量RGB图像：

训练集：10,795张图像

测试集：1,237张图像

火灾实例：约6,327个

烟雾实例：约4,468个

数据来源：现场巡检机器人采集 + 公开数据库

Google Drive：https://drive.google.com/file/d/1ddO1oVfL1i08FkAbp9mSLwuQDC6q41XW/view?usp=drive_link


