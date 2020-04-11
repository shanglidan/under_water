# 水下目标检测算法赛（声学）  underwater object detection algorithm contest 

[TOC]

## 整体思路
   + detection algorithm: Cascade R-CNN 
   + backbone: ResNet101 + FPN以及Se_ResNet101+ FPN
   + post process: soft nms
   + 基于mmdetection

## 代码环境及依赖

+ OS: Ubuntu16.04
+ GPU: 2080Ti * 4
+ python: python3.7
+ nvidia 依赖:
   - cuda: 10.0
   - nvidia driver version: 410.57
+ deeplearning 框架: pytorch1.1.0
+ 其他依赖请参考requirement.txt

## 数据准备

- **相应文件夹创建准备**

  - 在data目录下新建

    - submit
    - results
    - pretrain
    - acoustic
    
    并在acoustic目录下新建如下目录：
 
      |-- acoustic<br>
      |------a-test-image<br>
      |-------------- image<br>
      |------------------ front<br>
      |---------------------- xx.jpg(前视声纳测试图像)<br>
      |------------------ side<br>
      |---------------------- xx.jpg(侧扫声纳测试图像）<br>
      |------ train<br>
      |-------------- front<br>
      |-------------------annotations<br>
      |-------------------box<br>
      |--------------------- xx.xml<br>
      |-------------------image<br>
      |--------------------- xx.jpg<br>
      |-------------- negtive<br>
      |-------------------box<br>
      |--------------------- xx.xml<br>
      |-------------------image<br>
      |--------------------- xx.jpg<br>
      |--------------side<br>
      |-------------------annotations<br>
      |-------------------box<br>
      |--------------------- xx.xml<br>
      |-------------------image<br>
      |--------------------- xx.jpg<br>
    
- **label文件格式转换**

  - 官方提供的是VOC格式的xml类型label文件，转化为COCO格式用于训练
  - 为了方便利用mmd多进程测试，对test数据也生成一个伪标签文件
  - 运行内容：
    sh data_process.sh

- **预训练模型下载**
  
  - 下载casacde-rcnn-dconv-r101-fpn-1xCOCO预训练模型[cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth](https://pan.baidu.com/s/12NbxkQpeoIDQtlrZ8szwIg
  ) ，提取码：c1u3，并将权重放置于 data/pretrain 目录下
## 依赖安装及编译
   1. 安装 pytorch
        pip install torch==1.1.0 torchvision==0.3.0
        
   2. 安装其他依赖
        pip install cython && pip --no-cache-dir install -r requirements.txt

   3. 编译cuda op等：
        python setup.py develop


## 模型训练及预测

   - 模型训练及预测步骤均写在shell文档下

     - 训练过程文件及最终权重文件均保存在config文件指定的workdirs目录中
     - 预测结果json文件会保存在 data/results 目录下
     - 提交的csv文件在submit目录下

- **运行**
   
  ### 训练
    sh train.sh
  ### 推理
    sh inference.sh
  ### 融合
    sh merge.sh
    
训练好的权重链接https://pan.baidu.com/s/1FfMyRotjvcNWb-aKqTqwXg  提取码：abiu 

## Contact

    author：
    
    Tel：15734009096
