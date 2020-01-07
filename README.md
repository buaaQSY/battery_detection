# Faster R-CNN for Machine Learning Course Project

### 训练好的模型
[下载链接](https://www.dropbox.com/s/ut45d7pfv6po4rx/faster_rcnn_1_5_1999.pth?dl=0)

### 数据集
需要将VOC2007格式的数据集(Annatations文件夹、ImageSets文件夹以及JPEGImages文件夹)放置在data/VOCdevkit2007/VOC2007下

### 编译CUDA
训练前，先进入lib文件夹下运行：

`python setup.py build develop`

之后运行：
```
cd ..
pip install -r requirement.txt
```
### 加载预训练模型
在自己的数据集上开始训练自己的模型时，需要加载res101预训练模型，[下载链接](https://www.dropbox.com/s/rr0pkuzinjeurwx/resnet101_caffe.pth?dl=0)

将下载好的文件放在data/pretrained_model文件夹下

### 训练模型
Run command:
```
CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset pascal_voc --net res101 \
                   --epochs 8 --bs 1 --nw 4 \
                   --lr  1e-3 --lr_decay_step 5 \
                   --cuda
```

### 测试集上测试模型
Run command:
```
python test.py --img_path img_path --anno_path anno_path \
               --test_file test_file_path --net res101 \
               --checksession 1 --checkepoch 5 --checkpoint 1999 \
               --cuda
```

### Results
AP for 带电芯充电宝 = 0.8697

AP for 不带电芯充电宝 = 0.8368

Mean AP = 0.8533

~ ~ ~ ~ ~ ~ ~ ~

Results:

0.870

0.837

0.853

~ ~ ~ ~ ~ ~ ~ ~



