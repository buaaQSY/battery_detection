# Faster R-CNN for Machine Learning Course Project

### 训练好的模型
[下载链接](https://1drv.ms/u/s!AsTR1H0w0j_Mg2Zi9We4SDO8kPuD?e=LI4VVd)

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
在自己的数据集上开始训练自己的模型时，需要加载vgg16预训练模型，[下载链接](https://1drv.ms/u/s!AsTR1H0w0j_Mg2WnR9f4SpQLXt58?e=dsjSmg)

将下载好的文件放在data/pretrained_model文件夹下

### 训练模型
Run command:
```
CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --epochs 1 --bs 1 --nw 4 \
                   --lr  1e-3 --lr_decay_step 5 \
                   --cuda
```

### 测试集上测试模型
Run command:
```
python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession 1 --checkepoch 1 --checkpoint 5999 \
                   --cuda
```

### Results
AP for 带电芯充电宝 = 0.7647

AP for 不带电芯充电宝 = 0.7653

Mean AP = 0.7650

~ ~ ~ ~ ~ ~ ~ ~

Results:

0.765

0.765

0.765

~ ~ ~ ~ ~ ~ ~ ~



