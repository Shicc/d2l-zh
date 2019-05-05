# 从头训练or微调模型：迁移学习快速实战

## 为什么要做？

还记得2018年微软创新杯上，我和我同学牛牛的项目中需要训练一个智能识别食物种类的模型。当时没有计算资源的我们，写完代码用CPU跑了跑就提交了，止步复赛。今年一大段无法活动的时间里，整理了以前的代码，我捉摸着观察这一神奇的数据集，看着未训练的模型，心想着这东西必须得整一下，也为之后的某次盘问收集点会议资料，待到半年后大伤初愈后，估计又得上路了。

## 微调

在计算机视觉上，如果从头训练一个模型，我们不经需要足量的数据集来满足训练要求，也需要足够多的计算资源，显然这不是每一个人都能达到的，有一种办法那就是迁移学习（transfer learning），它将从源数据集学到的知识迁移到目标数据集上，预训练模型可以抽取较为通用的图像特征，只需更改模型的输出层并重新训练，使之满足目标数据集的要求。微调（fine tuning）是迁移学习中的一种常用技术。微调的意思就是对于目的模型的features采用预训练模型的features并用很小的学习率微调或者就固定不变，而输出层采用随机初始化并从头开始训练。

## 数据集介绍

在kaggle上，合适的数据集就是[food101](https://www.kaggle.com/prathmeshgodse/food101-zip)了。但神奇的是9GB大的数据集中，训练集和测试集内容都是一样的。严格意义上讲，测试集只能在模型的各种参数确定后用来测试模型的精度，但训练集和测试集一样的话，那相当于还是用的训练集在做测试，这样得出的泛化误差那就是训练误差，是不对的。当时做的时候也没管，用CPU去跑这个庞大的数据集也没能体验到fine-tuning的魅力，对于最后一个101类的分类器也不好训练，其实fing-tuning没啥东西，但有时候懂算法就是那一个全新的任务不好上手，跑不好模型还是时有发生的，这次闲着无聊整了一个5个类的小数据集，且正确划分了训练集和测试集。冥冥之中觉得又一个上手fing-tuning的好例子，小型数据集：[网盘](https://pan.baidu.com/s/1Oaf8m5IZiwLOBqc9gJ4aDw) 提取码: 8u46

训练集各类各700张，测试集各类各300张，其文件结构如下：

* food5
  * train
    * apple_pie
    * baby_back_ribs
    * baklava
    * beef_carpaccio
    * beef_tartare
  * test
    * apple_pie
    * baby_back_ribs
    * baklava
    * beef_carpaccio
    * beef_tartare

## 依赖

```python
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import data as gdata, loss as gloss, model_zoo
import os
```

## 读取数据

gluon提供了一个读取用文件夹名字分好类的图像数据集接口，同一类别的数据都放在同一文件夹中，文件夹名就是类别名。函数会自动生成相应的label。

```python
train_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'train'))
test_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'test'))
```

得到的train_imgs和test_imgs就是一个`ImageFolderDataset`类，通过`__len__()`可以得到其长度，即包含多少个样本， `.__getitem__(x)`会返回第x个样本和其类别。

处理数据时，我们对RGB三个颜色通道的数值做标准化，每个数值减去该通道所有数值的平均值，再除以该通道所有数值的标准差作为输出。当图片大小不一样时，我们先从图像中裁剪出随机大小和随机高宽比的一块随机区域，然后将该区域缩放为高和宽均为224像素的输入。测试时，我们将图像的高和宽均缩放为256像素，然后从中裁剪出高和宽均为224像素的中心区域作为输入。

```python
# 指定RGB三个通道的均值和方差来将图像通道归一化
# 此数据是ImageNet上的数据算出来的数值
normalize = gdata.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomResizedCrop(224),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    normalize])

test_augs = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    normalize])
```

调用`ImageFolderDataset`类的`transform_first(train_augs)`函数，传入变化的规则，它会把数据集的全部输入按照规则来变化而保留label不动，这就得到了我们数据处理后的可以拿来训练的数据集。在通过gluon提供的`DataLoader`类来获取批量数据，`DataLoader`类接受三个参数，第一个是数据集，第二个是`batch_size`，会把数据集按照`batch_size`来划分，第三个指定是否随机读取的，`shuffle=True`表示随机读取

```python
train_iter = gdata.DataLoader(train_imgs.transform_first(train_augs), batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_imgs.transform_first(test_augs), batch_size)
```

## 搭建模型

以前使用的`NumPy`的`vgg`参数文件基于`TensorFlow`1.x的代码做fine tuning很麻烦，代码臃肿还不好理解，详情请看这个[repo](https://github.com/Shicc/food101)。`mxnet`的`Gluon`前端出来后，便利上也广受好评，本次就放弃原先的代码，用`Gluon`3行代码快速做好fine tuning。

### 定义微调模型

```python
# 采用在ImageNet上预训练的resnet34_v2网络
pretrained_net = model_zoo.vision.resnet34_v2(pretrained=True)

finetune_net = model_zoo.vision.resnet34_v2(classes=5) #classes表示类别数
# 固定features中的参数：
# finetune_net.features.collect_params().setattr('grad_req', 'null')
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# output中的模型参数将在迭代中使用10倍大的学习率
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

### 微调模型

```python
def train_fine_tuning(net, learning_rate, batch_size=200, num_epochs=20):
    ctx = d2l.try_all_gpus()
    net.collect_params().reset_ctx(ctx) # 把数据复制到gpu上
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

train_fine_tuning(finetune_net, 0.01) #用0.01的小学习率微调features
```

## 结果

简单跑了一下结果也还能接受，但从最后几个epochs看来，还有优化空间

```teminal
training on [gpu(0)]
epoch 1, loss 3.8275, train acc 0.401, test acc 0.652, time 1299.6 sec
epoch 2, loss 1.1384, train acc 0.680, test acc 0.772, time 50.4 sec
epoch 3, loss 0.5352, train acc 0.817, test acc 0.801, time 51.5 sec
epoch 4, loss 0.4826, train acc 0.828, test acc 0.842, time 50.4 sec
epoch 5, loss 0.4427, train acc 0.842, test acc 0.813, time 49.8 sec
epoch 6, loss 0.3969, train acc 0.858, test acc 0.856, time 50.5 sec
epoch 7, loss 0.3310, train acc 0.879, test acc 0.867, time 49.6 sec
epoch 8, loss 0.3035, train acc 0.893, test acc 0.818, time 50.7 sec
epoch 9, loss 0.2848, train acc 0.895, test acc 0.865, time 51.3 sec
epoch 10, loss 0.3183, train acc 0.883, test acc 0.865, time 49.4 sec
epoch 11, loss 0.2798, train acc 0.895, test acc 0.875, time 50.3 sec
epoch 12, loss 0.2640, train acc 0.902, test acc 0.861, time 51.0 sec
epoch 13, loss 0.2406, train acc 0.915, test acc 0.877, time 49.7 sec
epoch 14, loss 0.2262, train acc 0.918, test acc 0.875, time 50.3 sec
epoch 15, loss 0.1985, train acc 0.929, test acc 0.869, time 50.4 sec
epoch 16, loss 0.2123, train acc 0.923, test acc 0.883, time 50.4 sec
epoch 17, loss 0.1978, train acc 0.925, test acc 0.886, time 50.3 sec
epoch 18, loss 0.1935, train acc 0.929, test acc 0.884, time 49.9 sec
epoch 19, loss 0.1899, train acc 0.934, test acc 0.878, time 50.4 sec
epoch 20, loss 0.1936, train acc 0.931, test acc 0.894, time 50.5 sec
```

### gpu使用情况

只是没想到这么少的数据都占了这么多显存

```teminal
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.56       Driver Version: 410.79       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   77C    P0    31W /  70W |  15071MiB / 15079MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

而finetuning_net的输出层则被改成了5个类

```python
>>pretrained_net.output
Dense(2048 -> 5, linear)
```