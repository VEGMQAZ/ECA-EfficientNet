# EfficientNet-ECA-Hswish  (TensorFlow and Keras)

[![PyPI version](https://badge.fury.io/py/efficientnet.svg)](https://badge.fury.io/py/efficientnet) [![Downloads](https://pepy.tech/badge/efficientnet/month)](https://pepy.tech/project/efficientnet/month)

This repository contains a Keras (and TensorFlow Keras) reimplementation of **EfficientNet**, a lightweight convolutional neural network architecture achieving the [state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPS](https://arxiv.org/abs/1905.11946), on both ImageNet and
five other commonly used transfer learning datasets.

The codebase is heavily inspired by the [TensorFlow implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## result

### plant datasets
|Dataset |Classes|Size   |Train Size|Test Size|
| :----  |:----: |:----: |:----:    |:----:   |
|Flower  |	102  |8,189  |6,149     |    2,040|
|Fruit360|	131  |90,380 |67,692    |	22,688|
|Leafsnap|	184  |7,719  |6,238     |	 1,481|
|Total   |	417  |106,288|80,079    |	26,209|

### Hyper-param setup
|Param  | value|
|:----  |:----|
|input  |224 x 224 x 3|
|epoch  |200|
|initlr |2e-5|
|minlr  |2e-9|
|delay  |0.1|
|arr    |[50, 100, 150, 200]|

### Flower dataset
|Method    |Top1 Acc|Params |FLOPs   |Training|Latency |
| :----    |:----:  |:----: |:----:  |:----:  |:----:  |
|ReLU+SE   |96.814% |4.180M |0.394G  |eca     |96.1%   |
|Swish+SE  |97.549% |4.180M |0.397G  |eca     |96.1%   |
|Hswish+SE |97.206% |4.180M |0.403G  |eca     |96.1%   |
|ReLU+ECA  |96.471% |3.545M |0.394G  |eca     |96.1%   |
|Swish+ECA |97.255% |3.545M |0.397G  |eca     |96.1%   |
|Hswish+ECA|97.157% |3.545M |0.403G  |eca     |96.1%   |

|Method          |Top1 Acc|Params |FLOPs   |Training|Latency |
| :----          |:----:  |:----: |:----:  |:----:  |:----:  |
|VGG16           |96.1%   |eca    |96.1%   |eca     |96.1%   |
|ResNet101       |96.1%   |eca    |96.1%   |eca     |96.1%   |
|InceptionV3     |96.1%   |eca    |96.1%   |eca     |96.1%   |
|DenseNet169     |96.1%   |eca    |96.1%   |eca     |96.1%   |
|NASNetMobile    |96.1%   |eca    |96.1%   |eca     |96.1%   |
|MobileNetV2     |96.1%   |eca    |96.1%   |eca     |96.1%   |
|ECA-EfficientNet|96.1%   |eca    |96.1%   |eca     |96.1%   |

### Leaf dataset
|Method    |Top1 Acc|Params |FLOPs   |Training|Latency |
| :----    |:----:  |:----: |:----:  |:----:  |:----:  |
|ReLU+SE   |96.1%   |4.285M |0.394G   |eca     |96.1%   |
|Swish+SE  |96.1%   |4.285M |0.397G   |eca     |96.1%   |
|Hswish+SE |96.1%   |4.285M |0.404G   |eca     |96.1%   |
|ReLU+ECA  |96.1%   |3.650M |0.394G  |eca     |96.1%   |
|Swish+ECA |96.1%   |3.650M |0.397G   |eca     |96.1%   |
|Hswish+ECA|96.1%   |3.650M |0.403G   |eca     |96.1%   |

|Method          |Top1 Acc|Params |FLOPs   |Training|Latency |
| :----          |:----:  |:----: |:----:  |:----:  |:----:  |
|VGG16           |96.1%   |eca    |96.1%   |eca     |96.1%   |
|ResNet101       |96.1%   |eca    |96.1%   |eca     |96.1%   |
|InceptionV3     |96.1%   |eca    |96.1%   |eca     |96.1%   |
|DenseNet169     |96.1%   |eca    |96.1%   |eca     |96.1%   |
|NASNetMobile    |96.1%   |eca    |96.1%   |eca     |96.1%   |
|MobileNetV2     |96.1%   |eca    |96.1%   |eca     |96.1%   |
|ECA-EfficientNet|96.1%   |eca    |96.1%   |eca     |96.1%   |

### Fruit360 dataset
|Method    |Top1 Acc|Params |FLOPs  |Training|Latency |
| :----    |:----:  |:----: |:----: |:----:  |:----:  |
|ReLU+SE   |96.1%   |4.217M |0.394G |eca     |96.1%   |
|Swish+SE  |96.1%   |4.217M |0.397G |eca     |96.1%   |
|Hswish+SE |96.1%   |4.217M |0.403G |eca     |96.1%   |
|ReLU+ECA  |96.1%   |3.582M |0.394G |eca     |96.1%   |
|Swish+ECA |96.1%   |3.582M |0.397G |eca     |96.1%   |
|Hswish+ECA|96.1%   |3.582M |0.403G |eca     |96.1%   |

|Method          |Top1 Acc|Params |FLOPs   |Training|Latency |
| :----          |:----:  |:----: |:----:  |:----:  |:----:  |
|VGG16           |96.1%   |eca    |96.1%   |eca     |96.1%   |
|ResNet101       |96.1%   |eca    |96.1%   |eca     |96.1%   |
|InceptionV3     |96.1%   |eca    |96.1%   |eca     |96.1%   |
|DenseNet169     |96.1%   |eca    |96.1%   |eca     |96.1%   |
|NASNetMobile    |96.1%   |eca    |96.1%   |eca     |96.1%   |
|MobileNetV2     |96.1%   |eca    |96.1%   |eca     |96.1%   |
|ECA-EfficientNet|96.1%   |eca    |96.1%   |eca     |96.1%   |


## History

### 2021-04-21 guangjinzheng

Precision Recall 

### 2021-04-20 guangjinzheng

split dataset

delete fruit360



### 2021-04-17 guangjinzheng

ReduceLROnPlateau

### 2021-04-16 guangjinzheng

add data excel

### 2021-04-13 guangjinzheng

add test vgg  https://github.com/keras-team/keras-applications 

add mobilenetv3 small modify

### 2021-04-12 guangjinzheng

add test models 


### 2021-04-12 guangjinzheng

add data


### 2021-04-10 guangjinzheng 

add eca attention 

train.py add save csv with opt argparse data

train.py add argparse 

``` python
python train.py 
python train.py --epoch 100 --lr 2e-5 --load 100 --data flower --af relu --at se --batch_size 32 --img_size 224
```

add af.py  arr relu swish hswish plot  add hswish like relu

modify models.py add myaf like relu swish and hswish

modify models.py add myEfficientNet() function

using tensorflow-gpu EfficientNet C:\Users\guangjinzheng\anaconda3\envs\tf2eca\Lib\site-packages\tensorflow\python\keras\applications

## Requirements

```bash
pip install pillow

pip install matplotlib

pip install scikit-learn

pip install tensorflow-addons

pip install tensorflow-gpu

conda install cudnn
```
