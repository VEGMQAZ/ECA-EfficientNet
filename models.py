from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.nasnet import NASNetMobile
import tensorflow.python.keras.applications.efficientnet as ef
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Flatten

# AlexNet
def myAlexNet(input_shape=(224, 224, 3), classes=1000):
    inputs = Input(shape=input_shape)
    c1 = Conv2D(48, (11, 11), strides=4, activation='relu', kernel_initializer='uniform', padding='valid')(inputs)
    c2 = BatchNormalization()(c1)
    c3 = MaxPool2D((3, 3), strides=2, padding='valid')(c2)
    c4 = Conv2D(128, (5, 5), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c3)
    c5 = BatchNormalization()(c4)
    c6 = MaxPool2D((3, 3), strides=2, padding='valid')(c5)
    c7 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c6)
    c8 = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c7)
    c9 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', kernel_initializer='uniform')(c8)
    c10 = MaxPool2D((3, 3), strides=2, padding='valid')(c9)
    c11 = Flatten()(c10)
    c12 = Dense(2048, activation='relu')(c11)
    c13 = Dropout(0.3)(c12)
    c14 = Dense(2048, activation='relu')(c13)
    c15 = Dropout(0.3)(c14)
    outputs = Dense(classes, activation='softmax')(c15)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model

# VGG16
def myVGG16(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = VGG16(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# ResNetV2101
def myResNetV2101(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = ResNet101V2(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = ResNet101V2(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# InceptionV3
def myInceptionV3(input_shape=(299, 299, 3), classes=1000):
    pre_trained_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = InceptionV3(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# Xception
def myXception(input_shape=(299, 299, 3), classes=1000):
    pre_trained_model = Xception(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = Xception(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# MobileNetV2
def myMobileNetV2(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = MobileNetV2(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# DenseNet201
def myDenseNet201(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = DenseNet201(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = DenseNet201(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# DenseNet169
def myDenseNet169(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = DenseNet169(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = DenseNet169(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# NASNetMobile
def myNASNetMobile(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = NASNetMobile(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = NASNetMobile(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    model.summary()
    return model

# EfficientNetB0
def myEfficientNetB0(input_shape=(224, 224, 3), classes=1000):
    pre_trained_model = ef.EfficientNetB0(input_shape=input_shape, weights='imagenet', include_top=False)
    # pre_trained_model = ef.EfficientNetB0(weights='imagenet', include_top=True)
    # pre_trained_model.summary()
    x = pre_trained_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.2, name='top_dropout')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=pre_trained_model.input, outputs=predictions)
    # for layer in model.layers[:-6]:
    #     layer.trainable = False
    #     print(layer.name)
    model.summary()
    return model

if __name__ == '__main__':
    myAlexNet()
    myVGG16()
    myResNetV2101()
    myInceptionV3()
    myXception()
    myMobileNetV2()
    myDenseNet169()
    myDenseNet201()
    myNASNetMobile()
    myEfficientNetB0()
    pass

# 2020-11-09 guangjinzheng models
