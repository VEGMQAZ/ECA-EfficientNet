from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from multiprocessing import Process
from PIL import ImageFile
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import time
import csv
import os
import argparse
import models
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='Flower', help="is Flower, Fruit or Leaf")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=2e-5, help="Adam: learning rate")
parser.add_argument("--af", type=str, default='swish', help="is relu, swish or hswish")
parser.add_argument("--at", type=str, default='eca', help="is se or eca")
parser.add_argument("--load", type=int, default=0, help="number of models")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--num", type=int, default=1, help="number of running train.py")
opt = parser.parse_args()
print(opt)
path = 'D:/deeplearning/datasets/imageclassification/'
if opt.data in 'Fruits360-131':
    path += 'Fruits360-131/'
elif opt.data in 'Leafsnap-184':
    path += 'Leafsnap-184/'
else:
    path += 'Flower-102/'

# learn rate
def lr_schedule(epoch):
    lr = opt.lr
    arr = [50, 100, 150, 200]
    if epoch > arr[3]:
        lr *= 0.0001
    elif epoch > arr[2]:
        lr *= 0.001
    elif epoch > arr[1]:
        lr *= 0.01
    elif epoch > arr[0]:
        lr *= 0.1
    print('Epoch: {}  Learning rate: {:.1e}'.format(epoch, lr))
    return lr

# train model
def trainmodel():
    classes = int(path.split('-')[-1].split('/')[0])
    modelx = 'EfficientNetB0'
    model = models.myEfficientNet(attention=opt.at, activation=opt.af, input_shape=(opt.img_size, opt.img_size, 3), classes=classes)
    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tfa.metrics.F1Score(name='f1score', num_classes=classes)
    ]
    model.compile(optimizer=Adam(lr_schedule(0)), loss='categorical_crossentropy', metrics=METRICS)
    # train data
    train_datagen = ImageDataGenerator(rescale=1./255., rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255.)
    train_generator = train_datagen.flow_from_directory("{}/train/".format(path), batch_size=opt.batch_size,
                                                        class_mode='categorical', target_size=(opt.img_size, opt.img_size))
    # valid_generator = test_datagen.flow_from_directory("{}/valid/".format(path), batch_size=batch_sizes,
    #                                                    class_mode='categorical', target_size=(w, h))
    test_generator = test_datagen.flow_from_directory("{}/test/".format(path), batch_size=opt.batch_size,
                                                      class_mode='categorical', target_size=(opt.img_size, opt.img_size))
    # callback
    if not os.path.exists('logs/cp'):
        os.makedirs('logs/cp')
    timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dirs = 'logs/{}/{}/'.format(modelx, timenow)
    tensorboard_callback = TensorBoard(log_dir="{}".format(dirs), histogram_freq=1)
    cp_callback = ModelCheckpoint(filepath="logs/cp/cp-{epoch:04d}.h5", period=1, save_weights_only=True, verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.2, patience=5, min_lr=1e-8)
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # load weights
    if opt.load > 0:
        model.load_weights('logs/cp/cp-{:04d}.h5'.format(opt.load))
    # history = model.fit(train_generator, epochs=epoch, validation_data=valid_generator,
    history = model.fit(train_generator, epochs=opt.epochs, callbacks=[tensorboard_callback, cp_callback])
    modelnum = history_csv(model, test_generator, history.history, pathcsv='{}/{}-{}-plt.csv'.format(dirs, modelx, timenow))
    model.load_weights('logs/cp/cp-{:04d}.h5'.format(modelnum))
    score = model.evaluate(test_generator)
    model.save('{}/{}-{:.6f}-{:.4f}.h5'.format(dirs, timenow, score[0], score[1]*100))
    print('{}'.format(score))

# save loss acc
def history_csv(model, test, history, pathcsv='plt.csv'):
    str_lossacc = ['loss', 'accuracy', 'precision', 'recall', 'id',
                   'test_loss', 'test_accuracy', 'test_precision', 'test_recall']
    epochs = len(history[str_lossacc[0]])
    modelmax, modelnum = 0, 0
    with open(pathcsv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=str_lossacc)
        writer.writeheader()
        for i in range(epochs):
            print('{}/{}'.format(i + 1, opt.epochs))
            model.load_weights("logs/cp/cp-{:04d}.h5".format(i + 1))
            score = model.evaluate(test)
            writer.writerow({str_lossacc[0]: history[str_lossacc[0]][i], str_lossacc[1]: history[str_lossacc[1]][i],
                             str_lossacc[2]: history[str_lossacc[2]][i], str_lossacc[3]: history[str_lossacc[3]][i],
                             str_lossacc[4]: '{}'.format(i + 1),
                             str_lossacc[5]: '{}'.format(score[0]), str_lossacc[6]: '{}'.format(score[1]),
                             str_lossacc[7]: '{}'.format(score[2]), str_lossacc[8]: '{}'.format(score[3])})
            if score[1] > modelmax:
                modelmax = score[1]
                modelnum = i + 1
        writer.writerow({str_lossacc[0]: opt})
    f.close()
    return modelnum

def times(x=0):
    arr_data = ['Flower', 'Leaf', 'Fruit']
    arr_at = ['se', 'eca']
    arr_af = ['hswish', 'swish', 'relu']
    num = 0
    for i in range(len(arr_data)):
        for j in range(len(arr_at)):
            for k in range(len(arr_af)):
                num = num + 1
                if num == x:
                    opt.data = arr_data[i]
                    opt.at = arr_at[j]
                    opt.af = arr_af[k]
                    print('{} {} {}'.format(opt.data, opt.at, opt.af))
                    break

if __name__ == '__main__':
    for i in range(opt.num):
        times(i+1)
        trainmodel()
    # for i in range(opt.num):
    #     times(i+1)
    #     if i != 0:
    #         time.sleep(60 * 2)
    #     p = Process(target=trainmodel)
    #     p.start()
    #     p.join()
    pass

# 2021-04-10 guangjinzheng tensorflow efficientnet
