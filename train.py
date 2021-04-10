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
import models
ImageFile.LOAD_TRUNCATED_IMAGES = True

path = 'D:/deeplearning/datasets/imageclassification/Flower-102/'
# path = 'D:/deeplearning/datasets/imageclassification/Fruits360-131/'
# path = 'D:/deeplearning/datasets/imageclassification/Leafsnap-184/'
classes = int(path.split('-')[-1].split('/')[0])
modelx = 'EfficientNetB0'
timenum = 1
hwd = 224
batch_sizes = 32
# epoch = 100
epoch = 50

# learn rate
def lr_schedule(epoch):
    # lr = 3e-5
    # lr = 3e-6
    lr = 3e-8
    if epoch > 300:
        lr *= 0.01
    elif epoch > 200:
        lr *= 0.5
    elif epoch > 100:
        lr *= 0.1 * 0.2
    elif epoch > 50:
        lr *= 0.1
    print('Learning rate: {:.1e}'.format(lr))
    return lr

# train model
def trainmodel():
    model = models.myEfficientNetB0(input_shape=(hwd, hwd, 3), classes=classes)
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
    train_generator = train_datagen.flow_from_directory("{}/train/".format(path), batch_size=batch_sizes,
                                                        class_mode='categorical', target_size=(hwd, hwd))
    # valid_generator = test_datagen.flow_from_directory("{}/valid/".format(path), batch_size=batch_sizes,
    #                                                    class_mode='categorical', target_size=(w, h))
    test_generator = test_datagen.flow_from_directory("{}/test/".format(path), batch_size=batch_sizes,
                                                      class_mode='categorical', target_size=(hwd, hwd))
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
    model.load_weights("logs/cp/cp-0050.h5")
    # history = model.fit(train_generator, epochs=epoch, validation_data=valid_generator,
    history = model.fit(train_generator, epochs=epoch, callbacks=[tensorboard_callback, cp_callback])
    modelnum = history_csv(model, test_generator, history.history, pathcsv='{}/plt.csv'.format(dirs))
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
            print('{}/{}'.format(i + 1, epoch))
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
    f.close()
    return modelnum

if __name__ == '__main__':
    for i in range(timenum):
        if i != 0:
            time.sleep(60 * 2)
        p = Process(target=trainmodel)
        p.start()
        p.join()
    pass

# 2021-04-09 guangjinzheng tensorflow efficientnet
