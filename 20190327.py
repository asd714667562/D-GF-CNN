import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.layers import regularizers
from keras import backend as K
from keras import regularizers
import h5py
import matplotlib.pyplot as plt
import keras
from keras.callbacks import TensorBoard
import math
from keras.utils import plot_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import seaborn as sb         # 一个构建在matplotlib上的绘画模块，支持numpy,pandas等数据结构


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix     # 混淆矩阵

import itertools

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix（SNR=NONE)',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # interpolation代表的是插值运算，'nearest'只是选取了其中的一种插值方式。
    # cmap表示绘图时的样式，这里选择的是ocean主题。
    plt.title(title)
    plt.colorbar()
    # 给子图添加colorbar（颜色条或渐变色条）
    tick_marks = np.arange(len(classes))
    # names=np.arange(4), ['2ASK', '2PSK', '2FSK', '4PSK']
    plt.xticks(np.arange(11), ('2ASK', '2PSK', '2FSK', '4PSK', 'AM', 'SSB', 'DSB', 'VSB', 'FM', '16QAM', '64QAM'), rotation=45)
    plt.yticks(np.arange(11), ('2ASK', '2PSK', '2FSK', '4PSK', 'AM', 'SSB', 'DSB', 'VSB', 'FM', '16QAM', '64QAM'), rotation=0)
    # plt.xticks(names, classes, rotation=0)
    # plt.yticks(names, classes)
    # plt.xticks(tick_marks, classes, rotation=0)
    # # plt.xticks(np.arange(4), ['one', 'two', 'three', 'four', 'five'])
    # plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# def l1_reg(x):
#     return (exp(x)-1)*(x)/(exp(x)+1)

# 全局变量
batch_size = 440
nb_classes = 11
epochs = 5
# input image dimensions
img_rows, img_cols = 90, 90
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets


train_dataset = h5py.File(r'F:\Gesture recognition using CNN and radar signal processing\main\dataNONE.h5','r')
train_set_x_orig = np.array(train_dataset['X_train'][:]) # your train set features
train_set_y_orig = np.array(train_dataset['Y_train'][:]) # your train set labels
test_set_x_orig = np.array(train_dataset['X_test'][:]) # your train set features
test_set_y_orig = np.array(train_dataset['Y_test'][:]) # your train set labels

print("数据集已读入成功")

input_shape = (img_rows, img_cols, 3)
X_train = train_set_x_orig.astype('float32')
X_test  = test_set_x_orig.astype('float32')
X_train /= 255
X_test  /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print("数据集归一化成功")

# 转换为one_hot类型
Y_train = np_utils.to_categorical(train_set_y_orig, nb_classes)
Y_test = np_utils.to_categorical(test_set_y_orig, nb_classes)
print("标签转换成one-hot成功")
# -----------------------------------------------------------------------------------------------------------------------------------------

# AlexNet
model = Sequential()
# 第一段
model.add(Conv2D(filters=96, kernel_size=kernel_size, padding='valid', dilation_rate=2, input_shape=input_shape, activation='relu'))#86*86*96
# model.add(Conv2D(filters=96, kernel_size=(5,5), padding='valid', input_shape=input_shape, activation='relu'))#86*86*96
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))#43*43*96
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid',  dilation_rate=2, activation='relu'))#39*39*128
# model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='valid', activation='relu'))#39*39*128
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))#19*19*128

# 第二段
# model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))#19*19*256
# model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='valid',  dilation_rate=2, activation='relu'))#15*15*256
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))#9*9*128
# 第三段
# model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
# model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
# model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))#4*4*128
# 第四段
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2048, activation='relu'))
# model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
model.add(Dropout(0.5))

# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.5))

# Output Layer
model.add(Dense(11,kernel_regularizer=regularizers.le()))
# model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print("模型构建成功")
model.summary()
#
# # AlexNet
# model = Sequential()
# # 第一段
# model.add(Conv2D(filters=96, kernel_size=kernel_size, padding='valid',dilation_rate=2, input_shape=input_shape, activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='valid',  activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# # model.add(Conv2D(filters=96, kernel_size=kernel_size, strides=(4, 4), padding='valid', input_shape=input_shape, activation='relu'))
# # model.add(BatchNormalization())
# # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
# # 第二段
# model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# # 第三段
# model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
# model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
# model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# # 第四段
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(4096, activation='relu'))
# # model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
# model.add(Dropout(0.5))
#
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.5))
#
# # Output Layer
# model.add(Dense(11,kernel_regularizer=regularizers.le()))
# # model.add(Dense(4))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# print("模型构建成功")
# model.summary()
# -------------------------------------------------------------------------------------------------------------------------------------
#训练模型
# plot_model(model,to_file='C:\model.png')

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test)
                  # callbacks=[TensorBoard(log_dir='./log_dir')]
                  )
print("模型训练结束")

# tensorboard --logdir=F:log_dir
# model.save_weights('my_model_weights.h5')
#评估模型
scoretrain= model.evaluate(X_train, Y_train, verbose=0)
scoretest = model.evaluate(X_test, Y_test, verbose=0)
print('train score:', scoretrain[0])
print('train accuracy:', scoretrain[1])
print('Test score:', scoretest[0])
print('Test accuracy:', scoretest[1])
print("模型评估结束")

pred_y = model.predict(X_test)
pred_label = np.argmax(pred_y, axis=1)
true_label = np.argmax(Y_test, axis=1)

confusion_mat = confusion_matrix(true_label, pred_label)

print(confusion_mat)

plot_confusion_matrix(confusion_mat, classes=range(11))



plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("model acc")
plt.ylabel("acc")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.show()
