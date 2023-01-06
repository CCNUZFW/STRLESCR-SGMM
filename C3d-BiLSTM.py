import keras
from keras.layers import LSTM, Dense, Dropout, Flatten,Conv2D,Bidirectional,GRU
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import csv


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


start = time.time()
input_feature = keras.layers.Input(shape=[80, 12, 16, 1])


def scheduler(epoch):
    # 每隔30个epoch，学习率减小为原来的1/10
    if epoch % 30 == 0 and epoch != 0:
        lr = K.get_value(m.optimizer.lr)
        K.set_value(m.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(m.optimizer.lr)


def build_model():
    hidden20 = keras.layers.Conv3D(16, (1, 1, 1), activation='relu', padding='valid', data_format='channels_last',name='layer_con0')(input_feature)
    hidden21 = keras.layers.Conv3D(16, (1, 3, 3), activation='relu', padding='valid', data_format='channels_last',name='layer_con1')(hidden20)
    hidden211 = keras.layers.BatchNormalization()(hidden21)
    hidden22 = keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid')(hidden211)
    hidden23 = keras.layers.Conv3D(16, (1, 3, 3), activation='relu', padding='valid', data_format='channels_last',name='layer_con2')(hidden22)
    hidden231 = keras.layers.BatchNormalization()(hidden23)
    # hidden24 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(hidden231)
    hidden25 = keras.layers.Conv3D(16, (1, 1, 1), activation='relu', padding='valid', data_format='channels_last',name='layer_con3')(hidden231)
    hidden251 = keras.layers.BatchNormalization()(hidden25)
    hidden26 = keras.layers.AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid')(hidden251)
    #
    hidden100 = keras.layers.Reshape((80, 384))(hidden26)
    hidden101 = Bidirectional(LSTM(128, name='layer_lstm1', return_sequences=True))(hidden100)
    hidden102 = Bidirectional(LSTM(64, name='layer_lstm2', return_sequences=True))(hidden101)

    hiddenatt10 = keras.layers.Reshape((80, 128, 1))(hidden102)
    hiddenatt11 = keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                      data_format='channels_last', name='att5')(hiddenatt10)
    hiddenatt12 = keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                      data_format='channels_last', name='att6')(hiddenatt11)
    hiddenatt13 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(hiddenatt12)
    hiddenatt14 = keras.layers.Reshape([1080])(hiddenatt13)
    hiddenatt15 = keras.layers.Dense(80 * 128, activation='sigmoid')(hiddenatt14)
    hiddenatt16 = keras.layers.Reshape((80, 128))(hiddenatt15)
    hiddenatt17 = keras.layers.multiply([hiddenatt16, hidden102])
    hiddenatt18 = keras.layers.Reshape([80 * 128])(hiddenatt17)
    hiddenatt19 = keras.layers.Dense(1024, activation='relu', name='layers_fully_att')(hiddenatt18)

    hidden04 = keras.layers.Dense(45, activation='softmax', name='layer_softmax')(hiddenatt19)
    model = keras.models.Model(inputs=[input_feature], outputs=[hidden04])
    return model


if __name__ == '__main__':
    m = build_model()
    m.summary()
    m.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 数据准备
    y_train = np.zeros((45*514, 45), 'float')
    y_test = np.zeros((45*128, 45), 'float')

    for i in range(45):
        y_train[i * 514:(i + 1) * 514, i] = 1
        y_test[i * 128:(i + 1) * 128, i] = 1

    # # 数据准备Soft label
    # y_train = np.random.uniform(0, 0.05, (45*514, 45))
    # y_test = np.random.uniform(0, 0.05, (45*128, 45))
    #
    # for i in range(45):
    #     y_train[i * 514:(i + 1) * 514, i] = np.random.uniform(0.9, 1.05)
    #     y_test[i * 128:(i + 1) * 128, i] = np.random.uniform(0.9, 1.05)

    path = 'SGMM_train.csv'
    inputs_train = pd.read_csv(path, header=None)
    x_train_input = inputs_train.values

    path = 'SGMM_test.csv'
    inputs_test = pd.read_csv(path, header=None)
    x_test_input = inputs_test.values

    X_train = x_train_input.reshape(1, x_train_input.shape[0], 80, 12, 16)
    X_train = X_train.transpose(1, 2, 3, 4, 0)
    X_test = x_test_input.reshape(1, x_test_input.shape[0], 80, 12, 16)
    X_test = X_test.transpose(1, 2, 3, 4, 0)

    reduce_lr = LearningRateScheduler(scheduler)
    m.fit([X_train], [y_train], epochs=100, batch_size=128, callbacks=[reduce_lr])

    accuracy = m.evaluate([X_test], [y_test])
    print(m.metrics_names)
    print('accuracy:', accuracy)

end = time.time()
print("Running time: %d minutes %d seconds " % (((end-start)//60), ((end-start) % 60)))

