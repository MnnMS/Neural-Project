import tensorflow as tf
import numpy as np
import tflearn
import os
from tqdm import tqdm
import cv2
import pandas as pd
from tensorflow.python.framework import ops
from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from Inception_paras import *



def create_train_data():
    X = []
    Y = []
    for img in tqdm(os.listdir(TRAIN_DIR_A)):
        path = os.path.join(TRAIN_DIR_A, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        X.append(np.array(img_data))
        Y.append([1, 0])
    for img in tqdm(os.listdir(TRAIN_DIR_N)):
        path = os.path.join(TRAIN_DIR_N, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        X.append(np.array(img_data))
        Y.append([0, 1])
    np.save(TRAIN_DIR, X)
    np.save(TEST_DIR, Y)
    return X, Y

def create_test_data(model):
    data = []
    names = []
    for img in tqdm(os.listdir(TEST_DATA_DIR)):
        names.append(img)
        path = os.path.join(TEST_DATA_DIR, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data = img_data.reshape(IMG_SIZE, IMG_SIZE,3)
        pred = model.predict([img_data])[0]
        data.append(pred)
    return data, names

def get_dataset():
    if os.path.exists(TRAIN_DIR):
        X = np.load(TRAIN_DIR)
        Y = np.load(TEST_DIR)
    else:
        X, Y = create_train_data()

    #X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=TRAIN_VALID_SPLIT, stratify=Y, shuffle=True)
    return X, Y

def fill_submit_file(model):
    file = pd.read_csv("Submit.csv")

    preds, names = create_test_data(model)
    file['Image'] = names
    output = []
    for pred in preds:
        if pred[0] > pred[1]:
            output.append(1)
        else:
            output.append(0)
    file['Label'] = output
    file = file.iloc[:, 1:]
    file.to_csv("Submit.csv")

#ops.reset_default_graph()
X, Y = get_dataset()
X = np.array (X).reshape(-1,IMG_SIZE,IMG_SIZE,3)

conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE,3],name='input')

conv1 = conv_2d(conv_input, 67, 7, strides=2, activation='relu')
pool1 = max_pool_2d(conv1, 3, strides=2)
pool1 = local_response_normalization(pool1)
conv2 = conv_2d(pool1, 64, 1, activation='relu')
conv3 = conv_2d(conv2, 192, 3, activation='relu')
conv3 = local_response_normalization(conv3)
pool3 = max_pool_2d(conv3, 3, strides=2)

conv1_con1_1x1 = conv_2d(pool3, 64, 1, activation='relu')
conv2_con1_1x1 = conv_2d(pool3, 96, 1, activation='relu')
conv2_con1_3x3 = conv_2d(conv2_con1_1x1, 128, 3, activation='relu')
conv3_con1_1x1 = conv_2d(pool3, 16, 1, activation='relu')
conv3_con1_5x5 = conv_2d(conv3_con1_1x1, 32, 5, activation='relu')
pool_con1 = max_pool_2d(pool3, 3, strides=1)
conv4_con1_1x1 = conv_2d(pool_con1, 32, 1, activation='relu')
concat1 = merge([conv1_con1_1x1, conv2_con1_3x3, conv3_con1_5x5, conv4_con1_1x1], mode='concat', axis=3)

conv1_con2_1x1 = conv_2d(concat1, 128, 1, activation='relu')
conv2_con2_1x1 = conv_2d(concat1, 128, 1, activation='relu')
conv2_con2_3x3 = conv_2d(conv2_con2_1x1, 192, 3, activation='relu')
conv3_con2_1x1 = conv_2d(concat1, 32, 1, activation='relu')
conv3_con2_5x5 = conv_2d(conv3_con2_1x1, 96, 5, activation='relu')
pool_con2 = max_pool_2d(concat1, 3, strides=1)
conv4_con2_1x1 = conv_2d(pool_con2, 64, 1, activation='relu')
concat2 = merge([conv1_con2_1x1, conv2_con2_3x3, conv3_con2_5x5, conv4_con2_1x1], mode='concat', axis=3)
pool4 = max_pool_2d(concat2, 3, strides=2)

conv1_con3_1x1 = conv_2d(pool4, 192, 1, activation='relu')
conv2_con3_1x1 = conv_2d(pool4, 96, 1, activation='relu')
conv2_con3_3x3 = conv_2d(conv2_con3_1x1, 208, 3, activation='relu')
conv3_con3_1x1 = conv_2d(pool4, 16, 1, activation='relu')
conv3_con3_5x5 = conv_2d(conv3_con3_1x1, 48, 5, activation='relu')
pool_con3 = max_pool_2d(pool4, 3, strides=1)
conv4_con3_1x1 = conv_2d(pool_con3, 64, 1, activation='relu')
concat3 = merge([conv1_con3_1x1, conv2_con3_3x3, conv3_con3_5x5, conv4_con3_1x1], mode='concat', axis=3)

conv1_con4_1x1 = conv_2d(concat3, 160, 1, activation='relu')
conv2_con4_1x1 = conv_2d(concat3, 112, 1, activation='relu')
conv2_con4_3x3 = conv_2d(conv2_con4_1x1, 224, 3, activation='relu')
conv3_con4_1x1 = conv_2d(concat3, 24, 1, activation='relu')
conv3_con4_5x5 = conv_2d(conv3_con4_1x1, 64, 5, activation='relu')
pool_con4 = max_pool_2d(concat3, 3, strides=1)
conv4_con4_1x1 = conv_2d(pool_con4, 64, 1, activation='relu')
concat4 = merge([conv1_con4_1x1, conv2_con4_3x3, conv3_con4_5x5, conv4_con4_1x1], mode='concat', axis=3)

conv1_con5_1x1 = conv_2d(concat4, 128, 1, activation='relu')
conv2_con5_1x1 = conv_2d(concat4, 128, 1, activation='relu')
conv2_con5_3x3 = conv_2d(conv2_con5_1x1, 256, 3, activation='relu')
conv3_con5_1x1 = conv_2d(concat4, 24, 1, activation='relu')
conv3_con5_5x5 = conv_2d(conv3_con5_1x1, 64, 5, activation='relu')
pool_con5 = max_pool_2d(concat4, 3, strides=1)
conv4_con5_1x1 = conv_2d(pool_con5, 64, 1, activation='relu')
concat5 = merge([conv1_con5_1x1, conv2_con5_3x3, conv3_con5_5x5, conv4_con5_1x1], mode='concat', axis=3)

conv1_con6_1x1 = conv_2d(concat5, 112, 1, activation='relu')
conv2_con6_1x1 = conv_2d(concat5, 144, 1, activation='relu')
conv2_con6_3x3 = conv_2d(conv2_con6_1x1, 288, 3, activation='relu')
conv3_con6_1x1 = conv_2d(concat5, 32, 1, activation='relu')
conv3_con6_5x5 = conv_2d(conv3_con6_1x1, 64, 5, activation='relu')
pool_con6 = max_pool_2d(concat5, 3, strides=1)
conv4_con6_1x1 = conv_2d(pool_con6, 64, 1, activation='relu')
concat6 = merge([conv1_con6_1x1, conv2_con6_3x3, conv3_con6_5x5, conv4_con6_1x1], mode='concat', axis=3)

conv1_con7_1x1 = conv_2d(concat6, 256, 1, activation='relu')
conv2_con7_1x1 = conv_2d(concat6, 160, 1, activation='relu')
conv2_con7_3x3 = conv_2d(conv2_con7_1x1, 320, 3, activation='relu')
conv3_con7_1x1 = conv_2d(concat6, 32, 1, activation='relu')
conv3_con7_5x5 = conv_2d(conv3_con7_1x1, 128, 5, activation='relu')
pool_con7 = max_pool_2d(concat6, 3, strides=1)
conv4_con7_1x1 = conv_2d(pool_con7, 128, 1, activation='relu')
concat7 = merge([conv1_con7_1x1, conv2_con7_3x3, conv3_con7_5x5, conv4_con7_1x1], mode='concat', axis=3)
pool5 = max_pool_2d(concat7, 3, strides=2)

conv1_con8_1x1 = conv_2d(pool5, 256, 1, activation='relu')
conv2_con8_1x1 = conv_2d(pool5, 160, 1, activation='relu')
conv2_con8_3x3 = conv_2d(conv2_con8_1x1, 320, 3, activation='relu')
conv3_con8_1x1 = conv_2d(pool5, 32, 1, activation='relu')
conv3_con8_5x5 = conv_2d(conv3_con8_1x1, 128, 5, activation='relu')
pool_con8 = max_pool_2d(pool5, 3, strides=1)
conv4_con8_1x1 = conv_2d(pool_con8, 128, 1, activation='relu')
concat8 = merge([conv1_con8_1x1, conv2_con8_3x3, conv3_con8_5x5, conv4_con8_1x1], mode='concat', axis=3)

conv1_con9_1x1 = conv_2d(concat8, 384, 1, activation='relu')
conv2_con9_1x1 = conv_2d(concat8, 192, 1, activation='relu')
conv2_con9_3x3 = conv_2d(conv2_con9_1x1, 384, 3, activation='relu')
conv3_con9_1x1 = conv_2d(concat8, 48, 1, activation='relu')
conv3_con9_5x5 = conv_2d(conv3_con9_1x1, 128, 5, activation='relu')
pool_con9 = max_pool_2d(concat8, 3, strides=1)
conv4_con9_1x1 = conv_2d(pool_con9, 128, 1, activation='relu')
concat9 = merge([conv1_con9_1x1, conv2_con9_3x3, conv3_con9_5x5, conv4_con9_1x1], mode='concat', axis=3)

pool6 = avg_pool_2d(concat9, 7, strides=1)

fully_layer = fully_connected(pool6, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

fully_layer = fully_connected(pool6,2, activation='softmax')

cnn_layers = regression(fully_layer, optimizer='momentum',
                        loss='categorical_crossentropy',
                        learning_rate=LR, name='targets')

model = tflearn.DNN(cnn_layers, checkpoint_path='model_googlenet',
                    max_checkpoints=1, tensorboard_verbose=2)

if os.path.exists('savedInception/model.tfl.meta') and not FORCE_TRAIN:
    model.load('savedInception/model.tfl')
else:
    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS,
          validation_set=0,
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('savedInception/model.tfl')


test_acc = model.evaluate(X, Y)
print(test_acc)


fill_submit_file(model)
