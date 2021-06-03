import cv2
import numpy as np
import pandas as pd
import os
from random import shuffle
from tqdm import tqdm
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from model_paras import *
import random

def rotate_image(image):
  arr = [random.randrange(5, 46), random.randrange(-46, -5)]
  angle = arr[random.randrange(2)]
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def create_test_data(model):
    data = []
    names = []
    imgs = os.listdir(TEST_DATA_DIR)
    shuffle(imgs)
    for img in tqdm(imgs):
        names.append(img)
        path = os.path.join(TEST_DATA_DIR, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        pred = model.predict([img_data])[0]
        data.append(pred)
    return data, names

def create_train_data():
    training_data = []
    augm_data = []

    for img in tqdm(os.listdir(TRAIN_DIR_A)):
        path = os.path.join(TRAIN_DIR_A, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img2 = rotate_image(img_data)
        img3 = cv2.flip(img_data, 1)

        training_data.append([np.array(img_data), np.array([1, 0])])
        augm_data.append([np.array(img2), np.array([1, 0])])
        augm_data.append([np.array(img3), np.array([1, 0])])

    for img in tqdm(os.listdir(TRAIN_DIR_N)):
        path = os.path.join(TRAIN_DIR_N, img)
        img_data = cv2.imread(path, RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img2 = rotate_image(img_data)
        img3 = cv2.flip(img_data, 1)

        training_data.append([np.array(img_data), np.array([0, 1])])
        augm_data.append([np.array(img2), np.array([0, 1])])
        augm_data.append([np.array(img3), np.array([0, 1])])


    shuffle(training_data)
    shuffle(augm_data)
    np.save('train_data.npy', training_data)
    np.save('aug_data.npy', augm_data)
    return training_data, augm_data

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

if os.path.exists('train_data.npy'):
    train_data =np.load('train_data.npy', allow_pickle=True)
    aug_data =np.load('aug_data.npy', allow_pickle=True)
else:
    train_data, aug_data = create_train_data()

tst_sz = -300
train = train_data[:tst_sz]
test = train_data[tst_sz:]

train = np.concatenate((train, aug_data))
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_train = X_train / 255
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_test = X_test / 255
y_test = [i[1] for i in test]

ops.reset_default_graph()

aug = tflearn.data_augmentation.ImageAugmentation()
aug.add_random_blur(0.3)
aug.add_random_flip_leftright()
aug.add_random_rotation(-45)

conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

conv1 = conv_2d(conv_input, 32, KERNAL_SIZE, activation='relu')
pool1 = max_pool_2d(conv1, KERNAL_SIZE)

conv2 = conv_2d(pool1, 64, KERNAL_SIZE, activation='relu')
pool2 = max_pool_2d(conv2, KERNAL_SIZE)

conv3 = conv_2d(pool2, 128, KERNAL_SIZE, activation='relu', regularizer='L2')
pool3 = max_pool_2d(conv3, KERNAL_SIZE)

conv4 = conv_2d(pool3, 64, KERNAL_SIZE, activation='relu')
pool4 = max_pool_2d(conv4, KERNAL_SIZE)

conv5 = conv_2d(pool4, 32, KERNAL_SIZE, activation='relu')
pool5 = max_pool_2d(conv5, KERNAL_SIZE)

fully_layer = fully_connected(pool5, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 2, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, best_checkpoint_path='saved/best_models/best', best_val_accuracy=0.88)

if os.path.exists('saved/model.tfl.meta') and not FORCE_TRAIN:
    model.load('saved/model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=EPOCHS,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('saved/model.tfl')

test_tmp = np.array(y_test)
test_acc = model.evaluate(X_test, test_tmp)
print("################################## \n" + str(test_acc))

#fill_submit_file(model)



