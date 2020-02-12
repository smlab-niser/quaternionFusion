#!/usr/bin/env python
# coding: utf-8


import sys
sys.setrecursionlimit(10000)
import logging                      as L
import numpy                        as np
from keras                          import *      
from quaternion_layers.utils        import Params, GetR, GetI, GetJ, GetK
from quaternion_layers.conv         import QuaternionConv2D, QuaternionConv1D
from quaternion_layers.dense        import QuaternionDense
from quaternion_layers.bn           import QuaternionBatchNormalization
import keras
from keras.callbacks                import Callback, ModelCheckpoint, LearningRateScheduler
from keras.layers                   import Layer, AveragePooling2D, Conv2D, Conv1D, Dot,Dropout, Permute,Multiply, MaxPooling1D,MaxPooling2D, AveragePooling3D, add, Softmax, Embedding, Add, concatenate, Concatenate, Input, Flatten, Dense, Convolution2D, BatchNormalization, Activation, Reshape, ConvLSTM2D, Conv2D
from keras.models                   import Model, load_model, save_model
from keras.preprocessing.image      import ImageDataGenerator
from keras.regularizers             import l2
from keras_adabound                 import AdaBound
from keras.utils.np_utils           import to_categorical
import keras.backend                as K
from tensorflow                     import image
from keras.preprocessing.text       import Tokenizer
from keras.preprocessing.sequence   import pad_sequences
import argparse                     as Ap
from pathlib                        import Path
from PIL                            import Image
import emoji
import re
import string
import os
import os.path
from os                             import path
import skimage
from skimage                        import color
from numpy                          import asarray
import tensorflow                   as tf
from keras                          import * 
import matplotlib.pyplot            as plt
from skimage                        import data
from skimage.color                  import rgb2gray
import matplotlib.pyplot            as plt


#PREPROCESSING
# Open and load the downloaded dataset (where path= location of the dataset folder where it is downloaded):


path = Path('')#path where the dataset is located
# join two paths using path = os.join(path, a) where a is the file one wants to read

import json
with open('path/MMHS150K/MMHS150K_GT.json') as f:
  data = json.load(f)


# Replace the emojis in the tweet text with their meanings, remove the urls and usernames, hashtags & underscores.


for key in data.keys():
        data[key]['tweet_text'] = emoji.demojize(data[key]['tweet_text']) 
        data[key]['tweet_text'] = re.sub(r'http\S+', '',data[key]['tweet_text'])
        data[key]['tweet_text'] = re.sub(r'@\S+', '',data[key]['tweet_text'])
        data[key]['tweet_text'] = re.sub(r'#', '',data[key]['tweet_text'])
        data[key]['tweet_text'] = re.sub(r'_', ' ',data[key]['tweet_text'])
        if data[key]['labels']==[0,0,0]:
            data[key]['labels']=0
        else:
            data[key]['labels']=1



# Prepare training data  (Resize the images into desired size. Here, we resize the images in 32x32 pixels due to our current setup)

train_keys= open('path/MMHS150K/splits/train_ids.txt')
train_keys= train_keys.read()
train_keys= train_keys.splitlines()

path1           = Path('path/MMHS150K/img_resized')
path2           = Path('path/MMHS150K/img_txt')
y_train_t       = np.zeros((134823,1))
docs_train_t    = ["" for x in range(134823)]
docs_train_img  = ["" for x in range(134823)]
image_train     = np.zeros((134823,32,32,3))
null           = np.zeros((32,32))
zeros          = np.zeros((32,32))
i=0
for keys in train_keys:
    path_im          = ""
    path_im_txt      = ""
    a                = ""
    b                = ""
    y_train_t[i]     = data[keys]['labels']
    docs_train_t[i]  = data[keys]['tweet_text']
    a                = a.join([keys, ".jpg"])
    path_im          = os.path.join(path1, a)
    img_t            = Image.open(path_im)
    if img_t.mode!='RGB':
        null=img_t.resize((32,32))
        image_train[i]= np.stack([null, zeros, zeros], axis=-1)
    else:
        image_train[i]= img_t.resize((32,32))
    b                = b.join([keys, ".json"])
    path_im_txt      = os.path.join(path2, b)
    if path.exists(path_im_txt):
        img_text         = json.load(open(path_im_txt))
        docs_train_img[i] = img_text['img_text']
    i=i+1


# Prepare Test data 


test_keys= open('path/MMHS150K/splits/test_ids.txt')
test_keys= test_keys.read()
test_keys= test_keys.splitlines()

path1           = Path('path/MMHS150K/img_resized')
path2           = Path('path/MMHS150K/img_txt')
y_test_t       = np.zeros((10000,1))
docs_test_t    = ["" for x in range(10000)]
docs_test_img  = ["" for x in range(10000)]
image_test     = np.zeros((10000,32,32,3))
null           = np.zeros((32,32))
zeros          = np.zeros((32,32))
i=0
for keys in test_keys:
    path_im          = ""
    path_im_txt      = ""
    a                = ""
    b                = ""
    y_test_t[i]     = data[keys]['labels']
    docs_test_t[i]  = data[keys]['tweet_text']
    a                = a.join([keys, ".jpg"])
    path_im          = os.path.join(path1, a)
    img_t            = Image.open(path_im)
    if img_t.mode!='RGB':
        null=img_t.resize((32,32))
        image_test[i]= np.stack([null, zeros, zeros], axis=-1)
    else:
        image_test[i]   = img_t.resize((32,32))
    b                = b.join([keys, ".json"])
    path_im_txt      = os.path.join(path2, b)
    if path.exists(path_im_txt):
        img_text         = json.load(open(path_im_txt))
        docs_test_img[i] = img_text['img_text']
    i=i+1


    
#We change the labels into 2 categories 
num_classes = 2
y_train_t   = keras.utils.to_categorical(y_train_t, num_classes)
y_test_t    = keras.utils.to_categorical(y_test_t, num_classes)


#Tweet text pre-processing
#Train_Tweet text tokenizing
t = Tokenizer()
t.fit_on_texts(docs_train_t)
vocab_size_t = len(t.word_index) + 1
encoded_docs_train_t = t.texts_to_sequences(docs_train_t)
print(encoded_docs_train_t)


#Padding
max_length = 150
padded_docs_train_t = pad_sequences(encoded_docs_train_t, maxlen=max_length, padding='post')
print(padded_docs_train_t)


#Preparing test data
encoded_docs_test_t = t.texts_to_sequences(docs_test_t)
padded_docs_test_t = pad_sequences(encoded_docs_test_t, maxlen=150, padding='post')
print(padded_docs_test_t)


#Retrieve pretrained embeddings from GloVe with dim_words=100


path= Path('') #location where glove is extracted
embeddings_index = dict()
f = open(path/'glove.twitter.27B.100d.txt', encoding="utf8")
for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


#Creating Matrix Embedding for our tweet text dictionary
embedding_matrix_t = np.zeros((vocab_size_t+1, 100))
for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
                embedding_matrix_t[i] = embedding_vector



#___________________________________________________xxxxx_________________________________________________________


#Image text pre-processing

#Train_Image text tokenizing
t1 = Tokenizer()
t1.fit_on_texts(docs_train_img)
vocab_size_im = len(t1.word_index) + 1
encoded_docs_train_img = t1.texts_to_sequences(docs_train_img)
print(encoded_docs_train_img)

#Padding
max_length = 150
padded_docs_train_img = pad_sequences(encoded_docs_train_img, maxlen=max_length, padding='post')
print(padded_docs_train_img)


#Preparing test data for image text
encoded_docs_test_img = t1.texts_to_sequences(docs_test_img)
padded_docs_test_img = pad_sequences(encoded_docs_test_img, maxlen=max_length, padding='post')
print(padded_docs_test_img)

#Creating Matrix Embedding
embedding_matrix_im = np.zeros((vocab_size_im+1, 100))
for word, i in t1.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
                embedding_matrix_im[i] = embedding_vector


max_words_im =vocab_size_im+1
max_words_t =vocab_size_t+1


#__________________________________________________xxxxxx_______________________________________________________

#Train_Image pre-processing

image_train_g= np.zeros((134823,32,32,1))
image_train_g = rgb2gray(image_train)
image_train_g=image_train_g.reshape((134823,32,32,1))


#Test_Image pre-processing
image_test_g= np.zeros((10000,32,32,1))
image_test_g = rgb2gray(image_test)
image_test_g=image_test_g.reshape((10000,32,32,1))


#__________________________________________________xxxxxx_______________________________________________________

#Model_1_Q
#QUARC 

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= QuaternionBatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= QuaternionBatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= QuaternionConv2D(16, (2,2), activation='sigmoid', padding='same')(add)
bn2= QuaternionBatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= QuaternionConv2D(8, (2, 2), activation='tanh', padding='same')(l24)
bn3= QuaternionBatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= QuaternionConv2D(8, (2, 2), activation='relu', padding='same')(l26)
bn4= QuaternionBatchNormalization()(l27)


p= Flatten()(bn4)
p1= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)

#Calculating the dependency ("attention") of p(query) on t_text(context) 
l31_t_p     = QuaternionDense(25, activation=None)(p)
x_t_p       = Dot(axes=-1)([l12_t, l31_t_p])
l32_t_p     = Softmax()(x_t_p)
l32_t_p     = Reshape((150,))(l32_t_p)
permute_t_p = Permute((2,1))(l12_t)
l33_t_p     = Dot(axes=-1)([permute_t_p, l32_t_p])

#Calculating the dependency ("attention") of p(query) on im_text(context) 
l31_i_p     = QuaternionDense(25, activation=None)(p)
x_i_p       = Dot(axes=-1)([l12_im, l31_i_p])
l32_i_p     = Softmax()(x_i_p)
l32_i_p     = Reshape((150,))(l32_i_p)
permute_i_p = Permute((2,1))(l12_im)
l33_i_p     = Dot(axes=-1)([permute_i_p, l32_i_p])


#Calculating a, the dependency ("attention") of t_text on im_text 
# 1. context: im_text ; query: t_text
l31_i_t     = QuaternionDense(25, activation=None)(flat_t)
x_i_t       = Dot(axes=-1)([l12_im, l31_i_t])
l32_i_t     = Softmax()(x_i_t)
l32_i_t     = Reshape((150,))(l32_i_t)
permute_i_t = Permute((2,1))(l12_im)
l33_i_t     = Dot(axes=-1)([permute_i_t, l32_i_t])


# 2. context: t_text ; query: im_text
l31_t_i     = QuaternionDense(25, activation=None)(flat_im)
x_t_i       = Dot(axes=-1)([l12_t, l31_t_i])
l32_t_i     = Softmax()(x_t_i)
l32_t_i     = Reshape((150,))(l32_t_i)
permute_t_i = Permute((2,1))(l12_t)
l33_t_i     = Dot(axes=-1)([permute_t_i, l32_t_i])

#Weighted sum of above two

l33_sum     = Add()([l33_t_i, l33_i_t])


#Fusion1 (Symmetric-Gated) (tweet text & photo vector)
l41_t1= Dense(1, activation=None)(l33_t_p)
l42_t1= Dense(1, activation=None)(l31_t_p)
l43_t1= Add()([l41_t1, l42_t1])
l44_t1= Activation('sigmoid')(l43_t1)
l41_p1= Dense(1, activation=None)(l33_t_p)
l42_p1= Dense(1, activation=None)(l31_t_p)
l43_p1= Add()([l41_p1, l42_p1])
l44_p1= Activation('sigmoid')(l43_p1)
l41_m1= QuaternionDense(25, activation=None)(l33_t_p)
l42_m1= QuaternionDense(25, activation=None)(l31_t_p)
l43_m1= Add()([l41_m1, l42_m1])
l44_m1= Activation('tanh')(l43_m1)
l44_11= Multiply()([l44_t1, l33_t_p])
l44_21= Multiply()([l44_p1, l44_m1])
l441= Add()([l44_11, l44_21])
l441= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l441)
l441= Dropout(0.2)(l441)

#Fusion2 (Symmetric-Gated) (image text & photo vector)
l41_t2= Dense(1, activation=None)(l33_i_p)
l42_t2= Dense(1, activation=None)(l31_i_p)
l43_t2= Add()([l41_t2, l42_t2])
l44_t2= Activation('sigmoid')(l43_t2)
l41_p2= Dense(1, activation=None)(l33_i_p)
l42_p2= Dense(1, activation=None)(l31_i_p)
l43_p2= Add()([l41_p2, l42_p2])
l44_p2= Activation('sigmoid')(l43_p2)
l41_m2= QuaternionDense(25, activation=None)(l33_i_p)
l42_m2= QuaternionDense(25, activation=None)(l31_i_p)
l43_m2= Add()([l41_m2, l42_m2])
l44_m2= Activation('tanh')(l43_m2)
l44_12= Multiply()([l44_t2, l33_i_p])
l44_22= Multiply()([l44_p2, l44_m2])
l442= Add()([l44_12, l44_22])
l442= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l442)
l442= Dropout(0.2)(l442)

#Fusion3 (Symmetric-Gated) {avg(tweet text, image text) & photo vector}
l41_t3= Dense(1, activation=None)(l33_sum)
l42_t3= Dense(1, activation=None)(l31_i_p)
l43_t3= Add()([l41_t3, l42_t3])
l44_t3= Activation('sigmoid')(l43_t3)
l41_p3= Dense(1, activation=None)(l33_sum)
l42_p3= Dense(1, activation=None)(l31_i_p)
l43_p3= Add()([l41_p3, l42_p3])
l44_p3= Activation('sigmoid')(l43_p3)
l41_m3= Dense(100, activation=None)(l33_sum)
l42_m3= Dense(100, activation=None)(l31_i_p)
l43_m3= Add()([l41_m3, l42_m3])
l44_m3= Activation('tanh')(l43_m3)
l44_13= Multiply()([l44_t3, l33_sum])
l44_23= Multiply()([l44_p3, l44_m3])
l443= Add()([l44_13, l44_23])
l443= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l443)
l443= Dropout(0.2)(l443)



con = Concatenate(axis=-1)([t1_t, t1_im, p1, l441, l442, l443])

x= QuaternionDense(64, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)



model_1_Q = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_1_Q.summary()
model_1_Q.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

hist1 = model_1_Q.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)


#Model_1_R
#Real with three inputs & multimodal fusion model

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= BatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= BatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= Conv2D(64, (2,2), activation='sigmoid', padding='same')(add)
bn2= BatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= Conv2D(32, (2, 2), activation='tanh', padding='same')(l24)
bn3= BatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= Conv2D(32, (2, 2), activation='relu', padding='same')(l26)
bn4= BatchNormalization()(l27)


p= Flatten()(bn4)
p1= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)

#Calculating the dependency ("attention") of p(query) on t_text(context) 
l31_t_p     = Dense(100, activation=None)(p)
x_t_p       = Dot(axes=-1)([l12_t, l31_t_p])
l32_t_p     = Softmax()(x_t_p)
l32_t_p     = Reshape((150,))(l32_t_p)
permute_t_p = Permute((2,1))(l12_t)
l33_t_p     = Dot(axes=-1)([permute_t_p, l32_t_p])

#Calculating the dependency ("attention") of p(query) on im_text(context) 
l31_i_p     = Dense(100, activation=None)(p)
x_i_p       = Dot(axes=-1)([l12_im, l31_i_p])
l32_i_p     = Softmax()(x_i_p)
l32_i_p     = Reshape((150,))(l32_i_p)
permute_i_p = Permute((2,1))(l12_im)
l33_i_p     = Dot(axes=-1)([permute_i_p, l32_i_p])


#Calculating a, the dependency ("attention") of t_text on im_text 
# 1. context: im_text ; query: t_text
l31_i_t     = Dense(100, activation=None)(flat_t)
x_i_t       = Dot(axes=-1)([l12_im, l31_i_t])
l32_i_t     = Softmax()(x_i_t)
l32_i_t     = Reshape((150,))(l32_i_t)
permute_i_t = Permute((2,1))(l12_im)
l33_i_t     = Dot(axes=-1)([permute_i_t, l32_i_t])


# 2. context: t_text ; query: im_text
l31_t_i     = Dense(100, activation=None)(flat_im)
x_t_i       = Dot(axes=-1)([l12_t, l31_t_i])
l32_t_i     = Softmax()(x_t_i)
l32_t_i     = Reshape((150,))(l32_t_i)
permute_t_i = Permute((2,1))(l12_t)
l33_t_i     = Dot(axes=-1)([permute_t_i, l32_t_i])

#Weighted sum of above two

l33_sum     = Add()([l33_t_i, l33_i_t])


#Fusion1 (Symmetric-Gated) (tweet text & photo vector)
l41_t1= Dense(1, activation=None)(l33_t_p)
l42_t1= Dense(1, activation=None)(l31_t_p)
l43_t1= Add()([l41_t1, l42_t1])
l44_t1= Activation('sigmoid')(l43_t1)
l41_p1= Dense(1, activation=None)(l33_t_p)
l42_p1= Dense(1, activation=None)(l31_t_p)
l43_p1= Add()([l41_p1, l42_p1])
l44_p1= Activation('sigmoid')(l43_p1)
l41_m1= Dense(100, activation=None)(l33_t_p)
l42_m1= Dense(100, activation=None)(l31_t_p)
l43_m1= Add()([l41_m1, l42_m1])
l44_m1= Activation('tanh')(l43_m1)
l44_11= Multiply()([l44_t1, l33_t_p])
l44_21= Multiply()([l44_p1, l44_m1])
l441= Add()([l44_11, l44_21])
l441= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l441)
l441= Dropout(0.2)(l441)

#Fusion2 (Symmetric-Gated) (image text & photo vector)
l41_t2= Dense(1, activation=None)(l33_i_p)
l42_t2= Dense(1, activation=None)(l31_i_p)
l43_t2= Add()([l41_t2, l42_t2])
l44_t2= Activation('sigmoid')(l43_t2)
l41_p2= Dense(1, activation=None)(l33_i_p)
l42_p2= Dense(1, activation=None)(l31_i_p)
l43_p2= Add()([l41_p2, l42_p2])
l44_p2= Activation('sigmoid')(l43_p2)
l41_m2= Dense(100, activation=None)(l33_i_p)
l42_m2= Dense(100, activation=None)(l31_i_p)
l43_m2= Add()([l41_m2, l42_m2])
l44_m2= Activation('tanh')(l43_m2)
l44_12= Multiply()([l44_t2, l33_i_p])
l44_22= Multiply()([l44_p2, l44_m2])
l442= Add()([l44_12, l44_22])
l442= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l442)
l442= Dropout(0.2)(l442)

#Fusion3 (Symmetric-Gated) {avg(tweet text, image text) & photo vector}
l41_t3= Dense(1, activation=None)(l33_sum)
l42_t3= Dense(1, activation=None)(l31_i_p)
l43_t3= Add()([l41_t3, l42_t3])
l44_t3= Activation('sigmoid')(l43_t3)
l41_p3= Dense(1, activation=None)(l33_sum)
l42_p3= Dense(1, activation=None)(l31_i_p)
l43_p3= Add()([l41_p3, l42_p3])
l44_p3= Activation('sigmoid')(l43_p3)
l41_m3= Dense(100, activation=None)(l33_sum)
l42_m3= Dense(100, activation=None)(l31_i_p)
l43_m3= Add()([l41_m3, l42_m3])
l44_m3= Activation('tanh')(l43_m3)
l44_13= Multiply()([l44_t3, l33_sum])
l44_23= Multiply()([l44_p3, l44_m3])
l443= Add()([l44_13, l44_23])
l443= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l443)
l443= Dropout(0.2)(l443)


con = Concatenate(axis=-1)([t1_t, t1_im, p1, l441, l442, l443])

x= Dense(256, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)


model_1_R = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_1_R.summary()
model_1_R.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

hist2 = model_1_R.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)


#______________________________________________________xxx_____________________________________________________________

#Model_2_Q
#Quaternion with three inputs & fusion(tweet_text, p)

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= QuaternionBatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= QuaternionBatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= QuaternionConv2D(16, (2,2), activation='sigmoid', padding='same')(add)
bn2= QuaternionBatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= QuaternionConv2D(8, (2, 2), activation='tanh', padding='same')(l24)
bn3= QuaternionBatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= QuaternionConv2D(8, (2, 2), activation='relu', padding='same')(l26)
bn4= QuaternionBatchNormalization()(l27)


p= Flatten()(bn4)
p1= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)

#Calculating the dependency ("attention") of p(query) on t_text(context) 
l31_t_p     = QuaternionDense(25, activation=None)(p)
x_t_p       = Dot(axes=-1)([l12_t, l31_t_p])
l32_t_p     = Softmax()(x_t_p)
l32_t_p     = Reshape((150,))(l32_t_p)
permute_t_p = Permute((2,1))(l12_t)
l33_t_p     = Dot(axes=-1)([permute_t_p, l32_t_p])




#Fusion1 (Symmetric-Gated) (tweet text & photo vector)
l41_t1= Dense(1, activation=None)(l33_t_p)
l42_t1= Dense(1, activation=None)(l31_t_p)
l43_t1= Add()([l41_t1, l42_t1])
l44_t1= Activation('sigmoid')(l43_t1)
l41_p1= Dense(1, activation=None)(l33_t_p)
l42_p1= Dense(1, activation=None)(l31_t_p)
l43_p1= Add()([l41_p1, l42_p1])
l44_p1= Activation('sigmoid')(l43_p1)
l41_m1= QuaternionDense(25, activation=None)(l33_t_p)
l42_m1= QuaternionDense(25, activation=None)(l31_t_p)
l43_m1= Add()([l41_m1, l42_m1])
l44_m1= Activation('tanh')(l43_m1)
l44_11= Multiply()([l44_t1, l33_t_p])
l44_21= Multiply()([l44_p1, l44_m1])
l441= Add()([l44_11, l44_21])
l441= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l441)
l441= Dropout(0.2)(l441)



con = Concatenate(axis=-1)([t1_t, t1_im, p1, l441])

x= QuaternionDense(64, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)



model_2_Q = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_2_Q.summary()
model_2_Q.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


hist3 = model_2_Q.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)


#______________________________________________________xxx_____________________________________________________________

#Model_2_R
#Real with three inputs & fusion(tweet_text, p)

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= BatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= BatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= Conv2D(64, (2,2), activation='sigmoid', padding='same')(add)
bn2= BatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= Conv2D(32, (2, 2), activation='tanh', padding='same')(l24)
bn3= BatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= Conv2D(32, (2, 2), activation='relu', padding='same')(l26)
bn4= BatchNormalization()(l27)


p= Flatten()(bn4)
p1= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)

#Calculating the dependency ("attention") of p(query) on t_text(context) 
l31_t_p     = Dense(100, activation=None)(p)
x_t_p       = Dot(axes=-1)([l12_t, l31_t_p])
l32_t_p     = Softmax()(x_t_p)
l32_t_p     = Reshape((150,))(l32_t_p)
permute_t_p = Permute((2,1))(l12_t)
l33_t_p     = Dot(axes=-1)([permute_t_p, l32_t_p])



#Fusion1 (Symmetric-Gated) (tweet text & photo vector)
l41_t1= Dense(1, activation=None)(l33_t_p)
l42_t1= Dense(1, activation=None)(l31_t_p)
l43_t1= Add()([l41_t1, l42_t1])
l44_t1= Activation('sigmoid')(l43_t1)
l41_p1= Dense(1, activation=None)(l33_t_p)
l42_p1= Dense(1, activation=None)(l31_t_p)
l43_p1= Add()([l41_p1, l42_p1])
l44_p1= Activation('sigmoid')(l43_p1)
l41_m1= Dense(100, activation=None)(l33_t_p)
l42_m1= Dense(100, activation=None)(l31_t_p)
l43_m1= Add()([l41_m1, l42_m1])
l44_m1= Activation('tanh')(l43_m1)
l44_11= Multiply()([l44_t1, l33_t_p])
l44_21= Multiply()([l44_p1, l44_m1])
l441= Add()([l44_11, l44_21])
l441= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l441)
l441= Dropout(0.2)(l441)



con = Concatenate(axis=-1)([t1_t, t1_im, p1, l441])

x= Dense(256, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)



model_2_R = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_2_R.summary()
model_2_R.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

hist4 = model_2_R.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)


#______________________________________________________xxx_____________________________________________________________

#Model_3_Q
#quaternion with three inputs & fusion(im_text, p)

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= QuaternionBatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= QuaternionBatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= QuaternionConv2D(16, (2,2), activation='sigmoid', padding='same')(add)
bn2= QuaternionBatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= QuaternionConv2D(8, (2, 2), activation='tanh', padding='same')(l24)
bn3= QuaternionBatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= QuaternionConv2D(8, (2, 2), activation='relu', padding='same')(l26)
bn4= QuaternionBatchNormalization()(l27)


p= Flatten()(bn4)
p1= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)

#Calculating the dependency ("attention") of p(query) on im_text(context) 
l31_i_p     = QuaternionDense(25, activation=None)(p)
x_i_p       = Dot(axes=-1)([l12_im, l31_i_p])
l32_i_p     = Softmax()(x_i_p)
l32_i_p     = Reshape((150,))(l32_i_p)
permute_i_p = Permute((2,1))(l12_im)
l33_i_p     = Dot(axes=-1)([permute_i_p, l32_i_p])



#Fusion2 (Symmetric-Gated) (image text & photo vector)
l41_t2= Dense(1, activation=None)(l33_i_p)
l42_t2= Dense(1, activation=None)(l31_i_p)
l43_t2= Add()([l41_t2, l42_t2])
l44_t2= Activation('sigmoid')(l43_t2)
l41_p2= Dense(1, activation=None)(l33_i_p)
l42_p2= Dense(1, activation=None)(l31_i_p)
l43_p2= Add()([l41_p2, l42_p2])
l44_p2= Activation('sigmoid')(l43_p2)
l41_m2= QuaternionDense(25, activation=None)(l33_i_p)
l42_m2= QuaternionDense(25, activation=None)(l31_i_p)
l43_m2= Add()([l41_m2, l42_m2])
l44_m2= Activation('tanh')(l43_m2)
l44_12= Multiply()([l44_t2, l33_i_p])
l44_22= Multiply()([l44_p2, l44_m2])
l442= Add()([l44_12, l44_22])
l442= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l442)
l442= Dropout(0.2)(l442)


con = Concatenate(axis=-1)([t1_t, t1_im, p1, l442])

x= QuaternionDense(64, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)



model_3_Q = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_3_Q.summary()
model_3_Q.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

hist5 = model_3_Q.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)

#______________________________________________________xxx_____________________________________________________________

#Model_3_R
#Real with three inputs & fusion(im_text, p)

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= BatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= BatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= Conv2D(64, (2,2), activation='sigmoid', padding='same')(add)
bn2= BatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= Conv2D(32, (2, 2), activation='tanh', padding='same')(l24)
bn3= BatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= Conv2D(32, (2, 2), activation='relu', padding='same')(l26)
bn4= BatchNormalization()(l27)


p= Flatten()(bn4)
p1= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)

#Calculating the dependency ("attention") of p(query) on im_text(context) 
l31_i_p     = Dense(100, activation=None)(p)
x_i_p       = Dot(axes=-1)([l12_im, l31_i_p])
l32_i_p     = Softmax()(x_i_p)
l32_i_p     = Reshape((150,))(l32_i_p)
permute_i_p = Permute((2,1))(l12_im)
l33_i_p     = Dot(axes=-1)([permute_i_p, l32_i_p])




#Fusion2 (Symmetric-Gated) (image text & photo vector)
l41_t2= Dense(1, activation=None)(l33_i_p)
l42_t2= Dense(1, activation=None)(l31_i_p)
l43_t2= Add()([l41_t2, l42_t2])
l44_t2= Activation('sigmoid')(l43_t2)
l41_p2= Dense(1, activation=None)(l33_i_p)
l42_p2= Dense(1, activation=None)(l31_i_p)
l43_p2= Add()([l41_p2, l42_p2])
l44_p2= Activation('sigmoid')(l43_p2)
l41_m2= Dense(100, activation=None)(l33_i_p)
l42_m2= Dense(100, activation=None)(l31_i_p)
l43_m2= Add()([l41_m2, l42_m2])
l44_m2= Activation('tanh')(l43_m2)
l44_12= Multiply()([l44_t2, l33_i_p])
l44_22= Multiply()([l44_p2, l44_m2])
l442= Add()([l44_12, l44_22])
l442= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l442)
l442= Dropout(0.2)(l442)



con = Concatenate(axis=-1)([t1_t, t1_im, p1, l442])

x= Dense(256, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)



model_3_R = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_3_R.summary()
model_3_R.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


hist6 = model_3_R.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)


#______________________________________________________xxx_____________________________________________________________

#Model_4_Q
#Quaternion with three inputs & fusion(t_sum, p)

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= QuaternionBatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= QuaternionBatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= QuaternionConv2D(16, (2,2), activation='sigmoid', padding='same')(add)
bn2= QuaternionBatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= QuaternionConv2D(8, (2, 2), activation='tanh', padding='same')(l24)
bn3= QuaternionBatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= QuaternionConv2D(8, (2, 2), activation='relu', padding='same')(l26)
bn4= QuaternionBatchNormalization()(l27)


p= Flatten()(bn4)
p1= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)



#Calculating a, the dependency ("attention") of t_text on im_text 
# 1. context: im_text ; query: t_text
l31_i_t     = QuaternionDense(25, activation=None)(flat_t)
x_i_t       = Dot(axes=-1)([l12_im, l31_i_t])
l32_i_t     = Softmax()(x_i_t)
l32_i_t     = Reshape((150,))(l32_i_t)
permute_i_t = Permute((2,1))(l12_im)
l33_i_t     = Dot(axes=-1)([permute_i_t, l32_i_t])


# 2. context: t_text ; query: im_text
l31_t_i     = QuaternionDense(25, activation=None)(flat_im)
x_t_i       = Dot(axes=-1)([l12_t, l31_t_i])
l32_t_i     = Softmax()(x_t_i)
l32_t_i     = Reshape((150,))(l32_t_i)
permute_t_i = Permute((2,1))(l12_t)
l33_t_i     = Dot(axes=-1)([permute_t_i, l32_t_i])

#Weighted sum of above two

l33_sum     = Add()([l33_t_i, l33_i_t])

l31_i_p= QuaternionDense(25, activation=None)(p)

#Fusion3 (Symmetric-Gated) {avg(tweet text, image text) & photo vector}
l41_t3= Dense(1, activation=None)(l33_sum)
l42_t3= Dense(1, activation=None)(l31_i_p)
l43_t3= Add()([l41_t3, l42_t3])
l44_t3= Activation('sigmoid')(l43_t3)
l41_p3= Dense(1, activation=None)(l33_sum)
l42_p3= Dense(1, activation=None)(l31_i_p)
l43_p3= Add()([l41_p3, l42_p3])
l44_p3= Activation('sigmoid')(l43_p3)
l41_m3= Dense(100, activation=None)(l33_sum)
l42_m3= Dense(100, activation=None)(l31_i_p)
l43_m3= Add()([l41_m3, l42_m3])
l44_m3= Activation('tanh')(l43_m3)
l44_13= Multiply()([l44_t3, l33_sum])
l44_23= Multiply()([l44_p3, l44_m3])
l443= Add()([l44_13, l44_23])
l443= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l443)
l443= Dropout(0.2)(l443)



con = Concatenate(axis=-1)([t1_t, t1_im, p1, l443])

x= QuaternionDense(64, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)



model_4_Q = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_4_Q.summary()
model_4_Q.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

hist7 = model_4_Q.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)


#______________________________________________________xxx_____________________________________________________________

#Model_4_R
#Real with three inputs & fusion(t_sum, p)

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= BatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= BatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= Conv2D(64, (2,2), activation='sigmoid', padding='same')(add)
bn2= BatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= Conv2D(32, (2, 2), activation='tanh', padding='same')(l24)
bn3= BatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= Conv2D(32, (2, 2), activation='relu', padding='same')(l26)
bn4= BatchNormalization()(l27)


p= Flatten()(bn4)
p1= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)


#Calculating a, the dependency ("attention") of t_text on im_text 
# 1. context: im_text ; query: t_text
l31_i_t     = Dense(100, activation=None)(flat_t)
x_i_t       = Dot(axes=-1)([l12_im, l31_i_t])
l32_i_t     = Softmax()(x_i_t)
l32_i_t     = Reshape((150,))(l32_i_t)
permute_i_t = Permute((2,1))(l12_im)
l33_i_t     = Dot(axes=-1)([permute_i_t, l32_i_t])


# 2. context: t_text ; query: im_text
l31_t_i     = Dense(100, activation=None)(flat_im)
x_t_i       = Dot(axes=-1)([l12_t, l31_t_i])
l32_t_i     = Softmax()(x_t_i)
l32_t_i     = Reshape((150,))(l32_t_i)
permute_t_i = Permute((2,1))(l12_t)
l33_t_i     = Dot(axes=-1)([permute_t_i, l32_t_i])

#Weighted sum of above two

l33_sum     = Add()([l33_t_i, l33_i_t])

l31_i_p= Dense(100, activation=None)(p)
#Fusion3 (Symmetric-Gated) {avg(tweet text, image text) & photo vector}
l41_t3= Dense(1, activation=None)(l33_sum)
l42_t3= Dense(1, activation=None)(l31_i_p)
l43_t3= Add()([l41_t3, l42_t3])
l44_t3= Activation('sigmoid')(l43_t3)
l41_p3= Dense(1, activation=None)(l33_sum)
l42_p3= Dense(1, activation=None)(l31_i_p)
l43_p3= Add()([l41_p3, l42_p3])
l44_p3= Activation('sigmoid')(l43_p3)
l41_m3= Dense(100, activation=None)(l33_sum)
l42_m3= Dense(100, activation=None)(l31_i_p)
l43_m3= Add()([l41_m3, l42_m3])
l44_m3= Activation('tanh')(l43_m3)
l44_13= Multiply()([l44_t3, l33_sum])
l44_23= Multiply()([l44_p3, l44_m3])
l443= Add()([l44_13, l44_23])
l443= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(l443)
l443= Dropout(0.2)(l443)



con = Concatenate(axis=-1)([t1_t, t1_im, p1, l443])

x= Dense(256, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)



model_4_R = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_4_R.summary()
model_4_R.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


hist8 = model_4_R.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)

#______________________________________________________xxx_____________________________________________________________


#Model_5_Q
#Quaternion with three inputs & simple concatenation

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = QuaternionConv1D(25, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= QuaternionBatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= QuaternionConv2D(16, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= QuaternionBatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= QuaternionConv2D(16, (2,2), activation='sigmoid', padding='same')(add)
bn2= QuaternionBatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= QuaternionConv2D(8, (2, 2), activation='tanh', padding='same')(l24)
bn3= QuaternionBatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= QuaternionConv2D(8, (2, 2), activation='relu', padding='same')(l26)
bn4= QuaternionBatchNormalization()(l27)


p= Flatten()(bn4)
p1= QuaternionDense(32, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)


con = Concatenate(axis=-1)([t1_t, t1_im, p1])

x= QuaternionDense(64, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)



model_5_Q = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_5_Q.summary()
model_5_Q.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


hist9 = model_5_Q.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)


#______________________________________________________xxx_____________________________________________________________


#Model_5_R
#Real with three inputs & simple concatenation

from keras import *
t_text = Input(shape=(150,), name='input1')
im_text = Input(shape=(150,), name='input2')
image = Input(shape=(32,32,3), name='input3')
image_g= Input(shape=(32,32,1), name='input4')

#processing tweet text
e_t   = Embedding(max_words_t, 100, weights=[embedding_matrix_t], input_length=max_length, trainable=False)(t_text)
l12_t = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_t)
l13_t = MaxPooling1D(pool_size=2)(l12_t)
flat_t= Flatten()(l13_t)
t1_t  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_t)
t1_t= Dropout(0.2)(t1_t)

#processing image text
e_im   = Embedding(max_words_im, 100, weights=[embedding_matrix_im], input_length=max_length, trainable=False)(im_text)
l12_im = Conv1D(100, kernel_size=(5), padding='same', activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(e_im)
l13_im = MaxPooling1D(pool_size=2)(l12_im)
flat_im= Flatten()(l13_im)
t1_im  = Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(flat_im)
t1_im= Dropout(0.2)(t1_im)

#processing image
img= Concatenate(axis=-1)([image_g, image])
l21= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1= BatchNormalization()(l21)
l22= MaxPooling2D(pool_size=(2,2))(bn1)

l21_1= Conv2D(64, kernel_size=(2, 2), activation='tanh', padding='same', data_format='channels_last', dilation_rate=2)(img)
bn1_1= BatchNormalization()(l21_1)
l22_1= MaxPooling2D(pool_size=(2,2))(bn1_1)

add= Add()([l22, l22_1])

l23= Conv2D(64, (2,2), activation='sigmoid', padding='same')(add)
bn2= BatchNormalization()(l23)
l24= MaxPooling2D(pool_size=(2, 2))(bn2)

l25= Conv2D(32, (2, 2), activation='tanh', padding='same')(l24)
bn3= BatchNormalization()(l25)
l26= MaxPooling2D(pool_size=(2, 2))(bn3)

l27= Conv2D(32, (2, 2), activation='relu', padding='same')(l26)
bn4= BatchNormalization()(l27)


p= Flatten()(bn4)
p1= Dense(128, activation='sigmoid', kernel_regularizer = regularizers.l2(l = 0.001))(p)
p1= Dropout(0.2)(p1)



con = Concatenate(axis=-1)([t1_t, t1_im, p1])

x= Dense(256, activation='relu', kernel_regularizer = regularizers.l2(l = 0.001))(con)
x= Dropout(0.35)(x)
res= Reshape((256,))(x)
out= Dense(2, activation='softmax')(res)



model_5_R = Model(inputs=[t_text, im_text, image, image_g], outputs=out)
model_5_R.summary()
model_5_R.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

hist10 = model_5_R.fit(x={'input1':padded_docs_train_t, 'input3':image_train, 'input2':padded_docs_train_img, 'input4':image_train_g},  y=y_train_t, epochs=10, verbose=1, validation_data=({'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}, y_test_t), batch_size=256)

#______________________________________________________xxx_____________________________________________________________

#Calculate ROC-AUC
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
models = [model_1_Q, model_1_R, model_2_Q, model_2_R, model_3_Q, model_3_R, model_4_Q, model_4_R, model_5_Q, model_5_R]

j=1
for model in models:
    y_pred_keras = model.predict(x={'input1':padded_docs_test_t, 'input3':image_test, 'input2':padded_docs_test_img, 'input4':image_test_g}).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_t.ravel(), y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    print("The AUC-ROC of model_",j,"is",auc_keras)
    j= j+1

#______________________________________________________xxx_____________________________________________________________

#Plot the graph (Validation loss VS Epoch)
plt.plot(hist1.history['val_loss'])
plt.plot(hist2.history['val_loss'])
plt.plot(hist3.history['val_loss'])
plt.plot(hist4.history['val_loss'])
plt.plot(hist5.history['val_loss'])
plt.plot(hist6.history['val_loss'])
plt.plot(hist7.history['val_loss'])
plt.plot(hist8.history['val_loss'])
plt.plot(hist9.history['val_loss'])
plt.plot(hist10.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend(['model_1_Q', 'model_1_R', 'model_2_Q', 'model_2_R', 'model_3_Q', 'model_3_R', 'model_4_Q', 'model_4_R', 'model_5_Q', 'model_5_R'], loc='upper right')
plt.show()