import cv2
import numpy as np 
import os
from random import shuffle
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation,Dropout
from keras.layers import Conv2D,MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#K.set_image_dim_ordering('tf')


train_dir='./humanActivityRecognisationImages/train_dir' 
img_size=50
lr=1e-3


MODEL_NAME = 'humanActivityRecognisation-{}-{}.model'.format(lr, 'conv-basic') # just so we remember which saved model is which, sizes must match

def label_images(img):
	word_label=img.split('.')[0]

	if word_label=='boxing': 
		return [1,0,0,0,0,0]
	elif word_label=='handclapping':
		return [0,1,0,0,0,0]
	elif word_label=='handwaving':
		return [0,0,1,0,0,0]
	elif word_label=='jogging':
		return [0,0,0,1,0,0]
	elif word_label=='running':
		return [0,0,0,0,1,0]
	elif word_label=='walking':
		return [0,0,0,0,0,1]

def create_train_data():
	training_data=[]
	for folder in (os.listdir(train_dir)):
		for img in os.listdir(os.path.join(train_dir,folder)):
			label=label_images(img)
			path=os.path.join(train_dir,folder,img)
			if os.path.exists(path):
				img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
				img=cv2.resize(img,(img_size,img_size))
				training_data.append([np.array(img),np.array(label)])

	print(len(training_data))
	shuffle(training_data)
	np.save('training_data.npy',training_data)
	return training_data


#train_data=create_train_data()
train_data=np.load('training_data.npy')

batch_size=32
num_classes=6
hm_epochs=1


train=train_data[:-100]
test=train_data[-100:]

x_train=np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
y_train=[i[1] for i in train]

x_test=np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
y_test=np.array([i[1] for i in test])
'''
#y_train=keras.utils.np_utils.to_categorical(y_train)
#y_test=keras.utils.np_utils.to_categorical(y_test)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255


#print(y_train.shape)
print(x_train.shape[1:])


def model_network():
  model=Sequential()
  model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',input_shape=x_train.shape[1:]))
  model.add(Activation('relu'))
  model.add(Conv2D(32,(3,3),strides=(1,1),padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  
  model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes,activation='softmax'))
  
  
  opt=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  
  model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  
  return model


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

cnn_n=model_network()
cnn_n.summary()
cnn_n.fit(x_train,y_train,epochs=hm_epochs,batch_size=batch_size)
#cnn_n.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),epochs=hm_epochs,validation_data=(x_test,y_test),shuffle=True)


scores = cnn_n.evaluate(x_test, y_test, verbose=0)
print('score=',score)
'''



convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

model.fit({'input': x_train}, {'targets': y_train}, n_epoch=3, validation_set=({'input': x_test}, {'targets': y_test}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)