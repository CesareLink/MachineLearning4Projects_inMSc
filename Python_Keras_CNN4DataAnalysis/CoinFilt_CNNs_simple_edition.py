"""
Created on Thur Apr 23 20:06:36 2020
CNN Model with channel averagepooling layer for EEG signal
@author: Cesare
"""
# -*- coding: utf-8 -*-
#______________________________________________________________________________
import keras
import scipy.io as scio
# use Sequential model building module
from keras.models import Sequential
# import Dense，Dropout，Flatten，Conv2D，MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
# for vision function and backend use
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os.path
from keras import backend as K

#______________________________________________________________________________
# Hyperparameters
batch_size = 64
num_classes = 3
epochs = 300

# input EEG signal dimensions
# 30/62 channels of 200Hz down-sample 1 seconds signal data
eeg_tem, eeg_ver = 30, 200
#set up the list for exporting the loss and acc
final_loss = [None] * 10
final_acc = [None] * 10
final_loss_f = [None] * 20  # save all the loss
final_acc_f = [None] * 20
#______________________________________________________________________________
#the loop for two times test for the same data
for msew in range(1, 2):
#the loop for dataset of different subjects
    for mseq in range(1, 6):  #the number of dataset
#load the data file
        dataFile = 'D:\Data\SEED_DATA\djc_1s_fold{}_2r.mat'.format(mseq)
        print(dataFile)
        #set up the train data and label
        data = scio.loadmat(dataFile)
        x1 = data['train_x{}'.format(mseq)]
        x2 = x1.reshape([8100, 62, 200, 1])
        x_train = x2.astype('float32')
        y1 = data['train_y{}'.format(mseq)]
        y_train = y1.astype('float64')
        #set up the test data and label
        x3 = data['test_x{}'.format(mseq)]
        x4 = x3.reshape([900, 62, 200, 1])
        x_test = x4.astype('float32')
        y2 = data['test_y{}'.format(mseq)]
        y_test = y2.astype('float64')
        #set up input shape
        input_shape = (eeg_tem, eeg_ver, 1)
#______________________________________________________________________________
# Build models with Sequential
        model = Sequential()
        # first layer is the 2D convolution layer
        # filters kernel number=16/32/64
        # kernel_size as (1, 3) in temporal direction and (3, 1)in vertical direction
        # ReLU as the activation function
        # first layer shall includes input_shape, the following layers shall not
        model.add(Conv2D(32, kernel_size=(1, 3), activation='relu', input_shape=input_shape,
                         kernel_initializer='glorot_normal'))  # conv1
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(1, 3), activation='relu', input_shape=input_shape,
                         kernel_initializer='glorot_normal'))  # conv2
        model.add(Conv2D(64, kernel_size=(1, 3), activation='relu', input_shape=input_shape,
                         kernel_initializer='glorot_normal'))  # conv3
        model.add(AveragePooling2D(pool_size=(1, 194), strides=(1, 194)))
        # Add flatten to turn the feature maps to 1D
        #model.add(Flatten())
        #model.add(AveragePooling2D(pool_size=(2, 1),strides=(1,1)))
        # add dense with 128 nodes
        model.add(Dense(128, activation='relu'))
        # another dropout
        model.add(Dropout(0.4))
        # softmax for 3 categories
        model.add(Dense(3, activation='softmax'))
#______________________________________________________________________________
# Build models with Sequential
        # Take Adam as optimizer and set up the hyper-parameters
        #learning rate:0.001, momentum=0.9, 0.999, and epsilon=1e-08
        #once use the SGD, the code is following:
        #sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss=keras.losses.categorical_crossentropy,  # optimizer=keras.optimizers.Adam(),
                      optimizer=adam,
                      metrics=['accuracy'])

        print(model.summary()) # print the model structure for verification

#______________________________________________________________________________
#Training the model
        # training the model, loading the data，verbose=1
        print('Training ------------')
        
        # save the weights for load and test
        weights_file = 'weights/djc-f200-1s-fold{}-{}.h5'.format(mseq, msew)
        print(weights_file)
        #callbacks function for model check point and early stop
        #set up the early-stop module, for avoiding the overfitting
        if os.path.exists(weights_file):
            print("Model loaded.")
        out_dir = "weights/"
        model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                           save_weights_only=False, verbose=1)
        model_earlystop = EarlyStopping(monitor='val_acc', patience=60, verbose=1, mode='max')
        callbacks = [model_earlystop]

        #set a history for the results
        his12 = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1,
                          validation_data=(x_test, y_test))
        finaldata = his12.history['val_acc']
#______________________________________________________________________________
#Evaluate the model
        #start to evaluate
        #verbose=0 means that not to output the logs
        #calculate the loss and accuracy
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        predict_y = model.predict(x_test)
        y_pred = np.argmax(predict_y, axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        #save the results in designative txt document
        file_object = open('D://test1/TXT1/djc_f200_1s_history_fold{}.txt'.format(mseq),'a')
        file_object.write(str(predict_y))
        file_object.write('\n')
        file_object.write('\n')
        file_object.write('\n')
        file_object.close()
        file_object = open('D://test1/TXT2/djc_f200_1s_history_fold{}_1.txt'.format(mseq),'a') 
        file_object.write(str(cm))
        file_object.write('\n')
        file_object.write('\n')
        file_object.write('\n')
        file_object.close()
        #print out the loss and accuracy of evaluation
        final_loss[mseq - 1] = loss
        final_acc[mseq - 1] = accuracy
        final_loss_f[(msew - 1) * 10 + mseq - 1] = loss
        final_acc_f[(msew - 1) * 10 + mseq - 1] = accuracy
        print(final_loss)
        print(final_acc)
#print out the final loss
print('\n')
print(final_loss_f)
print(final_acc_f)
# print(final_loss)
# print(final_acc)
#______________________________________________________________________________
#Supplementary code
#save the best weights and set up the check point
# filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
# callbacks_list = [checkpoint]
## Fit the model on the batches generated by datagen.flow().
# history_callback = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), samples_per_epoch=x_train.shape[0], nb_epoch=epochs, validation_data=(x_test, y_test), callbacks=callbacks_list, verbose=1)
#
##get the logs and save the model
#
# pandas.DataFrame(history_callback.history).to_csv("history.csv")
# model.save('keras_allconv.h5')
# model.save_weights(filepath)
# model.save('pilao_2lei.h5')






