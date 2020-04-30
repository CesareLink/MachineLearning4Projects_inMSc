# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:03:58 2018
Under testing
@author: Cesare
"""
#______________________________________________________________________________
#modules
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

#______________________________________________________________________________
#Hyperparameters
learning_rate = 0.01
input_size = 8
time_step = 8
batch_size = 128
epochs = 300

n_classes = 10

#______________________________________________________________________________
#Defination of the plot of confusion matrix
def confusion_matrix(CM, title='CM', cmap=plt.cm.binary):
    plt.imshow(CM, interpolation='nearest', cmap=cmap)    
    plt.colorbar()   
    plt.title(title)
    plt.xticks(np.linspace(0,9,10))    
    plt.yticks(np.linspace(0,9,10))    
    ax = plt.gca()  #get the information of axis
    ax.set_ylim(bottom=9.5, top=-0.5)
    plt.ylabel('Predicted label')    
    plt.xlabel('True label')
    for x_val in range(10):
        for y_val in range(10):
            c = CM[x_val, y_val]
            plt.text(y_val, x_val, "%i" % (c,), color='red', fontsize=9, va='center', ha='center')

#______________________________________________________________________________
#load the data file, the data files are proprecessed in matlab code
digits = datasets.load_digits()  
signals = digits.signals
X = digits.data.T
target_orig = digits.target

x_train = signals[:2500,:,:]
x_test = signals[2500:,:,:]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = keras.utils.to_categorical(target_orig[:2500], n_classes)
y_test = keras.utils.to_categorical(target_orig[2500:], n_classes)

#______________________________________________________________________________
# Build models with Sequential
#model = Sequential()
#model.add(LSTM(128, activation='tanh', batch_input_shape=(None, 
#                 time_step, input_size), unroll=True, return_sequences=True)) 
#model.add(Dropout(0.8))
#model.add(LSTM(32, activation='tanh', batch_input_shape=(None, 
#                       time_step, 128), unroll=True, return_sequences=False))
#model.add(Dropout(0.4))
#model.add(Dense(n_classes))
#model.add(Activation('softmax')) 
#model.summary()
#______________________________________________________________________________
# Build models without Sequential
(_, input_x, imput_y) = x_train.shape 
input_lstm = Input(shape=(input_x, imput_y), name='input_layer')
lstm_1 = LSTM(128, activation='tanh', unroll=True, 
              return_sequences=True, name='lstm_layer_1')(input_lstm)
lstm_1 = Dropout(0.8, name='dropout_lstm1')(lstm_1)
lstm_2 = LSTM(32, activation='tanh', unroll=True, 
              return_sequences=False, name='lstm_layer_2')(lstm_1)
lstm_2 = Dropout(0.4, name='dropout_lstm2')(lstm_2)
output_lstm = Dense(n_classes, activation = 'softmax',
                    name='dense_layer')(lstm_2)
model = Model(input_lstm, output_lstm)
model.summary()

#set optimizer
adam = Adam(lr=learning_rate)
#sgd = SGD(lr=learning_rate, momentum=0.0, nesterov=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


#_____________________________#
#Main#-Train and Evaluate

#training the model
History = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(x_test, y_test)) 
#______________________________________________________________________________
# Evaluate models
#model evaluate-import the evaluate model
scores = model.evaluate(x_test, y_test, verbose=0)
print('LSTM test loss:', scores[0])
print('LSTM test accuracy:', scores[1])
#calculate the loss and accuracy
#loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
#predict_y = model.predict(x_test)
#y_pred = np.argmax(predict_y, axis=1)
#y_true = np.argmax(y_test, axis=1)
#cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
#plot the loss function curve
costs = History.history["loss"]
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("losses")
plt.show()

#calculate the confusion matrix
y_test_predictions = model.predict(x_test)
CM = np.zeros((n_classes,n_classes))
for i in range(y_test.shape[0]):
    CM[int(np.argmax(y_test_predictions[i, :])),
       int(np.argmax(y_test[i, :]))] +=1

#vision
confusion_matrix(CM)







