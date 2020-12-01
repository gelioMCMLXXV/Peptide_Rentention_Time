### Neural network with keras (Learning Retention Time) 
### loading packges
#from numpy import loadtxt
#import pandas as pd
#import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
import csv

###The function below initialize the peptide vector with element composition
def ElementalComposition(peptide):
 AMINO_ACIDS=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"] 
 V=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
 for residue in peptide:
     j=AMINO_ACIDS.index(residue)  
     V[j]=V[j]+1
 ###print("V=",V)          
 ###print("Z=",Z)  
 return V;

training_X=([])
training_y=([])
testing_X=([])
testing_y=([])
###Reading training data
with open('/content/drive/My Drive/Data/peptide_rt_training_set1.csv', mode='r') as csv_file:
     csv_reader = csv.reader(csv_file, delimiter=',')
     for row in csv_reader:
         x = ElementalComposition(row[0]);
         training_X.append(x) ###peptide
         training_y.append(float(row[2])) ###retention time
print("training_X=",training_X)
#print("taining_y=",training_y)

###Reading testing data
with open('/content/drive/My Drive/Data/peptide_rt_testing_set1.csv', mode='r') as csv_file:
     csv_reader = csv.reader(csv_file, delimiter=',')
     for row in csv_reader:
         x = ElementalComposition(row[0]) ###peptide
         testing_X.append(x)
         testing_y.append(float(row[2])) ###retention time
#print("testing_X=",testing_X)
#print("testing_y=",testing_y)

### define the Neural Network Model
model = Sequential()
model.add(Dense(20, input_dim=20, activation='relu'))
model.add(Dense(10, input_dim=20, activation='relu'))
model.add(Dense(1, activation='linear'))

### compile the keras model
#opt = SGD(lr=0.01, momentum=1)
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mse'])
#model.compile(loss="mean_squared_error", optimizer= opt, metrics=['mse'])

### fit the keras model on the dataset
history=model.fit(training_X,training_y, validation_data=(testing_X, testing_y), epochs=20, batch_size=10)

### evaluate the keras model
# evaluate the model
_, train_mse = model.evaluate(training_X,training_y, verbose=0)
_, testing_mse = model.evaluate(testing_X,testing_y, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse,testing_mse))

# plot loss during training
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
