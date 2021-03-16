### Neural network with keras (Learning Retention Time) 
### loading packges
#from numpy import loadtxt
#import pandas as pd
#import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from matplotlib import pyplot
from sklearn import datasets, linear_model
import csv

###Code Parameters: option, evalue_cutoff
option=0  ###option 0==elemental composition, 1==Peptide Hydrophobicity Eisenberg
evalue_cutoff = 0.001   ###evalue_cutoff used to filter the peptides identifed from RAId

###The function below initialize the peptide vector with Hydrophobicity D. Eisenberg et. al. Faraday Symp. Chem. Soc. 17, 109-120 (1982). 
def PeptideHydrophobicity(peptide):
 AMINO_ACIDS=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"] 
 V=[0.25,0,-0.72,-0.62,0.61,0.16,-0.4,0.73,-1.1,0.53,0.26,-0.64,-0.07,-0.69,-1.8,-0.26,-0.18,0.54,0.37,0.02,0]
 V[20]=len(peptide)
 for residue in peptide:
     j=AMINO_ACIDS.index(residue)  
     V[j]=V[j]+1
 ###print("V=",V)          
 ###print("Z=",Z)  
 return V;

###The function below initialize the peptide vector with element composition
def ElementalComposition(peptide):
 AMINO_ACIDS=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"] 
 V=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
 V[20]=len(peptide)
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
count = 0
###Reading training data
with open('/content/drive/My Drive/Data/peptide_rt_training_set1.csv', mode='r') as csv_file:
     csv_reader = csv.reader(csv_file, delimiter=',')
     for row in csv_reader:
         if float(row[4]) <= evalue_cutoff:
              count=count+1 
              if option==1:
                 x = PeptideHydrophobicity(row[0]);
              else:
                 x = ElementalComposition(row[0]);
              training_X.append(x) ###peptide
              #training_y.append(60*float(row[2])) ###retention time minutes
              training_y.append(float(row[2])) ###retention time hours
print("Count training=",count)
#print("training_X=",training_X)
#print("taining_y=",training_y)

count=0;
###Reading testing data
with open('/content/drive/My Drive/Data/peptide_rt_testing_set1.csv', mode='r') as csv_file:
     csv_reader = csv.reader(csv_file, delimiter=',')
     for row in csv_reader:
        #print(float(row[4]))
        if float(row[4]) <= evalue_cutoff:
            count=count+1;
            if option==1:
               x = PeptideHydrophobicity(row[0]);
            else:
               x = ElementalComposition(row[0]); 
            testing_X.append(x)
            #testing_y.append(60*float(row[2])) ###retention time minutes
            testing_y.append(float(row[2])) ###retention time hours
print("Count testing=",count)
#print("testing_X=",testing_X)
#print("testing_y=",testing_y)

### define the Neural Network Model
model = Sequential()
model.add(Dense(21, input_dim=21, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(1, activation='linear'))

### compile the keras model with different optimizer
###adam optimizer
model.compile(loss="mean_squared_error", optimizer='Adam', metrics=['mse'])
###RMSprop optimizer
#model.compile(loss="mean_squared_error", optimizer='RMSprop', metrics=['mse'])
###Stochastic gradient descent(SGD)
#opt = SGD(lr=0.01, momentum=1)  
#model.compile(loss="mean_squared_error", optimizer= opt, metrics=['mse'])

### fit the keras model on the dataset
history=model.fit(training_X,training_y, validation_data=(testing_X, testing_y), epochs=20, batch_size=5)

### evaluate the keras model
# evaluate the model
_, train_mse = model.evaluate(training_X,training_y, verbose=0)
_, testing_mse = model.evaluate(testing_X,testing_y)
print('Train: %.3f, Test: %.3f' % (train_mse,testing_mse))

# plot loss during training
pyplot.title('Mean Squared Error versus Epoch')
pyplot.xlabel('Epoch') 
pyplot.ylabel('Mean Squared Error') 
pyplot.plot(history.history['loss'], label='Trainning Data')
pyplot.plot(history.history['val_loss'], label='Test Data')
pyplot.legend()
pyplot.savefig('/content/drive/My Drive/Data/MSE.png')
pyplot.show()

# get layer weights
#first_layer_weights = model.layers[0].get_weights()[0]
#print(first_layer_weights)

#predict values
pred = model.predict(testing_X)
#print("predict values=",pred)


pyplot.xlabel('Predicted RT') 
pyplot.ylabel('Experimental RT') 
pyplot.title('Experimental RT vesus Predicted RT')
#for i in range(len(pred)):
#	print("SEQ=%s, X=%s, Predicted=%s" % (testing_y[i], pred[i],testing_X[i]))

#Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(pred, testing_y)
print("R^2=",regr.score(pred,testing_y))
print("Coef=",regr.coef_) 
# Plot outputs
pyplot.scatter(pred,testing_y)
pyplot.plot(pred, regr.predict(pred), color='red',linewidth=3)
pyplot.savefig('/content/drive/My Drive/Data/RT.png')
pyplot.show()
