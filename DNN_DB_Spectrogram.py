
# Cross Validation Classification Confusion Matrix
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt 

print('\nDNN_CDB_Spectrogram')
Language = input('Enter Chinese (C) or English (E)  ')
filename = Language + 'DB.csv'
dataset = np.loadtxt(filename, delimiter=',',dtype=float)
np.random.shuffle(dataset)
print('dataset shape= ',dataset.shape)
X = dataset[:,1:785]
y = dataset[:,0]

# split into train and test sets
rows = y.shape[0]
n_train = int(.5*rows)  # 50% of data into training set  25% into validation set
Xtrain, Xtest = X[:n_train, :], X[n_train:, :]
ytrain, ytest = y[:n_train], y[n_train:]
 

def DNNfit(Nfeatures, Xtrain, ytrain):  
    
# Nepochs times Nsamples should be about 1E6  (million!)
    Nlayers = int(input('Number of Hidden layers= '))
    Nnodes = int(input('Number of nodes/layer= '))
      
    Nepochs = 100
    
    model = Sequential()

# first hidden layer
    model.add(Dense(Nnodes, input_dim=Nfeatures, activation='relu'))

# additional hidden layers    
    for n in range(Nlayers-1):  # first hidden layer is previous statement
        model.add(Dense(Nnodes, activation='relu'))

# output layer
    model.add(Dense(10, activation='softmax'))     
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
     
    ytrain_onehot = to_categorical(ytrain)
    history = model.fit(Xtrain, ytrain_onehot, 
                  validation_split= .3, 
                  epochs = Nepochs, verbose=1)
    return model, history
 
def DNNevaluate(model,Xtest,ytest):  # computes CM and PE for test set
    Ntest = Xtest.shape[0] # number of rows
    CM = np.zeros([10,10], dtype = int)
    ypred = model.predict(Xtest) # predicts entire set
    for i in range(Ntest):
        yclass = np.argmax(ypred[i])   
        ytrue = int(ytest[i])
        CM[ytrue,yclass] += 1

    Nerr = sum(sum(CM))-np.trace(CM)
    Ntotal = sum(sum(CM))
    PE = Nerr/Ntotal
    return Nerr, Ntotal,CM, PE


Nfeatures = 784
model, history = DNNfit(Nfeatures, Xtrain, ytrain)   
# plot loss learning curves
plt.subplot(211)
plt.title(filename + ' Cross-Entropy Loss', pad=-40)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
# plot accuracy learning curves
plt.subplot(212)
plt.title(filename + 'Accuracy', pad=-40)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()



Nerr, Ntotal, CM, PE = DNNevaluate(model,Xtest,ytest)
print('\n' + filename + ' CM=\n', CM)
print('\n' + filename + ' Nerr= %d' % Nerr)
print('\n' + filename + ' Ntotal= %d' % Ntotal)
print('\n' + filename + ' PEG= %.4f' % PE)
