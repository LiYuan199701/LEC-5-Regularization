
# Max norm weight regularization Text 13 p. 252

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# import activity regularizer
from keras.regularizers import l2  # l1: sum of absolute values  l2: sum of squares    
from keras.utils import to_categorical
import matplotlib.pyplot as plt 
from matplotlib import pyplot

 

print('\nDNN_CDB_Weight_SizeRegularization')
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
 
def DNNfit(Xtrain, ytrain):  
    
    Nfeatures = Xtrain.shape[1]   # number of columns minus label
    Nlayers = 2 # int(input('Number of Hidden layers= '))
    Nnodes = 50 # int(input('Number of nodes/layer= '))
    Nepochs = 100
    
    model = Sequential()
# first hidden layer
    model.add(Dense(Nnodes,  input_dim=Nfeatures, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))  # output layer
# additional hidden layers    
    for n in range(Nlayers-1):  # first hidden layer is previous statement
        model.add(Dense(Nnodes, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))  # output layer
    model.add(Dense(10, activation='softmax'))     
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])    
    ytrain_onehot = to_categorical(ytrain)
    history = model.fit(Xtrain, ytrain_onehot, 
                  validation_split= .3, 
                  epochs = Nepochs, verbose=0)
    return model, history
 
def DNNevaluate(model,Xtest,ytest):  # computes CM and PE for test set
    Ntest = Xtest.shape[0] # number of rows
    CM = np.zeros([10,10]).astype(int)
    ypred = model.predict(Xtest) # predicts entire set
    for i in range(Ntest):
        yclass = np.argmax(ypred[i])   # index of max softmax 
        ytrue = int(ytest[i])
        CM[ytrue,yclass] += 1

    Nerr = sum(sum(CM))-np.trace(CM)
    Ntotal = sum(sum(CM))
    PEG = Nerr/Ntotal
    return Nerr, Ntotal, CM, PEG

 
Nrestarts = 10
PEG_vals = np.zeros(Nrestarts)

for restart in range(Nrestarts):

    model, history = DNNfit(Xtrain, ytrain)
     # plot loss learning curves
    # plt.subplot(211)
    # plt.title(Language +' DNN Cross-Entropy Loss')
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='val')
    # plt.legend()
    # # plot accuracy learning curves
    # plt.subplot(212)
    # plt.title(Language + ' DNN Accuracy')
    # plt.plot(history.history['accuracy'], label='train')
    # plt.plot(history.history['val_accuracy'], label='val')
    # plt.legend()
    # plt.show()
    
    # evaluate PEG
    Nerr, Ntotal,CM, PEG = DNNevaluate(model,Xtest,ytest) 
    PEG_vals[restart] = PEG
    

    Nerr, Ntotal, CM, PE = DNNevaluate(model,Xtest,ytest)
    # print('\n' + Language + 'DB CM=\n', CM)
    # print('\n' + Language + 'DB Nerr= %d' % Nerr)
    # print('\n' + Language + 'DB Ntotal= %d' % Ntotal)
    print('\n' + Language + 'DB Restart=  %2d PEG= %.4f' % (restart,PE))
    
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(PEG_vals)
ax.set_title(Language + ' - Weight Size Regularization')
ax.set_xlabel('PEG')
ax.set_ylabel('PE')
plt.show()    
