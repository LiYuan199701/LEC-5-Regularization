print('\nDNN_Blobs_Overfitting_Demo')
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#from matplotlib import pyplot
from sklearn.datasets import make_blobs


SD = 15
Seed = 1490624

Nclasses = 3 #10
Nsamples = 100
Nfeatures = 10 #784


X,y = make_blobs(n_samples=Nsamples, centers = Nclasses,
                 n_features = Nfeatures, cluster_std = SD, random_state = Seed)

# split into train and test sets
rows = y.shape[0]
n_train = int(.5*rows)  # 50% of data into training set  25% into validation set
Xtrain, Xtest = X[:n_train, :], X[n_train:, :]
ytrain, ytest = y[:n_train], y[n_train:]
ytrain_onehot = to_categorical(ytrain)
ytest_onehot = to_categorical(ytest)



# define and compile DNN  
#Nlayers = 2 # int(input('Number of Hidden layers= '))
Nnodes = 200 # int(input('Number of nodes/layer= '))
model = Sequential()
model.add(Dense(Nnodes, input_dim=Nfeatures, activation='relu'))
model.add(Dense(Nnodes, activation='relu'))
model.add(Dense(Nnodes, activation='relu'))
model.add(Dense(Nclasses, activation='softmax'))    
model.compile(loss='categorical_crossentropy',
              optimizer='SGD', metrics=['accuracy'])    

 
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
    PE = Nerr/Ntotal
    return Nerr, Ntotal, CM, PE


NPE = 200
PET = np.zeros(NPE).astype(float)
PEV = np.zeros(NPE).astype(float)
for Num in range(NPE):
    model.fit(Xtrain, ytrain_onehot,
                    epochs = 1,  # training continues with additiaonal epoches            
                    verbose = 0)
    Nerr, Ntotal, CM, PET[Num] =  DNNevaluate(model,Xtrain,ytrain)    
    Nerr, Ntotal, CM, PEV[Num] =  DNNevaluate(model,Xtest,ytest)
 #   print('\nNum = %d   Nepochs = %3d PET= %.3f  PEV= %.3f' % (Num,Num*10+10,PET[Num],PEV[Num]))

plt.subplot(111)
plt.title('PET and PEV variation with epochs')
plt.plot(PET, label='PET')
plt.plot(PEV, label='PEV')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('PE')
plt.grid()
plt.show()
