"""
 Kfold validation
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from keras.utils import to_categorical

Language = input('Enter Chinese (C) or English (E)  ')
filename = Language + 'DB.csv'
dataset = np.loadtxt(filename, delimiter=',',dtype=float)
np.random.shuffle(dataset)
print('dataset shape= ',dataset.shape)
X = dataset[:,1:785]
y = dataset[:,0]

Nlayers = int(input('Number of Hidden layers (>=0) = '))
Nnodes = 0
if Nlayers > 0:
    Nnodes =  int(input('Number of nodes/layer= '))

def DNNfit(Xtrain, ytrain,Nlayers,Nnodes):  
    Nclasses = 10
    Nfeatures = Xtrain.shape[1]   # number of columns minus label
    Nepochs = 100
    model = Sequential()
    if Nlayers == 0:
        model.add(Dense(Nclasses, input_dim=Nfeatures, activation='softmax'))  # SLP 
    else:
        model.add(Dense(Nnodes, input_dim=Nfeatures, activation='relu'))
        for n in range(Nlayers-1):  # first hidden layer is previous statement
            model.add(Dense(Nnodes, activation='relu'))
        model.add(Dense(Nclasses, activation='softmax')) 
   
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])   
    ytrain_onehot = to_categorical(ytrain)
    history = model.fit(Xtrain, ytrain_onehot, 
                  #validation_split= .3, 
                  epochs = Nepochs, verbose=0)
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

 
# prepare the k-fold cross-validation configuration
n_folds = 10
kfold = KFold(n_folds, True, 1)
    # cross validation estimation of performance
scores = list()
fold = 1
for train_ix, test_ix in kfold.split(X):
    # select samples
    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]
    # evaluate model
    model, history = DNNfit(Xtrain,ytrain,Nlayers,Nnodes)



    Nerr, Ntotal, CM, PEG = DNNevaluate(model,Xtest,ytest)
    print('\nfold= %2d' % fold)
    print(Language + 'DB CM=\n', CM)
    print('Nerr= %d' % Nerr)
    print('Ntotal= %d' % Ntotal)
    print('DB PEG= %.4f' % PEG)
    scores.append(PEG)
    fold += 1
    #members.append(model)
# summarize expected performance
print('\nAverage PEG= %.4f   (SD= %.3f)' % (np.mean(scores), np.std(scores)))
print('\nMedian PEG= %.4f' % np.median(scores))
