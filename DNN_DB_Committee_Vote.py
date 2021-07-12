
# Cross Validation Classification Confusion Matrix
import numpy as np
from keras.models import load_model

print('\nDNN_DB')
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

def DNNevaluate(model,Xtest):  # computes CM and PE for test set
    ypred = model.predict(Xtest) # predicts entire set
    return ypred

Nmodels = 3

filename = 'DBmodel'+ Language + '0'
model0 = load_model(filename)
filename = 'DBmodel'+ Language + '1'
model1 = load_model(filename)
filename = 'DBmodel'+ Language + '2'
model2 = load_model(filename)

y0 = DNNevaluate(model0,Xtest)
y1 = DNNevaluate(model1,Xtest) 
y2 = DNNevaluate(model2,Xtest) 

Nerr = 0
for i in range(len(ytest)):
    v0 = np.argmax(y0[i])
    v1 = np.argmax(y1[i])
    v2 = np.argmax(y2[i])
    vote = np.median([v0,v1,v2])
    if int(vote) != int(ytest[i]):
        Nerr += 1

PEG = Nerr/len(ytest)
print('Committee of %d votes gives PEG= %.4f' % (Nmodels,PEG))  