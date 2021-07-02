import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.datasets import mnist

print('\nShow_Spectrograms')
filename = 'EDB.csv'
DB = np.loadtxt(filename,delimiter=',',dtype=float)
X = DB[:,1:785]
y = DB[:,0].astype(int)
classes = ['0', '1', '2','3','4','5','6','7','8','9']
for i in range(10):
    Image = np.reshape(X[i,:],(28,28)) # reshape data row as 28 x 28 image
    imshow(Image)
    plt.title('English '+ classes[y[i]])
    plt.show()
    
filename = 'CDB.csv'
DB = np.loadtxt(filename,delimiter=',',dtype=float)
X = DB[:,1:785]
y = DB[:,0].astype(int)
classes = ['0', '1', '2','3','4','5','6','7','8','9']
for i in range(10):
    Image = np.reshape(X[i,:],(28,28)) # reshape data row as 28 x 28 image
    imshow(Image)
    plt.title('Chinese '+ classes[y[i]])
    plt.show()
    
