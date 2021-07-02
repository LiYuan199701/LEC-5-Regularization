import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.datasets import mnist

print('\nShow_MNIST')
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
(X, y), (test_images, test_labels) = mnist.load_data()
classes = ['0', '1', '2','3','4','5','6','7','8','9']
for i in range(20):
    Image = np.reshape(X[i,:],(28,28)) # reshape data row as 28 x 28 image
    imshow(Image)
    plt.title(classes[y[i]])
    plt.show()
    
