# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.datasets import mnist

from keras.preprocessing.image import ImageDataGenerator
import os

import sklearn
from sklearn.utils import shuffle

from tensorflow import keras
from tensorflow.keras import layers

import copy

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# %% [markdown]
# ## New augmentation using S as a minibatch to create one new image

# %% [markdown]
# # Seperate digit data

# %%
p = 50

# %%
# the data, split between train and test sets
(X_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
x_test = np.expand_dims(x_test, -1)


# %%
print(X_train[y_train == 0].shape,
X_train[y_train == 1].shape,
X_train[y_train == 2].shape,
X_train[y_train == 3].shape,
X_train[y_train == 4].shape,
X_train[y_train == 5].shape,
X_train[y_train == 6].shape,
X_train[y_train == 7].shape,
X_train[y_train == 8].shape,
X_train[y_train == 9].shape)


# %%
X_train.shape


# %%
x0 = X_train[y_train == 0]
x1 = X_train[y_train == 1]
x2 = X_train[y_train == 2]
x3 = X_train[y_train == 3]
x4 = X_train[y_train == 4]
x5 = X_train[y_train == 5]
x6 = X_train[y_train == 6]
x7 = X_train[y_train == 7]
x8 = X_train[y_train == 8]
x9 = X_train[y_train == 9]


# %%
x0.shape

# %% [markdown]
# # Subset each class

# %%
np.random.seed(123)
x0_ind = np.random.choice(x0.shape[0], x0.shape[0]//p, False)
x1_ind = np.random.choice(x1.shape[0], x1.shape[0]//p, False)
x2_ind = np.random.choice(x2.shape[0], x2.shape[0]//p, False)
x3_ind = np.random.choice(x3.shape[0], x3.shape[0]//p, False)
x4_ind = np.random.choice(x4.shape[0], x4.shape[0]//p, False)
x5_ind = np.random.choice(x5.shape[0], x5.shape[0]//p, False)
x6_ind = np.random.choice(x6.shape[0], x6.shape[0]//p, False)
x7_ind = np.random.choice(x7.shape[0], x7.shape[0]//p, False)
x8_ind = np.random.choice(x8.shape[0], x8.shape[0]//p, False)
x9_ind = np.random.choice(x9.shape[0], x9.shape[0]//p, False)


# %%
x0_train = x0[x0_ind, :, :, :]
x1_train = x1[x1_ind, :, :, :]
x2_train = x2[x2_ind, :, :, :]
x3_train = x3[x3_ind, :, :, :]
x4_train = x4[x4_ind, :, :, :]
x5_train = x5[x5_ind, :, :, :]
x6_train = x6[x6_ind, :, :, :]
x7_train = x7[x7_ind, :, :, :]
x8_train = x8[x8_ind, :, :, :]
x9_train = x9[x9_ind, :, :, :]

# %% [markdown]
# # Combine each subclass

# %%
new_train = np.concatenate([x0_train, x1_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train, x9_train], axis = 0)
new_train.shape


# %%
np.array_equal(new_train[:x0_train.shape[0]], x0_train)


# %%
np.array_equal(new_train[x0_train.shape[0]:(x0_train.shape[0]+x1_train.shape[0])], x1_train)


# %%
new_y_train = np.array([0] * x0_train.shape[0] + 
[1] * x1_train.shape[0] + 
[2] * x2_train.shape[0] +
[3] * x3_train.shape[0] +
[4] * x4_train.shape[0] +
[5] * x5_train.shape[0] +
[6] * x6_train.shape[0] +
[7] * x7_train.shape[0] +
[8] * x8_train.shape[0] +
[9] * x9_train.shape[0])

# %% [markdown]
# # Shuffle new train data and labels

# %%
train, y = sklearn.utils.shuffle(new_train, new_y_train, random_state=0)


# %%
train.shape

# %% [markdown]
# # Train model with original data

# %%
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# %% [markdown]
# # Generate new data based on these partial data

# %%
def generator(dataset, num = 0, S = 50, p = 0.25, thre = 140):
    '''
    by tuning paramaters we generate one new digit image
    '''
    cla = dataset
    np.random.shuffle(cla) # shuffle the class
    sub_index = np.random.choice(cla.shape[0], S, False) # random subset S data
    basis = cla[sub_index, :, :, :] # form the basis

    new = np.zeros_like(basis[1])
    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            count = []
            for q in range(S):
                if basis[q, i, j, 0] >= thre:
                    count.append(basis[q, i, j, 0])
            if len(count) >= S * p:
                new[i, j, 0] = np.mean(count)
    #plt.imshow(new)
    return new


# %%
def create(dataset, mul = 3, num = 0, S = 50, p = 0.25, thre = 140):
    '''
    create a whole new data set
    '''
    x_new = copy.deepcopy(dataset)
    for i in range(x_new.shape[0] * mul):
        new = generator(x_new, num = num, S = S, p = p, thre = thre)
        x_new = np.append(x_new, np.expand_dims(new, axis=0) , axis = 0)
    return x_new

# %% [markdown]
# ## generate data

# %%
# 0
x0_new = create(x0_train, mul = 6, num = 0, S = 50, p = 0.25, thre = 140)
print(x0_new.shape)
# 1
x1_new = create(x1_train, mul = 6, num = 1, S = 50, p = 0.25, thre = 140)
print(x1_new.shape)
# 2
x2_new = create(x2_train, mul = 6, num = 2, S = 50, p = 0.3, thre = 175)
print(x2_new.shape)
# 3
x3_new = create(x3_train, mul = 6, num = 3, S = 50, p = 0.4, thre = 150)
print(x3_new.shape)
# 4
x4_new = create(x4_train, mul = 6, num = 4, S = 50, p = 0.3, thre = 17)
print(x4_new.shape)
# 5
x5_new = create(x5_train, mul = 6, num = 5, S = 50, p = 0.3, thre = 150)
print(x5_new.shape)
# 6
x6_new = create(x6_train, mul = 6, num = 6, S = 50, p = 0.45, thre = 100)
print(x6_new.shape)
# 7
x7_new = create(x7_train, mul = 6, num = 7, S = 50, p = 0.45, thre = 150)
print(x7_new.shape)
# 8
x8_new = create(x8_train, mul = 6, num = 8, S = 50, p = 0.4, thre = 120)
print(x8_new.shape)
# 9
x9_new = create(x9_train, mul = 6, num = 9, S = 50, p = 0.3, thre = 150)
print(x9_new.shape)


# %%
generated_train = np.concatenate([x0_new, x1_new, x2_new, x3_new, x4_new, x5_new, x6_new, x7_new, x8_new, x9_new], axis = 0)
print(generated_train.shape)

generated_y_train = np.array([0] * x0_new.shape[0] + 
[1] * x1_new.shape[0] + 
[2] * x2_new.shape[0] +
[3] * x3_new.shape[0] +
[4] * x4_new.shape[0] +
[5] * x5_new.shape[0] +
[6] * x6_new.shape[0] +
[7] * x7_new.shape[0] +
[8] * x8_new.shape[0] +
[9] * x9_new.shape[0])
print(generated_y_train.shape)

generated_train, generated_y = sklearn.utils.shuffle(generated_train, generated_y_train, random_state=0)

# %% [markdown]
# # define a function to recreate these process

# %%
def CNNfit(generated_train, generated_y):
    '''
    train the model
    '''
    # Scale images to the [0, 1] range
    generated_train = generated_train.astype("float32") / 255

    # convert class vectors to binary class matrices
    generated_y = keras.utils.to_categorical(generated_y, 10)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),        
        ]
        )

    batch_size = 128
    epochs = 7

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(generated_train, generated_y, batch_size=batch_size, epochs=epochs,              
    validation_split=0.1, verbose = 0)

    return model, history


# %%
def CNNevaluate(model, Xtest, ytest):  # computes CM and PE for test set
    Ntest = Xtest.shape[0] # number of rows
    CM = np.zeros([10,10], dtype = int)
    ypred = model.predict(Xtest, verbose = 1) # predicts entire set
    for i in range(Ntest):
        yclass = np.argmax(ypred[i])   
        ytrue = int(ytest[i])
        CM[ytrue,yclass] += 1

    Nerr = sum(sum(CM))-np.trace(CM)
    Ntotal = sum(sum(CM))
    PE = Nerr/Ntotal
    return Nerr, Ntotal,CM, PE


# %%
Nrestarts = 10
PEG_vals = np.zeros(Nrestarts)

# the data, split between train and test sets
(X_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_test = np.expand_dims(x_test, -1)

for restart in range(Nrestarts):

    model, history = CNNfit(generated_train, generated_y)
    

    Nerr, Ntotal,CM, PEG = CNNevaluate(model,x_test,y_test) 
    PEG_vals[restart] = PEG

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(PEG_vals)
ax.set_title('Data Augmentation')
ax.set_xlabel('PEG')
ax.set_ylabel('PE')
plt.show() 


# %%
Nrestarts = 10
PEG_vals = np.zeros(Nrestarts)

# the data, split between train and test sets
(X_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_test = np.expand_dims(x_test, -1)

for restart in range(Nrestarts):

    model, history = CNNfit(train, y)
    

    Nerr, Ntotal,CM, PEG = CNNevaluate(model,x_test,y_test) 
    PEG_vals[restart] = PEG

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(PEG_vals)
ax.set_title('Original Data')
ax.set_xlabel('PEG')
ax.set_ylabel('PE')
plt.show() 


