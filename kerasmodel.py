import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    Xtest = np.load('MNIST_X_test_1.npy')
    ytest = np.load('MNIST_y_test_1.npy')
    Xtrain = np.load('MNISTXtrain1.npy')
    ytrain = np.load('MNISTytrain1.npy')
    imageindex  = 7777
    #print(ytrain[imageindex])
    #plt.imshow(Xtrain[imageindex], cmap='Greys')
    #print(Xtrain.shape)
    # Reshaping the array to 4-dims so that it can work with the Keras API
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 28, 28, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    Xtrain = Xtrain.astype('float32')
    Xtest = Xtest.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    Xtrain /= 255
    Xtest /= 255
    print('x_train shape:', Xtrain.shape)
    print('Number of images in x_train', Xtrain.shape[0])
    print('Number of images in x_test', Xtest.shape[0])

    # Model 1: Basic 2-hidden-layer, based on theoretical minimum layer needed to solve any problem. Ignores dimensions.
    model1 = Sequential()
    model1.add(Flatten())
    model1.add(Dense(128))
    model1.add(Dense(10))

    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model1.fit(x=Xtrain, y=ytrain, epochs=10)

    # Model 2: Intermediate model using 2D layers for better image comprehension. Like model 3, without activations.
    model2 = Sequential()
    model2.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Flatten())
    model2.add(Dense(128))
    model2.add(Dense(10))

    model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model2.fit(x=Xtrain, y=ytrain, epochs=10)

    # Model 3: Loren's model. Adds activation functions and a dropout layer to reduce potential overfitting.
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=Xtrain, y=ytrain, epochs=10)

    # Model 4: High complexity model with multiple 2D layer groups. Testing increased complexity.
    model4 = Sequential()
    model4.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model4.add(MaxPooling2D(pool_size=(2, 2)))
    model4.add(Conv2D(28, kernel_size=(3, 3)))
    model4.add(MaxPooling2D(pool_size=(2, 2)))
    model4.add(Flatten())
    model4.add(Dense(128, activation=tf.nn.relu))
    model4.add(Dropout(0.2))
    model4.add(Dense(10, activation=tf.nn.softmax))

    model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model4.fit(x=Xtrain, y=ytrain, epochs=10)

    # Evaluate the models.
    score1, acc1 = model1.evaluate(Xtest, ytest)
    score2, acc2 = model2.evaluate(Xtest, ytest)
    score3, acc3 = model.evaluate(Xtest, ytest)
    score4, acc4 = model4.evaluate(Xtest, ytest)

    print("Model 1 score/accuracy: " + str(score1) + ", " + str(acc1))
    print("Model 2 score/accuracy: " + str(score2) + ", " + str(acc2))
    print("Model 3 score/accuracy: " + str(score3) + ", " + str(acc3))
    print("Model 4 score/accuracy: " + str(score4) + ", " + str(acc4))

    model4.save('MyBestModel_00956434_01023420.h5')  # creates a HDF5 file

if __name__ == '__main__':
    main()