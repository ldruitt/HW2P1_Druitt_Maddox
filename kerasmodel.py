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
    print(ytrain[imageindex])
    plt.imshow(Xtrain[imageindex], cmap='Greys')
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

    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=Xtrain, y=ytrain, epochs=10)

    model.evaluate(Xtest, ytest)


if __name__ == '__main__':
    main()