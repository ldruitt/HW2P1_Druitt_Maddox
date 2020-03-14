import numpy as np
import sys
import tensorflow as tf

def main():
    X = np.load(sys.argv[1])
    # Reshaping the array to 4-dims so that it can work with the Keras API
    X = X.reshape(X.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    X = X.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    X /= 255
    model = tf.keras.models.load_model('MyBestModel_00956434_01023420.h5', custom_objects={'softmax_v2': tf.nn.softmax})
    y = model.predict_classes(X)
    # checking the accuracy compared to ytrain: 0.9980333333333333
    # ytrain = np.load('MNISTytrain1.npy')
    # print(np.sum(y == ytrain)/y.size)
    outputString = "\n".join(map(str,y))
    outputFile = open('output.txt', 'w')
    outputFile.write(outputString)
    outputFile.close()



if __name__ == '__main__':
    main()