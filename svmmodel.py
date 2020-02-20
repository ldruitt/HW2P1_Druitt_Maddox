from sklearn import svm
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def main():
    Xtest = np.load('MNIST_X_test_1.npy')
    ytest = np.load('MNIST_y_test_1.npy')
    Xtrain = np.load('MNISTXtrain1.npy')
    ytrain = np.load('MNISTytrain1.npy')

    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(Xtrain, ytrain)

    # Predict the response for test dataset
    ypred = clf.predict(Xtest)
    print("Accuracy:", metrics.accuracy_score(ytest, ypred))

if __name__ == '__main__':
    main()