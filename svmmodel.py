from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def main():
    Xtest = np.load('MNIST_X_test_1.npy')
    ytest = np.load('MNIST_y_test_1.npy')
    Xtrain = np.load('MNISTXtrain1.npy')
    ytrain = np.load('MNISTytrain1.npy')

    Xtrain = Xtrain[:100]
    Xtest = Xtest[:100]
    ytrain = ytrain[:100]
    ytest = ytest[:100]

    Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
    Xtest = Xtest.reshape(Xtest.shape[0], -1)

    steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
    pipeline = Pipeline(steps)
    parameters = {'SVM__C': [0.001, 0.1, 100, 10e5], 'SVM__gamma': [10, 1, 0.1, 0.01]}
    grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    grid.fit(Xtrain, ytrain)
    print('score = ' + str(grid.score(Xtest, ytest)))
    print('best params = ' + str(grid.best_params_))

if __name__ == '__main__':
    main()