from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics, svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def main():
    Xtest = np.load('MNIST_X_test_1.npy')
    ytest = np.load('MNIST_y_test_1.npy')
    Xtrain = np.load('MNISTXtrain1.npy')
    ytrain = np.load('MNISTytrain1.npy')

    Xtrain = Xtrain[:1000]
    Xtest = Xtest[:1000]
    ytrain = ytrain[:1000]
    ytest = ytest[:1000]

    Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
    Xtest = Xtest.reshape(Xtest.shape[0], -1)



#    steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='sigmoid'))]
#    pipeline = Pipeline(steps)
#    parameters = {'SVM__C': [0.0001, 0.001, .01, .1],'SVM__gamma': [1000, 100, 10, 1]}
#    grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

#    grid.fit(Xtrain, ytrain)
#    print('score = ' + str(grid.score(Xtest, ytest)))
#    print('best params = ' + str(grid.best_params_))

    clf1 = svm.SVC(C=.01, kernel='poly',gamma=1000)
    clf2 = svm.SVC(C=.0001, kernel='linear',gamma=1000)
    clf3 = svm.SVC(C=.01, kernel='sigmoid',gamma=1000)

    clf1.fit(Xtrain,ytrain)
    clf2.fit(Xtrain,ytrain)
    clf3.fit(Xtrain,ytrain)

    scores1 = cross_val_score(clf1, Xtrain, ytrain, cv=5)
    scores2 = cross_val_score(clf2, Xtrain, ytrain, cv=5)
    scores3 = cross_val_score(clf3, Xtrain, ytrain, cv=5)

    print('Accuracy of classifier 1: %0.2f' % (scores1.mean()))
    print('Accuracy of classifier 2: %0.2f' % (scores2.mean()))
    print('Accuracy of classifier 3: %0.2f' % (scores3.mean()))

    joblib.dump(clf2, 'MyBestModel_00956434_.pkl')


if __name__ == '__main__':
    main()