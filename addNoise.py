#    name: addNoise.py
#  author: molloykp (Oct 2019)
# purpose: Accept a numpy npy file and add Gaussian noise
#          Parameters:

import numpy as np

def main():
    np.random.seed(1671)

    # make the script read the parameters (fill this out)
    # call the input matrix inMatrix

    # matrix must be floating point to add values
    # from the Gaussian
    inMatrix = inMatrix.astype('float32')
    inMatrix += np.random.normal(0,parms.sigma,(inMatrix.shape))
    inMatrix = inMatrix.astype('int')

    # noise may have caused values to go outside their allowable
    # range
    inMatrix[inMatrix < 0] = 0
    inMatrix[inMatrix > 255] = 255
    
    #save the perturbed matrix in a file

    

if __name__ == '__main__':
    main()