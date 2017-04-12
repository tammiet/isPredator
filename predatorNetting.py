from __future__ import division
import numpy as np

'''
predatorNetting.py records the performance of a neural network in its ability
to predict the correct predator value.

It takes a sample of animals and their attributes from zoo.csv and
divides the set into a training set and a test set. The program returns the proportion
of the errors (0 means no errors)

I learned how to make a neural net from
http://iamtrask.github.io/2015/07/12/basic-python-network/

@author: Tammie Thong
@version: 26 Jan 2017
'''

my_data = np.genfromtxt('zoo.csv', delimiter=',') # read in file
clean_data = my_data[:, 1:] # remove the animal's name from the dataset

prop = input("Please enter the proportion of entries to use in training set: \n")
reps = input ("Please enter the number of repetitions: \n")
l1_size = input("Please enter the number of nodes in hidden layer: \n")

# sigmoid function
def nonlin(x,deriv=False):
    """
    Maps to a value between 0 and 1. Converts numbers to probabilities

    :param float, x
    :param: bool, deriv
    :return: float

    """
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def train(data, l1_size, predictionColumn=6):
    """
    Trains neural network on a data set, using backpropagation

    :param: list, data
    :param: int, l1_size
    :param: int, predictionColumn
    :return: tuple, (syn0, syn1)

    """
    # delete predator column
    X = np.delete(data, predictionColumn, axis=1)

    # output dataset
    y = np.array([data[:, predictionColumn]]).T

    # seed random numbers to make calculation
    # deterministic (just a good practice)
    np.random.seed(1)

    # initialize random weights
    syn0 = 2*np.random.random((len(X[0, :]), l1_size)) - 1
    syn1 = 2*np.random.random((l1_size, 1)) - 1

    for j in xrange(60000):

        # Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))

        # how much did we miss the target value?
        l2_error = y - l2

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)

        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    return syn0, syn1

def test(syn0, syn1, test_data, predictionColumn=6):
    """
    Tests neural network against data where syn0 and syn1 are the matrices of the first and
    second of weights in the neural network

    :param: matrix, syn0
    :param: matrix, syn1
    :param list, test_data:
    :param predictionColumn:
    :return:
    """

    # assign first layer, second layer, and output column
    l0 = np.delete(test_data, predictionColumn, axis=1)
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    y = np.array([test_data[:, predictionColumn]]).T

    #determines number of instances neural network fails
    l2_error = y - l2
    n_correct = 0

    #rounds error values
    for iter in l2_error:
        if (iter <= 0.1):
            n_correct+=1

    return n_correct/len(l2_error)

def validation(data, proportion, reps, l1_size, predictionColumn=6):
    """
    Returns the performance of a neural network trained on a training set,
    based on its ability to determine the predator value on a test set

    Constructs a test set and a training set from zoo.csv.
    Data is split into a test set and training set, where the training set is used to
    train the neural network.
    It uses the test method to compare how well the neural network performs.


    :param: matrix: data
    :param: float,  proportion
    :param: int, reps
    :param: int, l1_size
    :param: int predictionColumn
    :return: average, double
    """
    split = int(len(data)*proportion)
    np.random.shuffle(data)
    trainingSet = data[0:split]
    testSet = data[split:]
    record = np.empty(reps)
    for iter in range(reps):
        syn0, syn1 = train(trainingSet, l1_size) # gets weights
        record[iter] = test(syn0, syn1, testSet)
    return np.average(record)


print "There are", validation(clean_data, prop, reps, l1_size), "%  errors."

#  code for running program multiple times and varying number of hidden nodes. This code can be uncommented to
#  write data to a .csv file, which can be used to make a 2D plot using plotPredator.py
'''
max_hidden_nodes = 20
scores = np.empty([max_hidden_nodes, 2])
for i, n in enumerate(range(0, max_hidden_nodes)):
    scores[i] = np.array([n, validation(clean_data, 0.5, 10, n+1)])

print scores

np.savetxt("20repsNetting.csv", scores, delimiter=",") # writes file to .csv file
'''
