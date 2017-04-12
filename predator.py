from __future__ import division
from learning import DataSet, DecisionTreeLearner, test
import random
import numpy as np

'''
predator.py creates multiple random forests and finds the average of all the forests.

It gets a data set from zoo.csv, containing a list of animals and their attributes.
The zoo data is divided into a test set and a training set.
It uses the aima.learning module to construct randomly generated decision trees for the training set.
The training set is used to guess the 'predator' value of each entry in the test set.
The mode of all the predicted values by the training set, for each animal in the test set, is stored
as the forest's prediction.

The forest's prediction is compared to the actual training set to determine its accuracy. Based on
the number of repetitions the program returns the average accuracy of all the forests. This is a number
between 1 and 0 where 1: all correct, 0: none correct

The four parameters are:
    forest_size: number of forests
    tree_size: number of animals in the tree
    test_data_size: percent of data used in training set (the rest is used in test set)
    reps: number of times a forest is generated

@author: Tammie Thong
@version: 26 Jan 2017

'''

forest_size = input("Please enter the number of trees: \n")
tree_size = input("Please enter the size of each tree: \n")
test_data_size = input("Please input a proportion used for training set between 0 and 1 (ex: .90): \n")
reps = input("Please enter number of repetitions: \n")

def test_forest_structure(numTrees, treeSize, testDataSize, reps=1000, verbose=False, testAttr= 'predator'):
    """
    This method has sub-methods which constructs a random forest and returns the average accuracy of
    all the forests.

    :param: int, numTrees
    :param: int, treeSize
    :param: double, testDataSize
    :param: int, reps
    :param: bool, verbose
    :param string, testAttr:
    :return: double

    """
    attribute_names = 'animal_name hair feathers eggs milk airborne aquatic predator toothed backbone breathes venomous fins legs tail domestic catsize type'
    animals = DataSet(name='zoo', target=testAttr, attrnames=attribute_names)
    testAttrIndex = animals.getTarget(testAttr) #gets index of test attribute
    animalList = animals.examples[:] #make copy of animals.examples to manipulate
    random.shuffle(animalList) #shuffle dataset
    testListStartIndex = int(len(animalList)*testDataSize) #gets index to divide dataset into test and training set
    trainingSet = animalList[0:testListStartIndex] #select first x entries, based on proportion used for validation
    testSet = animalList[testListStartIndex:] #select remaining entries to use in the test dataset
    treeList = [] #Array of trees in forest

    def make_group():
        """
        constructs a list of animals to make a decision tree

        :return: list, tree

        """
        random.shuffle(trainingSet)
        tree = trainingSet[0:treeSize]
        return tree

    est_dict = {}  # dictionary of animals their estimated predator values

    def entryScore(treeList):
        """
        Populates est_dict{}

        The mode of the estimated value of predator is taken from each
        decision tree generated, for each animal in the test set.

        :param: int, treeList

        """
        for i in range(0, numTrees):  # adds each tree to the list of trees in a forest
            treeList.append(make_group())
        treeScores = []  # array of estimated scores on test set for a decision tree
        forestOfTrees = []  # list of decision trees
        for a in range(0, len(treeList)):   # make decision trees and append to forestOfTrees
            animals.examples = treeList[a]  # rewrite example dataset to match each tree's dataset
            animal_tree = DecisionTreeLearner(animals)
            forestOfTrees.append(animal_tree)
        for i in range(0,len(testSet)):  # for each entry in the test set, determine the what the predicted answer is
            for x in range(0, len(forestOfTrees)):
                score = test(forestOfTrees[x], animals, examples=[testSet[i]], verbose=0)
                treeScores.append(score)
                if verbose:
                    print("Score: ", score)
            answer = mode(treeScores) # find the mode of of treeScores, to find the forest's result
            est_dict[testSet[i][0]]= answer
            if verbose:
                print "Mode: ", answer
                print "testSetName", testSet[i][0]
            del treeScores[:]  # delete items in treeScores for next tree entry

    def mode(aList):
        """
        finds the mode of a list

        :param: list, aList
        :return: int, key

        """
        dict = {}
        for key in aList:
            if key in dict:
                    dict[key] += 1
            else:
                dict[key] = 1
        return max(dict, key=dict.get)  # return the key with the highest value

    def TestForest(reps):
        """
        Compares the list of predator values for the actual test set against the forest's
        estimated predator values

        :param: int, reps
        :return: float, (avgScore/reps)

        """
        avgScore=0  # cumulative correct scores of all forest's predictions
        for i in range(0, reps):
            entryScore(treeList)  # popluate est_dict{}
            testSetValues = {}  # dictionary of values from the test set to be compared to forest's predicted answers
            for i in range(0, len(testSet)):  # enters name of animal as key, and the predator value, to testSetValues{}
                testSetValues[testSet[i][0]] = testSet[i][testAttrIndex]
            if verbose:  # prints answers from forest and actual answers from test set
                print "\n testSet: "
                for k, v in testSetValues.items():
                        print("Key : {0}, Value : {1}".format(k, v))

                print "\n est_dict: "
                for k, v in est_dict.items():
                        print("Key : {0}, Value : {1}".format(k, v))
            accuracy = 0
            for key, value in testSetValues.viewitems() & est_dict.viewitems():
                if testSetValues[key] == est_dict[key]:  # compares predator values for each set
                    accuracy += 1
            avgScore += (accuracy/len(est_dict)) # calcuates
            if verbose:
                print "accuracy: ", (accuracy/len(est_dict))
        return (avgScore/reps)*100

    return TestForest(reps)


print "Percent accuracy: "
print test_forest_structure(forest_size, tree_size, test_data_size, reps)  # percent accuracy of all forests


#  code for running program multiple times and varying tree size and forest size. This writes data to a .csv file, which can be used to make a 3D plot in plotPredtor.py
'''
treeSizes=np.arange(1, 20, 3)
forestSize=np.arange(1, 20, 3)
performanceMatrix=np.empty([len(treeSizes)*len(forestSize), 3])
for tS_i, tS in enumerate(treeSizes):
    for fS_i, fS in enumerate(forestSize):
        performanceMatrix[(tS_i*len(treeSizes))+fS_i] = np.array([tS, fS, test_forest_structure(tS, fS, .5, reps=100)])
np.savetxt("100reps.csv", performanceMatrix, delimiter=",")  # write performanceMatrix to .csv file

print performanceMatrix
'''


