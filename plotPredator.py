import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

'''
plotPredator.py runs on the command line, taking in two arguments:
    fileName: name of the .csv file, with the data to plot
    dimension: the dimension of the plot '2' for neural network data and '3' for random forest data
The data in the two .csv files submitted aim to observe the trends of the neural network (2D plot) and the
random forest (3D plot), as their respective parameters are varied.

Note: the random forest's plot looks at % correct (where 1 is a perfect score)
and the neural network looks at the %error (where 0 represents a perfect score)

@author: Tammie Thong
@version: 26 Jan 2017
'''

csv_file_name = sys.argv[1]
dimension = sys.argv[2]

my_data = np.genfromtxt(csv_file_name, delimiter=',')  # read in .csv file
fig = plt.figure()

# displays a 3D plot of data generated by random forest
if dimension == '3':
    fig.suptitle('Random forests performance: 50% data used for test set and 50% used for training set',
                 fontsize=13, fontweight='bold')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Tree Size')
    ax.set_ylabel('Forest Size')
    ax.set_zlabel('% Correct')
    ax.scatter(my_data[:, 0], my_data[:, 1], zs=my_data[:, 2])
# displays 2D plot of data generated by neural network
if dimension == '2':
    fig.suptitle('Neural network performance as number of nodes in hidden layer increases',
                 fontsize=13, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel('# Hidden Nodes')
    ax.set_ylabel('Error')
    ax.scatter(my_data[:, 0], my_data[:, 1])

plt.show()