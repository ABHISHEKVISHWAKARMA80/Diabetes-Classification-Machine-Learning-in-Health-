# Diabetes-Classification-Machine-Learning-in-Health-
Predict whether a person is suffering from diabetes or not using K-Nearest Neighbors Algorithm.

# k-nearest neighbors algorithm

The k-nearest neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems. It is mostly used to classifies a data point based on how its neighbours are classified.

In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:
a) In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
b) In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.

- A small value of k means that noise will have a higher influence on the result and a large value make it computationally expensive.

- Data scientists usually choose as an odd number if the number of classes is 2 and another simple approach to select k is set k=sqrt(n).

- classifies the new data or case based on a similarity measure.

- ‘k’ in KNN is a parameter that refers to the number of nearest neighbours to include in the majority of the voting process.

- use euclidean distance. (For categorical variables, the hamming distance must be used.) 

# prerequisite
1. install jupyter notebook (or you can use any other python platform also)
2. install numpy, pandas & matplotlib on it.

# Description of repository:
1. Diabetes_XTrain.csv, Diabetes_YTrain.csv & Diabetes_Xtest.csv are given data set.
2. code.ipynb is code in python to implement K-Nearest Neighbors Algorithm.
3. output.csv is the final result.

# Pros of KNN
-Simple to implement

-Flexible to feature/distance choices

-Naturally handles multi-class cases

-Can do well in practice with enough representative data

# Cons of KNN
-Need to determine the value of parameter K (number of nearest neighbors)

-Computation cost is quite high because we need to compute the distance of each query instance to all training samples.

-Storage of data

-Must know we have a meaningful distance function.

K-NN is also a lazy learner because it doesn’t learn a discriminative function from the training data but “memorizes” the training dataset instead.
