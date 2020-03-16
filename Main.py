import HW06 as hw
import pandas as pd
from sklearn import datasets, preprocessing
from statistics import mode, StatisticsError


def printSSE(i):
    print("Sum of Squared Error is: " + str(i))


def getAdjustedIrisData():
    iris = datasets.load_iris()
    irisData = pd.DataFrame(iris.data, columns=iris.feature_names)
    return adjustIrisData(irisData, iris)


def getNormalizedAdjustedIrisData():
    iris = datasets.load_iris()
    irisData = pd.DataFrame(preprocessing.normalize(iris.data, axis=0), columns=iris.feature_names)
    return adjustIrisData(irisData, iris)


def adjustIrisData(irisData, iris):
    irisData['target'] = pd.Series(iris.target)
    cols = list(irisData)
    cols.insert(0, cols.pop(cols.index('target')))
    adjustedData = irisData.loc[:, cols]
    return adjustedData.values.tolist()


def question(centroids, dataset, similarity, k):
    hw.showDataset2D(dataset)
    clustering = hw.kmeans(dataset, k, True, centroids, similarity=similarity)
    print("-------------------------------------------")
    printSSE(clustering["withinss"])
    hw.printTable(clustering["centroids"])
    print("-------------------------------------------")
    return clustering


def getAccuracy(clusters):
    mode0 = getModeCount(clusters[0])
    mode1 = getModeCount(clusters[1])
    mode2 = getModeCount(clusters[2])
    return (mode0 + mode1 + mode2) / 150


def getModeCount(cluster):
    labels = []
    for centroid in cluster:
        labels.append(centroid[0])
    listMode = -1
    try:
        listMode = mode(labels)
    except StatisticsError:
        print("Either list was empty or two modes exist. The accuracy will be wrong for this calculation")
    return labels.count(listMode)


# similarity: 0 = Euclidean, 1 = Cosine, 2 = Manhattan, 3 = Jaccard
# Question 1.1
data = hw.loadCSV('footballData.csv')
centroidPoints = [[0, 4, 6], [0, 5, 4]]
print("Manhattan")
question(centroidPoints, data, 2, 2)

# Question 1.2
print("Euclidean")
question(centroidPoints, data, 0, 2)

# Question 1.3
print("Manhattan")
centroidPoints = [[0, 3, 3], [0, 8, 3]]
question(centroidPoints, data, 2, 2)

# Question 1.4
print("Manhattan")
centroidPoints = [[0, 3, 2], [0, 4, 8]]
question(centroidPoints, data, 2, 2)

print("\n----------------Part 2-------------------\n")
print("\n----------------Question 2.1-------------\n")
print("Only Cosine Normalized")
print("Euclidean")
nonNormalEucCluster = question(None, getAdjustedIrisData(), 0, 3)

print("Cosine")
cosCluster = question(None, getNormalizedAdjustedIrisData(), 1, 3)
print("Jaccard")
nonNormalJacCluster = question(None, getAdjustedIrisData(), 3, 3)


print("All Normalized")
print("Euclidean")
normEucCluster = question(None, getNormalizedAdjustedIrisData(), 0, 3)
print("Jaccard")
normJacCluster = question(None, getNormalizedAdjustedIrisData(), 3, 3)

print("\n----------------Question 2.2-------------\n")
print("Non normalized")
print("Euclidean")
print(getAccuracy(nonNormalEucCluster["clusters"]))
print("Jaccard")
print(getAccuracy(nonNormalJacCluster["clusters"]))

print("\nNormalized")
print("Euclidean")
print(getAccuracy(normEucCluster["clusters"]))
print("Cosine")
print(getAccuracy(cosCluster["clusters"]))
print("Jaccard")
print(getAccuracy(normJacCluster["clusters"]))

"""
Questions 2.3 and 2.4 exist within HW06 through print statements.
 
For 2.3, a print statement on line 200 was made once each SSE reaches its minimum value. The print 
statement identifies how many iterations it took to reach that minimum value.

For 2.4, a few changes were made, but mainly the if statement on lines 193 and 194.
After running this script many times, there was never an instance where the SSE increase.
I also never ran more then 14 iterations.
"""
