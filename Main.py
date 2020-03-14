import HW06 as hw


def printSSE(i):
    print("Sum of Squared Error is: "+str(i))

# iris = datasets.load_iris()
# data = pd.DataFrame(preprocessing.normalize(iris.data), columns=iris.feature_names)
# need to normalize all columns except target so we can figure out what it is at the end
# data = pd.DataFrame(iris.data, columns = iris.feature_names)
# data = preprocessing.normalize(data.iloc[:,:-1])
# data['target'] = pd.Series(iris.target)
# cols = list(data)
# cols.insert(0, cols.pop(cols.index('target')))
# data = data.loc[:, cols]
# dataset = data.values.tolist()


def question(centroids, dataset, similarity):
    hw.showDataset2D(dataset)
    clustering = hw.kmeans(dataset, 2, True, centroids, similarity=similarity)
    print("-------------------------------------------")
    printSSE(clustering["withinss"])
    hw.printTable(clustering["centroids"])
    print("-------------------------------------------")


# similarity: 0 = Euclidean, 1 = Cosine, 2 = Manhattan, 3 = Jaccard
# Question 1.1
data = hw.loadCSV('footballData.csv')
centroidPoints = [[0, 4, 6], [0, 5, 4]]
print("Manhattan")
question(centroidPoints, data, 2)

# Question 1.2
print("Euclidean")
question(centroidPoints, data, 0)

# Question 1.3
print("Manhattan")
centroidPoints = [[0, 3, 3], [0, 8, 3]]
question(centroidPoints, data, 2)

# Question 1.4
print("Manhattan")
centroidPoints = [[0, 3, 2], [0, 4, 8]]
question(centroidPoints, data, 2)
