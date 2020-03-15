import random
import time
from tkinter import *
import Kmeans_Jaccard as jac
import Kmeans_Cosine as cos
import Kmeans_Euclidean as euc


######################################################################
# This section contains functions for loading CSV (comma separated values)
# files and convert them to a dataset of instances.
# Each instance is a tuple of attributes. The entire dataset is a list
# of tuples.
######################################################################
# Loads a CSV files into a list of tuples.
# Ignores the first row of the file (header).
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
#   fileName: name of the CSV file to be read
# Returns: a list of tuples
def loadCSV(fileName):
    fileHandler = open(fileName, "r")
    lines = fileHandler.readlines()
    fileHandler.close()
    del lines[0]  # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    return dataset


# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    cleanLine = line.strip()
    # get rid of quotes
    cleanLine = cleanLine.replace('"', '')
    # separate the fields
    lineList = cleanLine.split(",")
    # convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple


# Destructively converts all the string elements representing numbers
# to floating point numbers.
# Parameters:
#   myList: a list of strings
# Returns None
def stringsToNumbers(myList):
    for i in range(len(myList)):
        if isValidNumberString(myList[i]):
            myList[i] = float(myList[i])


# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
    if len(s) == 0:
        return False
    if len(s) > 1 and s[0] == "-":
        s = s[1:]
    for c in s:
        if c not in "0123456789.":
            return False
    return True


######################################################################
# This section contains functions for clustering a dataset
# using the k-means algorithm.
######################################################################
# manhattan
def manDistance(instance1, instance2):
    if instance1 is None or instance2 is None:
        return float("inf")
    sum = 0
    for i in range(1, len(instance1)):
        sum += abs(instance1[i] - instance2[i])
    return sum


def distance(instance1, instance2, similarity):
    """
    Calculates the distance using the specified distance.

    :param similarity: 0 = Euclidean, 1 = Cosine, 2 = Manhattan, 3 = Jaccard
    """
    if similarity == 0:
        return euc.distance(instance1, instance2)
    elif similarity == 1:
        return cos.cdistance(instance1, instance2)
    elif similarity == 2:
        return manDistance(instance1, instance2)
    elif similarity == 3:
        return jac.jdistance(instance1, instance2)
    else:
        print("ERROR")
        return


def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if numInstances == 0:
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes - 1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)


def assign(instance, centroids, similarity):
    minDistance = distance(instance, centroids[0], similarity)
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i], similarity)
        if d < minDistance:
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex


def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList


def assignAll(instances, centroids, similarity):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids, similarity)
        clusters[clusterIndex].append(instance)
    return clusters


def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        # name = "centroid" + str(i)
        name = i
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids


def kmeans(instances, k, animation=False, initCentroids=None, similarity=0):
    """
    Calculates the kmeans cluster using the values specified belowed.
    :param similarity: 0 = Euclidean, 1 = Cosine, 2 = Manhattan, 3 = Jaccard
    :type similarity: int
    """
    result = {}
    if initCentroids is None or len(initCentroids) < k:
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
        # print(centroids)
    else:
        centroids = initCentroids
    prevCentroids = []
    if animation:
        delay = 1.0  # seconds
        canvas = prepareWindow(instances)
        clusters = createEmptyListOfLists(k)
        clusters[0] = instances
        paintClusters2D(canvas, clusters, centroids, "Initial centroids")
        time.sleep(delay)
    iteration = 0
    while centroids != prevCentroids:
        iteration += 1
        clusters = assignAll(instances, centroids, similarity)
        if animation:
            paintClusters2D(canvas, clusters, centroids, "Assign %d" % iteration)
            time.sleep(delay)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids, similarity)
        if animation:
            paintClusters2D(canvas, clusters, centroids,
                            "Update %d, withinss %.1f" % (iteration, withinss))
            time.sleep(delay)
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    return result


def computeWithinss(clusters, centroids, similarity):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += distance(centroid, instance, similarity) ** 2
    return result


######################################################################
# This section contains functions for visualizing datasets and
# clustered datasets.
######################################################################
def printTable(instances):
    for instance in instances:
        if instance is not None:
            line = str(instance[0]) + "\t"
            for i in range(1, len(instance)):
                line += "%.2f " % instance[i]
            print(line)


def extractAttribute(instances, index):
    result = []
    for instance in instances:
        result.append(instance[index])
    return result


def paintCircle(canvas, xc, yc, r, color):
    canvas.create_oval(xc - r, yc - r, xc + r, yc + r, outline=color)


def paintSquare(canvas, xc, yc, r, color):
    canvas.create_rectangle(xc - r, yc - r, xc + r, yc + r, fill=color)


def drawPoints(canvas, instances, color, shape):
    random.seed(0)
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2 * margin) / (maxX - minX)
    scaleY = float(height - 2 * margin) / (maxY - minY)
    for instance in instances:
        x = 5 * (random.random() - 0.5) + margin + (instance[1] - minX) * scaleX
        y = 5 * (random.random() - 0.5) + height - margin - (instance[2] - minY) * scaleY
        if shape == "square":
            paintSquare(canvas, x, y, 5, color)
        else:
            paintCircle(canvas, x, y, 5, color)
    canvas.update()


def connectPoints(canvas, instances1, instances2, color):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2 * margin) / (maxX - minX)
    scaleY = float(height - 2 * margin) / (maxY - minY)
    for p1 in instances1:
        for p2 in instances2:
            x1 = margin + (p1[1] - minX) * scaleX
            y1 = height - margin - (p1[2] - minY) * scaleY
            x2 = margin + (p2[1] - minX) * scaleX
            y2 = height - margin - (p2[2] - minY) * scaleY
            canvas.create_line(x1, y1, x2, y2, fill=color)
    canvas.update()


def mergeClusters(clusters):
    result = []
    for cluster in clusters:
        result.extend(cluster)
    return result


def prepareWindow(instances):
    width = 500
    height = 500
    margin = 50
    root = Tk()
    canvas = Canvas(root, width=width, height=height, background="white")
    canvas.pack()
    canvas.data = {}
    canvas.data["margin"] = margin
    setBounds2D(canvas, instances)
    paintAxes(canvas)
    canvas.update()
    return canvas


def setBounds2D(canvas, instances):
    attributeX = extractAttribute(instances, 1)
    attributeY = extractAttribute(instances, 2)
    canvas.data["minX"] = min(attributeX)
    canvas.data["minY"] = min(attributeY)
    canvas.data["maxX"] = max(attributeX)
    canvas.data["maxY"] = max(attributeY)


def paintAxes(canvas):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    canvas.create_line(margin / 2, height - margin / 2, width - 5, height - margin / 2,
                       width=2, arrow=LAST)
    canvas.create_text(margin, height - margin / 4,
                       text=str(minX), font="Sans 11")
    canvas.create_text(width - margin, height - margin / 4,
                       text=str(maxX), font="Sans 11")
    canvas.create_line(margin / 2, height - margin / 2, margin / 2, 5,
                       width=2, arrow=LAST)
    canvas.create_text(margin / 4, height - margin,
                       text=str(minY), font="Sans 11", anchor=W)
    canvas.create_text(margin / 4, margin,
                       text=str(maxY), font="Sans 11", anchor=W)
    canvas.update()


def showDataset2D(instances):
    canvas = prepareWindow(instances)
    paintDataset2D(canvas, instances)


def paintDataset2D(canvas, instances):
    canvas.delete(ALL)
    paintAxes(canvas)
    drawPoints(canvas, instances, "blue", "circle")
    canvas.update()


def showClusters2D(clusteringDictionary):
    clusters = clusteringDictionary["clusters"]
    centroids = clusteringDictionary["centroids"]
    withinss = clusteringDictionary["withinss"]
    canvas = prepareWindow(mergeClusters(clusters))
    paintClusters2D(canvas, clusters, centroids,
                    "Withinss: %.1f" % withinss)


def paintClusters2D(canvas, clusters, centroids, title=""):
    canvas.delete(ALL)
    paintAxes(canvas)
    colors = ["blue", "red", "green", "brown", "purple", "orange"]
    for clusterIndex in range(len(clusters)):
        color = colors[clusterIndex % len(colors)]
        instances = clusters[clusterIndex]
        centroid = centroids[clusterIndex]
        drawPoints(canvas, instances, color, "circle")
        if centroid is not None:
            drawPoints(canvas, [centroid], color, "square")
        connectPoints(canvas, [centroid], instances, color)
    width = canvas.winfo_reqwidth()
    canvas.create_text(width / 2, 20, text=title, font="Sans 14")
    canvas.update()
