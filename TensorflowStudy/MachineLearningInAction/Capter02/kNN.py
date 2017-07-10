from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

from os import listdir


def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = tile(inX, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distances = sq_distance ** 0.5
    sorted_dist_indices = distances.argsort()
    class_count = {}
    for i in range(k):
        voteIlabale = labels[sorted_dist_indices[i]]
        class_count[voteIlabale] = class_count.get(voteIlabale, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet /= tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large does']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier mils earned per years?"))
    iceCream = float(raw_input("liters of ice cream consumed per years?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])

    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    hwDataSet = zeros((m, 1024))
    for i in range(m):
        filename = trainingFileList[i]
        label = int(filename.split('_')[0])
        hwLabels.append(label)
        hwDataSet[i, :] = img2vector('trainingDigits/%s' % filename)

    accuracy = 0.
    testFileList = listdir('testDigits')
    mTest = len(testFileList)
    for testFile in testFileList:
        testLabel = int(testFile.split('_')[0])
        testVector = img2vector('testDigits/%s' % testFile)
        pred = classify(testVector, hwDataSet, hwLabels, 3)
        print("Pred:", pred, "testLabel", testLabel)
        if pred == testLabel:
            accuracy += 1.

    accuracy /= mTest
    print("Accuracy:", accuracy)


# demo 1
# group, labels = create_dataset()
# print(classify([3, 0], group, labels, 3))

# demo 2
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels)

# demo 3
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()

# demo 4
# normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print(minVals)

# demo 5
# datingClassTest()

# demo 6
# classifyPerson()

# demo 7
# testVector = img2vector('testDigits/0_13.txt')
# print(testVector[0, 0:31])

# demo 8
handwritingClassTest()
