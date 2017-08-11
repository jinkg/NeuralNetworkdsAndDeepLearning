import numpy as np


def loadDataSet():
    dataMat = []
    dataLabel = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            arrayLine = line.strip().split()
            dataMat.append([1.0, float(arrayLine[0]), float(arrayLine[1])])
            dataLabel.append(int(arrayLine[2]))
    return dataMat, dataLabel


def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))


def gradAscent(dataArray, targetLabels):
    dataMatrix = np.mat(dataArray)
    labelMat = np.mat(targetLabels).transpose()

    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))
    itCount = 500
    alpha = 0.001
    for i in range(itCount):
        result = sigmoid(dataMatrix * weights)
        error = (labelMat - result)
        weights += alpha * dataMatrix.transpose() * error
    print(type(weights))
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, dataLabel = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if dataLabel[i] == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    aix = fig.add_subplot(111)
    aix.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    aix.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    plt.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


dataMat, dataLabel = loadDataSet()
weights = gradAscent(dataMat, dataLabel)
plotBestFit(weights)
