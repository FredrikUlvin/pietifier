import numpy as np
import cv2 as cv
from matplotlib import pyplot as mp
from scipy.signal import filtfilt, butter, argrelmax, argrelmin
import time
from math import sqrt

maxpoints = argrelmax
minpoints = argrelmin
# Filter with 0 phase lag
b, a = butter(3,0.05)

# To check working directory
import os
print os.getcwd()


# Return colourvalue for RGB, rows and colums both
def colourSum(img):
    rowSum = img.sum(axis=1)
    colSum = img.sum(axis=0)
    rowSum = [rowSum[:,0],rowSum[:,1],rowSum[:,2]]
    colSum = [colSum[:,0],colSum[:,1],colSum[:,2]]
    return rowSum, colSum

def colourVar(img):
    rowSum = img.sum(axis=1)
    colSum = img.sum(axis=0)

    rowVar = np.array([(i - (np.sum(i, dtype=np.uint64)/3)) for i in rowSum], dtype=np.int64)
    colVar = np.array([(i - (np.sum(i, dtype=np.uint64)/3)) for i in colSum], dtype=np.int64)

    rowVar = [rowVar[:,0], rowVar[:,1], rowVar[:,2]]
    colVar = [colVar[:,0], colVar[:,1], colVar[:,2]]

    return rowVar, colVar

def maxSection(a, section):
    if len(a)<section:
        return

    high = sum(a[0:section])
    block = high
    topp = 0

    for i in range(len(a)-section):
        block = block - a[i] + a[section+i]
        if block > high:
            high = block
            topp = i + 1
    return topp

# Return max indices
def getSquare(rowSum, colSum, secLen):

    rowLen = len(rowSum)
    colLen = len(colSum)

    rowAvg = sum(rowSum)/len(rowSum)
    colAvg = sum(colSum)/len(colSum)

    rowMaxSection = maxSection(rowSum, int(rowLen/secLen))
    colMaxSection = maxSection(colSum, int(colLen/secLen))

    rowMaxIndex = np.argmax(rowSum[rowMaxSection:rowMaxSection+int(rowLen/secLen)])+rowMaxSection
    colMaxIndex = np.argmax(colSum[colMaxSection:colMaxSection+int(colLen/secLen)])+colMaxSection

    rowMax = rowSum[rowMaxIndex]
    colMax = colSum[colMaxIndex]

    aspect = (rowMax*rowLen/float(rowAvg))/(colMax*colLen/float(colAvg))

    # Total of excess pixels in the max section
    #total = (sum(rowSum[rowMaxSection:rowMaxSection+int(rowLen/secLen)])-rowAvg*int(rowLen/secLen))/255

    # Total of non-zero values in array
    tooLow = rowSum < 0
    total = (np.sum(rowSum[np.logical_not(tooLow)], dtype=np.uint64))/25

    sideX = int(sqrt(total/aspect))
    sideY = int(sqrt(total*aspect))

    if sideX > colLen:
        sideX = colLen
    elif sideX < 0:
        sideX = 0
    if sideY > rowLen:
        sideY = rowLen
    elif sideY < 0:
        sideY = 0

    if colMaxIndex - sideX/2 < 0:
        colMaxIndex = sideX/2
    elif colMaxIndex + sideX/2 > colLen:
        colMaxIndex = colLen - sideX/2

    if rowMaxIndex - sideY/2 < 0:
        rowMaxIndex = sideY/2
    elif rowMaxIndex + sideY/2 > rowLen:
        rowMaxIndex = rowLen - sideY/2

    return (colMaxIndex, rowMaxIndex) , (sideX, sideY)

def getMaxPoints(arr):
    # [TODO] Work out for RGB rather than array, and maybe we don't need the filter, but hopefully speeds it up.
    # Reference http://scipy-cookbook.readthedocs.io/items/FiltFilt.html
    arra = filtfilt(b,a,arr)
    maxp = maxpoints(arra, order=(len(arra)/20), mode='wrap')
    minp = minpoints(arra, order=(len(arra)/20), mode='wrap')

    points = []

    for i in range(3):
        mas = np.equal(np.greater_equal(maxp,(i*(len(arra)/3))), np.less_equal(maxp,((i+1)*len(arra)/3)))
        k = np.compress(mas[0], maxp)
        if len(k)==0:
            continue
        points.append(sum(k)/len(k))

    if len(points) == 1:
        return points, []

    points = np.compress(np.greater_equal(arra[points],(max(arra)-min(arra))*0.40 + min(arra)),points)
    rifts = []
    for i in range(len(points)-1):
        mas = np.equal(np.greater_equal(minp, points[i]),np.less_equal(minp,points[i+1]))
        k = np.compress(mas[0], minp)
        rifts.append(k[arra[k].argmin()])

    return points, rifts

def drawSquares(img, centre, sides, colour, line):
    x , y   = len(img[0]), len(img)
    x0 , y0 = centre[0], centre[1]
    x1 , y1 = 0 , 0

    if centre[0] - ( sides[0] / 2 ) < 0:
        x0 , x1 = 0 , sides[0]
    elif centre[0] + (sides[0] / 2) > x:
        x0 , x1 = x - sides[0], x
    else:
        x0 , x1 = centre[0] - int(sides[0]/2)  ,  centre[0] + int(sides[0] / 2)

    if centre[1] - ( sides[1] / 2 ) < 0:
        y0 , y1 = 0 , sides[1]
    elif centre[1] + (sides[1] / 2) > y:
        y0 , y1 = y - sides[1], y
    else:
        y0 , y1 = centre[1] - int(sides[1]/2)  ,  centre[1] + int(sides[1] / 2)

    return cv.rectangle(img , (x0,y0) , (x1,y1) , colour , thickness=line)

def addLines(img, centres, sides):
    thickness = int(min(len(img), len(img[0]))*0.005)
    B = img
    #B = 255*np.ones((len(img),len(img[0]),3), np.uint8)#, np.zeros((len(img),len(img[0]),3), np.uint8), np.zeros((len(img),len(img[0]),3), np.uint8)

    for i in range(3):
            cv.line(B,(0,centre[i][1]-sides[i][1]/2),(len(B[0]),centre[i][1]-sides[i][1]/2),(0,0,0),thickness)
            cv.line(B,(0,centre[i][1]+sides[i][1]/2),(len(B[0]),centre[i][1]+sides[i][1]/2),(0,0,0),thickness)
            cv.line(B,(centre[i][0]-sides[i][0]/2,0),(centre[i][0]-sides[i][0]/2,len(B)),(0,0,0),thickness)
            cv.line(B,(centre[i][0]+sides[i][0]/2,0),(centre[i][0]+sides[i][0]/2,len(B)),(0,0,0),thickness)

#    cv.line(B,(0,centre[0][1]-sides[0][1]/2),(len(B[0]),centre[0][1]-sides[0][1]/2),(0,0,0),thickness)
#    cv.line(B,(0,centre[0][1]+sides[0][1]/2),(len(B[0]),centre[0][1]+sides[0][1]/2),(0,0,0),thickness)
#    cv.line(B,(centre[0][0]-sides[0][0]/2,0),(centre[0][0]-sides[0][0]/2,len(B)),(0,0,0),thickness)
#    cv.line(B,(centre[0][0]+sides[0][0]/2,0),(centre[0][0]+sides[0][0]/2,len(B)),(0,0,0),thickness)
    return B

## Main program

# Get image
img = cv.imread('pietifier/test2.jpg',1)
#img = colourVar(img)
rowsum, colsum = colourVar(img)

centre = [None, None, None]
sides = [None, None, None]

centre[0], sides[0] = getSquare(rowsum[0], colsum[0], 3)
centre[1], sides[1] = getSquare(rowsum[1], colsum[1], 3)
centre[2], sides[2] = getSquare(rowsum[2], colsum[2], 3)


img2 = 255*np.ones((len(img),len(img[0]),3), np.uint8)

drawSquares(img2, centre[0], sides[0], (255,0,0),-1)
drawSquares(img2, centre[1], sides[1], (0,255,0),-1)
drawSquares(img2, centre[2], sides[2], (0,0,255),-1)

addLines(img2, centre, sides)

'''
start = time.clock()
a = np.argmax([sum(test[i:i+int(len(test)/3)]) for i in range(int(2*len(test)/3))])
end = time.clock()
print a, end - start
'''
# Print img

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

mp.figure()

mp.hold(True)

mp.subplot(1,2,1)
mp.imshow(img, cmap = 'gray', interpolation = 'bicubic')
mp.xticks([]), mp.yticks([])

mp.subplot(1,2,2)
mp.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
mp.xticks([]), mp.yticks([])

mp.show()

mp.hold(False)
'''
mp.subplot(2,1,1)
mp.hold(True)
mp.plot(t, rowsum[0])

mp.subplot(2,1,2)
mp.plot(t, filtfilt(b,a,rowsum[0]))

mp.hold(False)
mp.show()
'''
# Exit program
a = cv.waitKey(0)
if a == 27:
    cv.destroyAllWindows()
elif a == ord('s'):
    cv.imwrite('test.png', img)
    cv.destroyAllWindows()
