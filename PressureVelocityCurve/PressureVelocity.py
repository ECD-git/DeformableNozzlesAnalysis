# Ethan D 14-12-25
import os
import numpy as np
import cv2
from pathlib import Path
import math

global TIMESTEP; TIMESTEP = 7 / 1000
global TIMESTEPERR; TIMESTEPERR = 1/1000

parentDir = str(Path(__file__).resolve().parent)
nozzleDir = parentDir + "/N1 28-10-25"
results = "0.5b1"

def ReadScaleFactor(Dir):
    data = np.genfromtxt(Dir + "/cal.txt", delimiter=',')
    return data[0], data[1]

def Velocity(filePath, scale, scaleErr):
    images = sorted(os.listdir(filePath))
    images = [string for string in images if not string[0] == '.']

    print(filePath + "/" + images[0])

    # Process images
    first = cv2.imread(filePath + "/" + images[0])
    last = cv2.imread(filePath + "/" + images[-1])
    firstGray = Gray(first)
    lastGray = Gray(last)
    difference = cv2.subtract(lastGray, firstGray)
    differenceProcessed = ProcessImage(difference)

    # mask to remove any effects from the lightring etc
    mask = np.zeros_like(differenceProcessed)
    mask = cv2.rectangle(mask, (round(mask.shape[1]*0.45), 30), (round(mask.shape[1]*0.65), round(mask.shape[0]*0.95)), (255,255,255), -1)
    differenceProcessed = cv2.bitwise_and(differenceProcessed, mask)
    
    # Find contours
    contours, hierarchy = cv2.findContours(differenceProcessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    minArea = 350
    contoursSorted = [contour for contour in contours if cv2.contourArea(contour) > minArea]
    
    # draw contours
    differenceProcessed = Color(differenceProcessed)
    cv2.drawContours(differenceProcessed, contoursSorted, -1, (0, 0, 255), 2)
    cv2.imshow('test', differenceProcessed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find bounding boxes
    boundingBoxes = []
    for contour in contoursSorted:
        x, y, w, h = cv2.boundingRect(contour)
        boundingBoxes.append([x,y,x+w,y+h]) # store 4 corners of bounding box
    boundingBoxes = np.array(boundingBoxes) # convert to np array

    x_min = np.min(boundingBoxes[:,0]) #
    x_max = np.max(boundingBoxes[:,2]) #
    y_min = np.min(boundingBoxes[:,1]) #
    y_max = np.max(boundingBoxes[:,3]) #

    print([x_min, x_max, y_min, y_max])

    distPixels = y_max - y_min
    time = TIMESTEP * (len(images)-1)
    dist = distPixels/scale
    velocityM = dist/time
    velocityMErr = np.sqrt((scaleErr/scale)**2 + (TIMESTEPERR/TIMESTEP)**2) * velocityM
    print(velocityM)
    
    cv2.rectangle(last, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
    cv2.putText(last, "vel = {0:3.2f} +- {1:3.2f}".format(velocityM, velocityMErr), (x_max+15, round((y_max-y_min)/2)), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),3)
    
    #cv2.imshow('fist', first)
    cv2.imshow('last', last)
    cv2.imwrite(parentDir+"/" + results + ".png", last)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    vel = 0
    return vel # pixels per second

def ProcessImage(im):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    # Bilateral smoothing
    im = cv2.bilateralFilter(im, 9, 75, 75)
    ret, im = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY)
    return im

def Gray(im):
    '''
    Docstring for Gray
    
    :param im: Description
    '''
    # convert to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im

def Color(im):
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im

def main():
    if os.path.exists(parentDir + "/cal.txt"):
        scaleFactor, scaleFactorErr = ReadScaleFactor(parentDir) # pixels per m
    Vel = Velocity(nozzleDir + "/" + results, scaleFactor, scaleFactorErr)
    

if __name__ == "__main__":
    main();