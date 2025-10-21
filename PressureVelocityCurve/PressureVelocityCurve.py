import os
import numpy as np
import cv2
from pathlib import Path
import math
import skimage.morphology as morph

images_path = str(Path(__file__).resolve().parent) + "/Images/"
calibration_path = str(Path(__file__).resolve().parent) + "/Calibration_Test/"
time_step = 10 #[ms] BE SURE TO CHECK THIS BEFORE TAKING RESULTS
ruler_length = 6 # [cm]

def Calibrate():
    images = sorted(os.listdir(calibration_path))
    images = [string for string in images if not string[0] == '.']
    print(images[0])
    ruler = cv2.imread(calibration_path + images[1])
    background = cv2.imread(calibration_path + images[0])

    diff = cv2.subtract(background, ruler)
    
    diff = diff[int(diff.shape[0]*0.2) : int(diff.shape[0]*0.8), int(diff.shape[1]*0.715) : int(diff.shape[1]*0.8)]
    
    test = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    rev, test = cv2.threshold(test, 10, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    test = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    test = morph.skeletonize(test)
    test = (test*255).astype(np.uint8)

    lines = cv2.HoughLinesP(test, 1, math.pi/2, 10, None, 5, 3)
    
    diff = cv2.subtract(background, ruler)

    coords = np.empty([len(lines),4])
        
    for i in range(len(lines)):
        x1,y1,x2,y2 = lines[i][0]
        coords[i] = [x1,y1,x2,y2]
        cv2.line(diff, (int(diff.shape[1]*0.715)+x1,int(diff.shape[0]*0.2)+y1), (int(diff.shape[1]*0.715)+x2,int(diff.shape[0]*0.2)+y2), (0,255,0), 1)
    
    y_scale_pixels = (coords[:,1] + coords[:,3])/2
    scale_factor = ruler_length/(np.max(y_scale_pixels) - np.min(y_scale_pixels))

    cv2.imshow("diff", diff[int(diff.shape[0]*0):int(diff.shape[0]*0.9), int(diff.shape[1]*0.5):int(diff.shape[1]*0.9)])
    #cv2.imwrite(str(Path(__file__).resolve().parent) + "/Ruler.png", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return scale_factor;

def ProcessResults(): # ADD INPUT SCALE FACTOR HERE TO RETURN ACCURATE RESULTS
    # Iterating over all results in the folders
    for file in os.listdir(images_path):
        images = sorted(os.listdir(images_path + file))
        # Theres duplicate hidden files starting with a '.' for some reason so im skipping those
        images = [string for string in images if not string[0] == '.']
        back = images_path + file + '/' + images[-1]
        # The array is ordered as [image0,...,imageN,background]
        first = images_path + file + '/' + images[0]
        last = images_path + file + '/' + images[-2]

        background_im = cv2.imread(back)
        first_image = cv2.imread(first)
        last_image = cv2.imread(last)
        # need to get number of iterations of last image
        num_frames = np.size(images) - 2
        
        # Take difference
        difference = cv2.subtract(last_image, first_image)
        # Crop
        difference = difference[int(difference.shape[0]*0.2):int(difference.shape[0]*0.9), int(difference.shape[1]*0.4):int(difference.shape[1]*0.6)]
        # Convert to Grayscale
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        # seperate objects from background (should just be black)
        rev, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        # Im not overly sure what these are for, but are important the thing breaks without them
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        dilated = cv2.dilate(close, kernel, iterations=2)
        
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours)==2 else contours[1]
        # gets the 10 largest contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        dimensions = []
        
        # stored as x,y,w,h
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                dimensions.append(cv2.boundingRect(cnt))
          
        #for cnt in contours:
        #    if cv2.contourArea(cnt) > 50:
        #        x, y, w, h = cv2.boundingRect(cnt)
        #        cv2.rectangle(difference, (x,y), (x+w, y+h), (0,255,0), 1)
        #        cv2.putText(difference, str(w) + " by " + str(h) + " pixels", (x-100, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) 

        cv2.rectangle(difference, (min([i[0] for i in dimensions]), min([i[1] for i in dimensions])), (min([i[0] for i in dimensions]) + max([i[2] for i in dimensions]), max([i[1] for i in dimensions]) + max([i[3] for i in dimensions])), (255,0,0), 1)
        
        size = [min([i[0] for i in dimensions]) + max([i[2] for i in dimensions])-min([i[0] for i in dimensions]), max([i[1] for i in dimensions]) + max([i[3] for i in dimensions]) - min([i[1] for i in dimensions])]
        cv2.putText(difference, str(size[0]) + " by " + str(size[1]) + "pixels", (10,min(i[1] for i in dimensions)-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
        #print(dimensions)

        cv2.imshow("First", first_image[int(first_image.shape[0]*0.2):int(first_image.shape[0]*0.9), int(first_image.shape[1]*0.4):int(first_image.shape[1]*0.6)])
        #cv2.imwrite(str(Path(__file__).resolve().parent) + "/first.png", first_image[int(first_image.shape[0]*0.2):int(first_image.shape[0]*0.9), int(first_image.shape[1]*0.4):int(first_image.shape[1]*0.6)])

        cv2.imshow("last", last_image[int(last_image.shape[0]*0.2):int(last_image.shape[0]*0.9), int(last_image.shape[1]*0.4):int(last_image.shape[1]*0.6)])
        #cv2.imwrite(str(Path(__file__).resolve().parent) + "/last.png", last_image[int(last_image.shape[0]*0.2):int(last_image.shape[0]*0.9), int(last_image.shape[1]*0.4):int(last_image.shape[1]*0.6)])

        #cv2.imshow("dilated", dilated)
        cv2.imshow("differece", difference)
        #cv2.imwrite(str(Path(__file__).resolve().parent) + "/difference.png", difference)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

def PressureVelocityCurve():
    scale_factor = Calibrate()
    ProcessResults()
    print("EXECUTED")

PressureVelocityCurve()
