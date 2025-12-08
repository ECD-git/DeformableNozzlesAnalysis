# dependencies
import os;
from pathlib import Path;
import numpy as np;
import math;
import scipy as sp;
import cv2;
import matplotlib.pyplot as plt;

# Constants/paths
directory = str(Path(__file__).resolve().parent);
step = 7; #[ms], should be lowest possible val for camera
# ruler length = [AS VISIBLE ON IMAGE, this might change slightly across days so this may become obselete]

def UnitVec(num):
    return num/np.abs(num)

def AngleAgainstNormal(x1,y1,x2,y2):
    angle = np.arctan(np.abs(x1-x2)/np.abs(y1-y2))
    if y1>y2 and x1<x2:
        angle = -angle
    elif y2>y1 and x2<x1:
        angle = -angle
    return angle

def ExtendLine(x1,y1,x2,y2, length=200):
    angle = AngleAgainstNormal(x1,y1,x2,y2)
    dx = x1-x2
    dy = y1-y2
    length0 = np.sqrt(dx**2 + dx**2)
    dx_normalized = dx / length0
    dy_normalized = dy / length0

    # Extend the line in both directions
    x1_new = int(x1 - length * dx_normalized)
    y1_new = int(y1 - length * dy_normalized)
    x2_new = int(x2 + length * dx_normalized)
    y2_new = int(y2 + length * dy_normalized)

    return (x1_new, y1_new, x2_new, y2_new)


def Calibrate(imPath):
    '''
    Takes provided tiff image (of a ruler over a back light) and returns a scale factor of pixels to mm from ruler
    '''
    print("--cal")
    images = sorted(os.listdir(imPath))
    images = [string for string in images if not string[0] == '.'] # this accounts for those weird ._ files macOS generates sometimes

    # Process image
    ruler = cv2.imread(imPath + '/'+images[0])
    mean_img=cv2.pyrMeanShiftFiltering(ruler,20,30)
    gray_img = cv2.cvtColor(mean_img, cv2.COLOR_BGR2GRAY)
    th=cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    th2 = th[round(ruler.shape[0]*0.1):round(ruler.shape[0]*0.45),round(ruler.shape[1]*0.84):round(ruler.shape[1]*0.87)]
    body = ruler[round(ruler.shape[0]*0):round(ruler.shape[0]*0.5),round(ruler.shape[1]*0.68):round(ruler.shape[1]*0.9)]
    # this effectively generates a thin image of horizontal lines
    # the bounds of the cropping may be best auto generated but since we're only calibrated 2-3 times we've opted to do it manually

    # Find peaks and scale factor
    projection = np.sum(th2, axis=1)
    peaks, properties = sp.signal.find_peaks(projection, height=4000)
    # num of peaks = num of mm
    scale_factor = (peaks[-1]-peaks[0])/(len(peaks)-1) #[pixels mm^-1]
    scale_factor = scale_factor*10 #[pixels cm^-1]
    # adding the ruler and angular fractional erros 
    scale_factor_err = ((0.5/40) + (1-np.cos(np.pi/18)))*scale_factor
    
    # Display results just for confirmation
    cv2.imshow("ruler", th2)
    #cv2.imwrite(directory+"/markings.png", th2)
    cv2.imshow('body', body)
    #cv2.imwrite(directory+"/ruler.png", body)
    #plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return scale_factor, scale_factor_err;

def ProcessImage(im):
    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    # Bilateral smoothing
    im = cv2.bilateralFilter(im, 9, 75, 75)
    ret, im = cv2.threshold(im, 40, 255, cv2.THRESH_BINARY)
    return im

def Gray(im):
    # convert to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im

def Color(im):
    # convert to color
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im

def Characterise(imPath):
    '''
    
    '''
    # Sort images and remove non image files
    images = sorted(os.listdir(imPath))
    images = [string for string in images if not string[0] == '.']
    
    # Find and isolate background
    background = Gray(cv2.imread(imPath+images[0]))
    total = Gray(cv2.imread(imPath+images[1]))
    total = cv2.subtract(total, background)
    display = total; # An unprocessed copy for figures
    total = ProcessImage(total)

    images = images[2:]
    # iterate over all images in file
    for image in images: #[round(len(images)*0.5):round(len(images)*0.5)+1]:
        diff = cv2.subtract(Gray(cv2.imread(imPath+image)),background)
        display = cv2.addWeighted(display, 1, diff, 1, 0.5)
        diff = ProcessImage(diff)
        total = cv2.addWeighted(total, 1, diff, 0.5, 0.5)
    display = Color(display)    

    # Find the inner bound of the light, and use it as a mask
    circles = cv2.HoughCircles(
        total,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,      
        param1=40,         
        param2=20,      
        minRadius=450,       
        maxRadius=550) 
    # Draw only the first detected circle + crop to it
    if circles is not None:
        print("Light radius confirmed")
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        print("RADIUS", r)
        cv2.circle(display, (x, y), r, (0, 255, 0), 2)  # Circle outline
        cv2.circle(display, (x, y), 2, (0, 255, 0), 3)  #
        # masking 
        maskradius = round(r*0.99) # we crop just below the radius
        circlemask = np.zeros_like(total)
        circlemask = cv2.circle(circlemask, (x,y), maskradius, (255,255,255), -1)
        maskx = round(r*0.6) 
        rectanglemask = np.zeros_like(total)
        rectanglemask = cv2.rectangle(rectanglemask, (x-maskx,0),(x+maskx,y+r), (255,255,255), -1)
        total = cv2.bitwise_and(total, circlemask)
        total = cv2.bitwise_and(total,rectanglemask)#[0:y+r,x-xcrop:x+xcrop])
        

    # probabilistic hough line transform to detect lines
    lines = cv2.HoughLinesP(total, 1, np.pi/180, 50, minLineLength=70, maxLineGap=20)
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = AngleAgainstNormal(x1,y1,x2,y2)
            angles.append(angle)
            cv2.line(display, (x1,y1), (x2,y2), (0,0,255),1)
    '''
    minMaxIndex = angles.index(min(angles)), angles.index(max(angles))
    print("ANGLES = "+str(angles[minMaxIndex[0]]) +' '+str(angles[minMaxIndex[1]]))
    for i in minMaxIndex:
        x1,y1,x2,y2 = lines[i][0]
        x1,y1,x2,y2 = ExtendLine(x1,y1,x2,y2)
        cv2.line(display, (x1+x-xcrop,y1), (x2+x-xcrop,y2), (0,255,0),2)
        cv2.circle(display, (x1+x-xcrop,y1), 2, (255, 0, 0), 3)
    '''
    # TODO:
    # - GET SLOPE, INTERCEPT, AND JOINING OF LINES
    # - COMPUTE ANGLE BETWEEN THEN
    # - GET MEAN LINE and find extension beyond joining point to top of image
    #
    #
    
    
    cv2.imshow('display', display)
    cv2.imshow('total', total)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # VELOCITY
    # - TIME ERROR IS HALF TIME STEP

    sprayAngle, breakLength, velocity = 0,0,0
    return sprayAngle, breakLength, velocity;


def main():
    # Measurements are 0.5b to 3b in increments of 0.5 bar, times 3, so 3 sets of 6 measurements
    # each measurement consists of the spray angle,breakup length and velocity for that pressure/iteration
    # array = numpy.array(["", numpy.empty((3,6), dtype=object)],dtype=object)
    # result = numpy.array([angle, length, velocity])
    pressures = np.array([0.5,1.0,1.5,2.0,2.5,3.0])

    # Iterate over all sessions, calibrates, then iterates over all resultds
    for session in os.listdir(directory+"/Sessions/"):
        print("-"+session)
        if os.path.exists(directory+"/Sessions/"+session+"/cal.txt"):
            data = np.genfromtxt(directory+"/Sessions/"+session+"/cal.txt", delimiter=',')
            scale_factor = data[0]
            scale_factor_err = data[1]
        else:
            scale_factor, scale_factor_err = Calibrate(directory+"/Sessions/"+session+"/cal");
            np.savetxt(directory+"/Sessions/"+session+"/cal.txt", np.array([scale_factor, scale_factor_err]), delimiter=',')
        print("Scale factor in pixels per cm", scale_factor)
        for nozzle in os.listdir(directory+"/Sessions/"+session):
            if nozzle != "cal" and nozzle != "cal.txt" and nozzle[0] != '.':
                print("--" + nozzle)
                for result in os.listdir(directory+"/Sessions/"+session+'/'+nozzle):
                    # ie path to result is directory+"/Sessions/"+session+'/'+nozzle+'/'+result
                    print("---"+result)
                    
                    test = directory+"/Sessions/"+session+'/'+nozzle+'/'+"3.5b1"+'/'
                    Characterise(test);
                        
                    break;
                

if __name__ == "__main__":
    main();

