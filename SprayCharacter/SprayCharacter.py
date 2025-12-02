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
    scale_factor_err = (0.5/40) + (1-np.cos(np.pi/18))
    
    # Display results
    '''
    plt.figure()
    plt.plot(projection, label='projection', linestyle='-')
    plt.scatter(peaks, projection[peaks], label='peaks')
    plt.title("Image Analysis Calibration")
    plt.ylabel("horizontally summed pixel values")
    plt.xlabel("pixels")
    plt.legend()
    
    cv2.imshow("ruler", th2)
    #cv2.imwrite(directory+"/markings.png", th2)
    cv2.imshow('body', body)
    #cv2.imwrite(directory+"/ruler.png", body)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return scale_factor, scale_factor_err;

def Characterise(imPath):
    '''
    
    '''
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
        scale_factor, scale_factor_err = Calibrate(directory+"/Sessions/"+session+"/cal");
        print("Scale factor in pixels per cm", scale_factor)
        for nozzle in os.listdir(directory+"/Sessions/"+session):
            if nozzle != "cal":
                print("--" + nozzle)
                for result in os.listdir(directory+"/Sessions/"+session+'/'+nozzle):
                    # ie path to result is directory+"/Sessions/"+session+'/'+nozzle+'/'+result
                    print("---"+result)
                    
                    # Sort images and remove non image files
                    test = directory+"/Sessions/"+session+'/'+nozzle+'/'+"3.5b1"+'/'
                    images = sorted(os.listdir(test))
                    images = [string for string in images if not string[0] == '.']
                    
                    # Find and isolate background
                    background = cv2.imread(test+images[0])
                    images = images[1:]

                    # iterate over all images in file
                    for image in images:
                        diff = cv2.subtract(cv2.imread(test+image),background)
                        cv2.imshow('diff', diff)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        
                    break;

                    # VELOCITY
                    # - TIME ERROR IS HALF TIME STEP
                

if __name__ == "__main__":
    main();

