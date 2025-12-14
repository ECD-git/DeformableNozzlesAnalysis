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
global step; step = 0.007; #[ms], should be lowest possible val for camera
# ruler length = [AS VISIBLE ON IMAGE, this might change slightly across days so this may become obselete]

def Calibrate(imPath):
    '''
    Docstring for Calibrate
    
    :param imPath: Description
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
    scale_factor = scale_factor*1000 #[pixels m^-1]
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
    '''
    Docstring for ProcessImage
    
    :param im: Description
    '''
    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im = clahe.apply(im)
    # Bilateral smoothing
    im = cv2.bilateralFilter(im, 9, 75, 75)
    ret, im = cv2.threshold(im, 40, 255, cv2.THRESH_BINARY)
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
    '''
    Docstring for Color
    
    :param im: Description
    '''
    # convert to color
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im

def GetAngle(top, right, left, rightErr, leftErr):
    '''
    Docstring for GetAngle
    
    :param x1: Description
    :param y1: Description
    :param x2: Description
    :param y2: Description
    '''
    OppA = (top[0]-right[0])
    AdjA = (right[1]-top[1])
    angleA, angleAErr = GetTanAngle(OppA, AdjA, rightErr[0], rightErr[1])
    OppB = (top[0]-left[0])
    AdjB = (left[1]-top[1])
    angleB, angleBErr = GetTanAngle(OppB, AdjB, leftErr[0], leftErr[1])
    return np.abs(angleA) + np.abs(angleB), np.mean([angleAErr,angleBErr])

def GetTanAngle(opp, adj, oppErr, adjErr):
    x = opp/adj
    xErr = oppErr/opp
    angleErr = np.rad2deg(xErr/(x**2+1))
    angle = np.rad2deg(np.arctan(opp/adj))
    return angle, angleErr

def GetTopPoint(x1,y1,x2,y2):
    '''
    Docstring for GetMidPoint
    
    :param x1: Description
    :param y1: Description
    :param x2: Description
    :param y2: Description
    '''
    if y1 < y2:
        return [x1,y1]
    else:
        return [x2,y2]

def MeanTopCluster(positions, minXdiff = 10, minYdiff = 15):
    '''
    Docstring for MeanTopCluster
    
    :param positions: Description
    :param minXdiff: Description
    :param minYdiff: Description
    '''
    # sort in y dir from low to high
    sorted = positions[positions[:, 1].argsort()]
    cluster = []

    for i in range(len(sorted)):
        if np.abs(sorted[i,1] - sorted[i+1,1]) < minYdiff:
            cluster.append(sorted[i])
        else:
            break;
    cluster = np.array(cluster)
    # error on x is the std of the mean, err on y is basically a pixel and is 0
    return [round(np.mean(cluster, axis=0)[0]),round(cluster[-1,1])], [np.sqrt(np.std(cluster, axis=0)[0]),0]

def MeanPos(position):
    '''
    Docstring for MeanPos
    
    :param position: x1,y1,x2,y2
    '''
    x1,y1,x2,y2 = position[0]
    mean = [round((x1+x2)/2),round((y1+y2)/2)]
    err = [np.sqrt(np.abs(x1-x2)/2),np.sqrt(np.abs(y1-y2)/2)]
    return mean, err

def Mask(im, r, x, y):
    maskradius = round(r*0.99) # we crop just below the radius
    circlemask = np.zeros_like(im)
    circlemask = cv2.circle(circlemask, (x,y), maskradius, (255,255,255), -1)
    maskx = round(r*0.6) 
    rectanglemask = np.zeros_like(im)
    rectanglemask = cv2.rectangle(rectanglemask, (x-maskx,0),(x+maskx,y+r), (255,255,255), -1)
    im = cv2.bitwise_and(im, circlemask)
    im = cv2.bitwise_and(im,rectanglemask)#[0:y+r,x-xcrop:x+xcrop])
    return im

def GetContourPosition(cnt):
    M = cv2.moments(cnt)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return center

def GetAbsDist(pos1, pos2):
    x = np.abs(pos1[0]-pos2[0])
    y = np.abs(pos1[1]-pos2[1])
    #print("DIST =",np.sqrt(x**2 + y**2))
    return np.sqrt(x**2 + y**2)

def Characterise(imPath, scale, scaleErr):
    '''
    Docstring for Characterise
    
    :param imPath: Description
    :out sprayAngle: The angle of the spray from the nozzle tip, error from the angular error of the houghLines function.
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
        minRadius=350,       
        maxRadius=2000) 
    
    # Draw only the first detected circle + crop to it
    if circles is not None:
        print("Light radius confirmed")
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        if r>550:
            r = 500
        print("RADIUS", r)
        cv2.circle(display, (x, y), r, (0, 255, 0), 2)  # Circle outline
        cv2.circle(display, (x, y), 2, (0, 255, 0), 3)  #
        
        # masking 
        total = Mask(total, r, x, y)

    '''
    cv2.imshow('display', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # probabilistic hough line transform to detect lines
    HoughError = np.pi/180
    lines = cv2.HoughLinesP(total, 1, HoughError, 50, minLineLength=70, maxLineGap=20)
    positions = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            position = GetTopPoint(x1,y1,x2,y2)
            positions.append(position)
            
    
    # make angles and positions a numpy array
    positions = np.array(positions)
    
    # Need to account for the cluster of line points at the top of the image due to the geom of the nozzle tip
    top, topErr = MeanTopCluster(positions) # this is eff the nozzle tip,
    # if wanna try get a breakup length, here is the best point

    # remove all points above the circle center point
    ind = np.where(positions[:,1] < y)
    positions = np.delete(positions,ind , axis=0)
    lines = np.delete(lines, ind, axis=0)

    # remove the left most point (avoid anomalies)
    for i in range(2):
        leftInd = np.where(positions[:,0] == np.min(positions[:,0]))
        positions = np.delete(positions,leftInd , axis=0)
        lines = np.delete(lines, leftInd, axis=0)

    for i in range(5):
        rightInd = np.where(positions[:,0] == np.max(positions[:,0]))
        positions = np.delete(positions,rightInd , axis=0)
        lines = np.delete(lines, rightInd, axis=0)

    for pos in positions:
        cv2.circle(display, (pos[0], pos[1]), 1, (255,0,0), 3)



    leftInd = np.where(positions[:,0] == np.min(positions[:,0]))
    rightInd = np.where(positions[:,0] == np.max(positions[:,0]))
    left, leftErr = MeanPos(lines[leftInd][0])
    right, rightErr = MeanPos(lines[rightInd][0])
    sprayAngle, sprayAngleErr = GetAngle(top, right, left, rightErr, leftErr)

    '''
    # Draw all these points of interest
    cv2.circle(display, (top), 1, (0,255,0), 8)
    cv2.circle(display, (left), 1, (0,255,0), 8)
    cv2.circle(display, (right), 1, (0,255,0), 8)
    cv2.line(display, left, top, (0,0,255),3)
    cv2.line(display, right, top, (0,0,255),3)
    
    cv2.putText(display, "Angle = {0:3.2f} +- {1:3.2f}".format(sprayAngle,sprayAngleErr),(top[0]+25, top[1]+50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),3)
    
    cv2.imshow('display', display)
    cv2.imwrite(directory+"/test.png", display)
    #cv2.imshow('total', total)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # VELOCITY
    boundary = round((y+r)*0.75) #set the boundary to be about 75% of the distance from the nozzle tip to the edge of the light ring, to ensure all droplets are caught
    # Iterate over the first 8-10 images
    lastJetPos, lastJetPosErr, firstJetPos, firstJetPosErr = [], [], [], []
    for i in range(len(images[:8])):
        diff = cv2.subtract(Gray(cv2.imread(imPath+images[i])),background)
        diff = ProcessImage(diff)
        diff = Mask(diff, r, x, y)
        
        contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # sort the contours by area
        contoursSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        # filter by area
        minArea = 300
        contoursSorted = [contour for contour in contours if cv2.contourArea(contour) > minArea]
        
        diff = Color(diff)
        cv2.drawContours(diff, contoursSorted, -1, (0, 0, 255), 2)

        # sort contours by y value in    
        contoursSorted = sorted(contoursSorted, key=lambda x: GetContourPosition(x)[1], reverse=True)
            
        # Display
        
        # Quick velocity cal, if its more than 5, assume anomalous and remove
        if i > 0:
            for j in range(2):
                print(GetAbsDist(GetContourPosition(contoursSorted[0]), top)/(i*step*scale))
                if GetAbsDist(GetContourPosition(contoursSorted[0]), top)/(i*step*scale) > 5:
                    contoursSorted = contoursSorted[1:]
        cv2.circle(diff, GetContourPosition(contoursSorted[0]), 50, (0,0,255),5)

        for cnt in contoursSorted:
            cv2.circle(diff, GetContourPosition(cnt), 3, (255,0,0),3)
        cv2.circle(diff, (top), boundary, (0, 255, 0), 2)  # Circle outline
        cv2.circle(diff, (top), 2, (0, 255, 0), 3)  #
        
        cv2.imshow(image, diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # check if lowest contour is outside of bounds
        if GetAbsDist(GetContourPosition(contoursSorted[0]), top) > boundary:
            print(i)
            lastframe = i
            lastJetPos = GetContourPosition(contoursSorted[0])
            x,y,w,h = cv2.boundingRect(contoursSorted[0])
            lastJetPosErr = [w/2, h/2]
            break;
        if i == 0:
            firstJetPos = GetContourPosition(contoursSorted[0])
            #x,y,w,h = cv2.boundingRect(contoursSorted[0])
            #firstJetPosErr = [w/2,h/2]

    timeStep = 0.007 #[s]
    time = i * timeStep
    dist = GetAbsDist(lastJetPos, firstJetPos)
    distErr = np.sqrt(lastJetPosErr[0]**2 + lastJetPosErr[1]**2)
    velocity = (dist/scale)/time 
    VelocityErr = (distErr)/(time*scale) * (velocity * (scaleErr/scale))

    # Display results
    cv2.line(display, left, top, (0,0,255),3)
    cv2.line(display, right, top, (0,0,255),3)
    
    cv2.putText(display, "Angle = {0:3.2f} +- {1:3.2f} deg".format(sprayAngle,sprayAngleErr),(top[0]-50, top[1]+50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),3)
    cv2.putText(display, "Vel = {0:3.2f} +- {1:3.2f} m/s".format(velocity, VelocityErr), (right[0]-50, right[1]+50), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),3)

    #cv2.line(diff, lastJetPos, firstJetPos, (255,0,255), 3)
    cv2.imshow(image, display)
    cv2.imwrite(directory+"/"+"Result"+".png", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        print("Scale factor in pixels per m", scale_factor)
        for nozzle in os.listdir(directory+"/Sessions/"+session):
            if nozzle != "cal" and nozzle != "cal.txt" and nozzle[0] != '.':
                print("--" + nozzle)
                for result in os.listdir(directory+"/Sessions/"+session+'/'+nozzle):
                    # ie path to result is directory+"/Sessions/"+session+'/'+nozzle+'/'+result
                    print("---"+result)
                    
                    test = directory+"/Sessions/"+session+'/'+nozzle+'/'+"0.5b1"+'/'
                    Characterise(test, scale_factor, scale_factor_err);
                        
                    break;
if __name__ == "__main__":
    main();

