# Desired modules
import numpy as np;
import matplotlib.pyplot as pyplot;
import os;
from pathlib import Path;

# Set the default path for sets of results to a folder could Results in the parent dictionary
default_path = str(Path(__file__).resolve().parent) + "/Results/"

for filename in os.listdir(default_path):
    if(filename==""):
        print("No more files found")
        break;
    try:
        print("Current file: " + filename)
        rawdata = np.genfromtxt(default_path + filename, delimiter=",", skip_header=1, dtype=float);
    except:
        print("No more files found")
        break;
    # Sort the data set by the "pressure" column to group all same values together
    # This step isnt strictly needed but its useful for bug fixing so I left it in
    rawdata = rawdata[rawdata[:,0].argsort()];
    
    # gathers every unique pressure value into an array, makes a seperate array for errors (assigned to be 0.1 bar)
    pressures = np.unique(rawdata[:,0]);
    pressures_err = np.repeat(0.1, np.size(pressures));
    # create empty arrays for the flows and errors now
    flow_rates = np.empty(np.size(pressures));
    flow_rates_err = np.empty(np.size(pressures));

    # index over all unique pressures
    for pressure in pressures:
        # get all indexes for all data points we have for this pressure
        ind = np.where(rawdata[:,0]==pressure)[0];
        # mean and std the data points given by ind, add to arrays in the same position as this pressure
        # the mean is weighted by the 1/variance of the values, obtained by adding the standard errors in quadrature
        flow_rates[np.where(pressures==pressure)[0]] = np.average(rawdata[ind,1]/rawdata[ind,2], weights=np.pow((np.pow(rawdata[ind,1]/rawdata[ind,3],2))+np.pow(rawdata[ind,4]/rawdata[ind,2],2),-1));
        flow_rates_err[np.where(pressures==pressure)[0]] = np.std(rawdata[ind,1]/rawdata[ind,2], ddof=1)/np.sqrt(np.size(pressures));
    
    # do a least squares fit of data to a 1d polynomial
    coef, cov = np.polyfit(pressures, flow_rates, 1, cov=True, w=np.pow(np.pow(flow_rates_err,2),-1))

    reduced_chi_squared = np.sum((np.polyval(coef, pressures)-flow_rates)**2)/np.size(pressures)

    # plot all the data and the worked results
    fig = pyplot.figure()
    pyplot.scatter(rawdata[:,0], rawdata[:,1]/rawdata[:,2], marker='x');
    pyplot.scatter(pressures, flow_rates, marker='o', color='red');
    pyplot.errorbar(pressures, flow_rates, yerr=flow_rates_err, xerr=pressures_err, linestyle='none', color='red')
    pyplot.plot(rawdata[:,0], coef[0]*rawdata[:,0] + coef[1], "r--")
    pyplot.text(np.min(pressures), np.max(flow_rates), "y = {0:3.2f}$\pm${2:2.1f} x + {1:3.2f}$\pm${3:2.1f} \n$\chi^2_R$ = {4:3.2f}".format(coef[0], coef[1], np.sqrt(cov[0][0]), np.sqrt(cov[1][1]), reduced_chi_squared))
    fig.supxlabel("Pressure /bar")
    fig.supylabel("Flow rate /ml s$^{-1}$")
    fig.suptitle("Pressure/Flow rate calibration curve, " + filename + " 9th Oct 25")
    pyplot.show()

    print(pressures)
    print(flow_rates)
    print(flow_rates_err)
          

