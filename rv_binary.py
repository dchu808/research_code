##program explore binary stars with rv data - D.Chu, 2016-11-09

import numpy as np
import pylab
import matplotlib.pyplot as plt
import asciidata
import efit5_util_final
from astropy.stats import LombScargle
import astropy.units as u

def extract_chains(mnestfile,star_num=0):
    ##extract a model from the multinest run, usually file is efit_.txt
    ##a lot of this code came from efit5_util_final.plot_param_hist
    ##this only focuses on when BH parameters are not fixed
    ##also assumes no extended mass

    ##get information from the chains file
	##came from around line 142 of efit5_util_final.plot_param_hist
    inFile = np.genfromtxt(mnestfile)
    ##weights are the first column of chains file
    weights = inFile[:,0]
    
    ##get orbital parameter values from chains file
    mass = inFile[:,2]
    xo = -inFile[:,3]
    yo = inFile[:,4]
    Vx = -inFile[:,5]
    Vy = inFile[:,6]
    Vz = inFile[:,7]
    D = inFile[:,8]
    Omega = inFile[:,9+star_num*6]
    omega = inFile[:,10+star_num*6]
    i = inFile[:,11+star_num*6]
    P = inFile[:,12+star_num*6]
    To = inFile[:,13+star_num*6]
    e = inFile[:,14+star_num*6]
    T_next = To + P
    logLikes = inFile[:,1]
    where_max = np.argmin(logLikes)
    
	##make histogram of the chains values, with weights
	
	
    
def rv_resid_file(rv_file,model_file,star):
    rv_table = asciidata.open(rv_file)
    #read in RV data
    daterv = rv_table[0].tonumpy()
    rv = rv_table[1].tonumpy()
    rverr = rv_table[2].tonumpy()
    mjdrv = rv_table[3].tonumpy()
    model_table = asciidata.open(model_file)
    ## RV from model
    date = model_table[0].tonumpy()
    vz = -model_table[6].tonumpy()

    ##calculate the residuals from the fit
    idx = np.zeros(len(rv), dtype = int)
    resid = np.zeros(len(rv))
    for i in range(len(rv)):
        minimum = (np.abs(date-daterv[i])).argmin()
        idx[i] = minimum 
    # print idx
    for i in range(len(rv)):
        resid[i] = rv[i] + vz[idx[i]]
    # print resid

    ##write output file with residals
    output = open(star+'_rv_resid.txt','w')
    # output.write("{0:>15.6}\t{1:>15.8}\t{2:>15.3}\n".format(mjdrv,resid,rverr))
    for i in range(len(rv)):
        output.write("{:.6f}  {}  {}\n".format(mjdrv[i],resid[i],rverr[i]))
    output.close()

##lomb scargle process
def lombscargle(resid_file):
    data = np.genfromtxt(resid_file, names=['rvmjd','resid','rverr'])
    # print
    # print len(data)
    mjd = np.zeros(len(data))
    resid = np.zeros(len(data))
    rverr = np.zeros(len(data))
    for i in range(len(data)):
        mjd[i] = data[i][0]
        resid[i] = data[i][1]
        rverr[i] = data[i][2]
    # print mjd
    # mjd_days = mjd * u.day
    ##maximum frequency works out to about 1000 day period
    frequency, power = LombScargle(mjd,rverr,rverr).autopower(minimum_frequency=0.001,maximum_frequency=1.)
    plt.plot(1./frequency, power)
    plt.xlabel('Period (Days)')
    plt.ylabel('Power')
    plt.show()