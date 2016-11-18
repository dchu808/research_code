##program explore binary stars with rv data - D.Chu, 2016-11-09

import numpy as np
import pylab
import matplotlib.pyplot as plt
import asciidata
import efit5_util_final
from astropy.stats import LombScargle
import astropy.units as u
from tqdm import tqdm

def main_code(mnestfile,rv_file,star_num=0):
    ##extract a model from the multinest run, usually file is efit_.txt
    ##a lot of this code came from efit5_util_final.plot_param_hist
    ##this only focuses on when BH parameters are not fixed
    ##also assumes no extended mass
    
    ##open the rv_file once, to help speed of program
    ##make an array of rv data, keep it throughout loop of program
    rv_table = asciidata.open(rv_file)
    #read in RV data
    daterv = rv_table[0].tonumpy()
    rv = rv_table[1].tonumpy()
    rverr = rv_table[2].tonumpy()
    mjdrv = rv_table[3].tonumpy()

    ##get information from the chains file
    ##came from around line 142 of efit5_util_final.plot_param_hist
    inFile = np.genfromtxt(mnestfile)

    ##frequency file
    # freq_output = open('freq_file.txt','w')
    ##.txt file is too large, using np.save instead
    # freq_array = np.zeros(10000)
    freq_array = np.linspace(0.1,1.,10000)
    
    ##make power file
    # power_output = open('power_file.txt','w')
    ##.txt file is too large, using np.save instead
    big_power_array = np.zeros((len(inFile),len(freq_array)))

    ##Make an array for reduced chisquare of each model produced by chains
    red_chisq_arr = np.zeros(len(inFile))

	##weights array
	chain_weights = np.zeros(len(inFile))

    ##start looping through each chain in the chains file
    for j in tqdm(range(len(inFile))):
    ##as a test, shorten the loop
    # for j in tqdm(range(20)):
        
        params = np.zeros(13)
        ##weights are the first column of chains file
        chain_weights[j] = inFile[j,0]
    
        ##get orbital parameter values from chains file
        params[0] = inFile[j,2]        ##mass
        params[1] = -inFile[j,3]     ##xo
        params[2] = inFile[j,4]            ##yo
        params[3] = -inFile[j,5]            ##Vx
        params[4] = inFile[j,6]            ##Vy
        params[5] = inFile[j,7]            ##Vz
        params[6] = inFile[j,8]                ##D
        params[7] = inFile[j,9+star_num*6]        ##Omega
        params[8] = inFile[j,10+star_num*6]        ##omega
        params[9] = inFile[j,11+star_num*6]            ##i
        params[10] = inFile[j,12+star_num*6]            ##P
        params[11] = inFile[j,13+star_num*6]        ##To
        params[12] = inFile[j,14+star_num*6]            ##e
        # T_next = To + P
        # logLikes = inFile[j,1]
        # where_max = np.argmin(logLikes)

        ##with these parameters, can generate a model from them
        ##only produces array of times and vz_model 
        times, vz_model = make_model(params)
        
        ##once the model is generated, get the residuals
        resid, red_chisq = calc_resid(daterv, rv, rverr, times, vz_model)
        red_chisq_arr[j] = red_chisq
        
        ##with residuals, Lomb Scargle can be run
        power = lombscargle(mjdrv,resid,rverr)
        big_power_array[j] = power
        # big_power_array = np.append(big_power_array,power,axis=0)

        #print power
        ##write to the power file
        # for p in power:
        #     power_output.write("{} ".format(p))
        # # power_output.write(power)
        # power_output.write("\n")
        
    ##make frequency file
    # freq_output.write(frequency)
    # power_output.close()
    # freq_output.close()

    ##append frequency to freq array
    # freq_array = np.append(freq_array,frequency)

    ##save using numpy.save, faster than txt files
    np.save('power_array',big_power_array)
    np.save('freq_array',freq_array)
    np.save('chi_squares',red_chisq_arr)

def envelope_cdf(freqarray,powerarray,mnestfile):
    ##calculate cdf
    ##weights are in the mnestfile
    
    ##get the frequencies
    # freq = np.genfromtxt(freq_file)
    ##turn the frequencies into periods, which is inverse
    # periods = 1./freq
    
    ##make array of weights, first column in chains file
    inFile = np.genfromtxt(mnestfile)
    weights = inFile[:,0]
	##in the future, array of weights will be produced
	# weights = np.load(weights_array)

    power_array = np.load(powerarray)
	freq_array = np.load(freqarray)

    ##go through the power file one line at a time to make cdfs
    ##Each value in one row has a weight value attached to it as well
    for j in range(len(power_file)):
        # power_array = [j]
    ##tests
    # print powerfile[0]
    # print powerfile[0][0]
    # print weights[0]
    
def make_model(orbit_params,tmin=1995.0,tmax=2018.0,increment=0.005):
    ##make model from orbital parameters
    ##working from make_model_orbitparams in efit5_util_final, but modifying it to not output file
    
    times = np.linspace(tmin, tmax, ((tmax - tmin)/increment) + 1.)
    vz_model = np.zeros(len(times))
    for j in range(len(times)):
        t = times[j]
        # MAP parameters
        elem = np.zeros(8)
        elem[0] = orbit_params[6]  # Distance
        elem[2] = orbit_params[10]  # Period
        elem[3] = orbit_params[12] # Eccentricity
        elem[4] = orbit_params[11] #t0
        elem[5] = orbit_params[8]  #w
        elem[6] = orbit_params[9]  #i
        elem[7] = orbit_params[7]  #Omega
        
        # Get a from M and Period
        mass = orbit_params[0]
        # Add drift
        xo = orbit_params[1]
        yo = orbit_params[2]
        Vxo = orbit_params[3]
        Vyo = orbit_params[4]
        Vzo = orbit_params[5]
        drift_params = (xo, yo, Vxo, Vyo, Vzo)
        
        x, y, z, vx, vy, vz, v = efit5_util_final.get_orbit_prediction(elem, t, mass, drift_params)
        
        ##instead of writing file, just make array of vz (that's all we're interested in)
        vz_model[j] = vz
        
    ##goal is to output 2 arrays: 1 time array, 1 array of vz_model
    return times, vz_model

def open_rv_file(rv_file):
    ##open the rv_file once, to help speed of program
    ##make an array of rv data, keep it throughout loop of program
    rv_table = asciidata.open(rv_file)
    #read in RV data
    daterv = rv_table[0].tonumpy()
    rv = rv_table[1].tonumpy()
    rverr = rv_table[2].tonumpy()
    mjdrv = rv_table[3].tonumpy()

    return daterv, rv, rverr, mjdrv

def calc_resid(daterv, rv, rverr, times, vz_model):
    ##calculate the residuals from rv file and model generated from chains
    # times, vz_model = make_model(orbit_params)
    
    ##calculate the residuals from the fit
    idx = np.zeros(len(rv), dtype = int)
    resid = np.zeros(len(rv))
    chisq = np.zeros(len(rv))
    for i in range(len(rv)):
        minimum = (np.abs(times-daterv[i])).argmin()
        idx[i] = minimum 
    # print idx
    for i in range(len(rv)):
        resid[i] = rv[i] + vz_model[idx[i]]
        ##calculate chisquare
        chisq[i] = resid[i]**2/np.abs(vz_model[idx[i]])
    ##sum array chisq to get the chi-squared
    ##divide by datapoints - 1 for reduced chi-squared
    red_chisq = np.sum(chisq)/(len(rv) - 1) 
    return resid, red_chisq

def make_rv_resid_file(rv_file,model_file,star):
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
def lombscargle_file(resid_file):
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
    frequency, power = LombScargle(mjd,rverr,rverr).autopower(minimum_frequency=0.001,maximum_frequency=1.,samples_per_peak=2.,method='fast')
    print len(frequency)
    #plt.plot(1./frequency, power)
    plt.semilogx(1./frequency, power)
    plt.xlabel('Period (Days)')
    plt.ylabel('Power')
    #plt.xlim(0,30)
    plt.show()

def lombscargle(mjd,resid,rverr):
    ##maximum frequency works out to about 1000 day period
    ##reducing number of samples at peak to help with calculations
    # frequency, power = LombScargle(mjd,rverr,rverr).autopower(minimum_frequency=0.001,maximum_frequency=1.,samples_per_peak=2.,method='fast')
    ##doing a uniform sample of frequency between 1 day
    frequency = np.linspace(0.1,1.,10000)
    power = LombScargle(mjd,rverr,rverr).power(frequency,method='fast')
    return power