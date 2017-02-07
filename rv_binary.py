##program explore binary stars with rv data - D.Chu, 2016-11-09

import numpy as np
import scipy
from scipy.optimize import leastsq,curve_fit
from scipy import stats
import pylab
import matplotlib.pyplot as plt
import asciidata
import efit5_util_final
from astropy.stats import LombScargle
import astropy.units as u
from astropy import constants as const
#from u import cds
from tqdm import tqdm

def main_code(mnestfile,rv_file,star_num=0,min_freq=0.1,max_freq=1.):
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
    freq_array = np.linspace(min_freq,max_freq,10000)
    
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
    # for j in tqdm(range(10)):
        
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
        # print vz_model

        ##once the model is generated, get the residuals
        resid, red_chisq = calc_resid(daterv, rv, rverr, times, vz_model)
        red_chisq_arr[j] = red_chisq

        ##test
        # print resid
        # print red_chisq_arr
        
        ##with residuals, Lomb Scargle can be run
        power = lombscargle(mjdrv,resid,rverr,min_freq,max_freq)
        big_power_array[j] = power
        # print power
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
    np.save('weights',chain_weights)

def envelope_cdf(freqarray,powerarray,weights_array):
    ##calculate cdf
    ##weights are in the mnestfile
    
    ##get the frequencies
    # freq = np.genfromtxt(freq_file)
    ##turn the frequencies into periods, which is inverse
    # periods = 1./freq
    
    ##make array of weights, first column in chains file
    # inFile = np.genfromtxt(mnestfile)
    # weights = inFile[:,0]
    ##in the future, array of weights will be produced
    weights = np.load(weights_array)

    power_array = np.load(powerarray)
    freq_array = np.load(freqarray)

    ##create an array of median power values for each frequency
    median_array = np.zeros(len(freq_array))
    
    ##do same for +/- 1 sigma
    minus_array = np.zeros(len(freq_array))
    plus_array = np.zeros(len(freq_array))
    
    ##go through the power file one line at a time to make cdfs
    ##Each value in one row has a weight value attached to it as well
    for j in tqdm(range(len(freq_array))):
        ##each column is the power of a particular frequency. Read through columns
        col = power_array[:,j]
        ##want to take cdf of this column
        ##start take by making a histogram, weighting it by weights
        power,bin_edges = np.histogram(col,bins=10000,normed=False,weights=weights)
        # print power
        ##start cdf process, normalize
        power_norm = np.array(power, dtype=float) / power.sum()
        
        # sid = (power_norm.argsort())[::-1] # indices for a reverse sort
        sid = (power_norm.argsort())
        powerSort = power_norm[sid]
        ##sort the original power array - should be the same as powerSort, but not normalized
        # powerSort_not_norm = power[sid]
        
        ##cdf
        cdf = np.cumsum(powerSort)
        
        ##Determine points for median, +/- 1 sigma
        idxm = (np.where(cdf > 0.5))[0] #median
        idx1m = (np.where(cdf > 0.3173))[0] #1 sigma minus
        idx1p = (np.where(cdf > 0.6827))[0] #1 sigma plus
        
        median = bin_edges[idxm[0]] + 0.5*(bin_edges[1]-bin_edges[0]) ##is this last part appropriate?
        level1m = bin_edges[idx1m[0]] + 0.5*(bin_edges[1]-bin_edges[0])
        level1p = bin_edges[idx1p[0]] + 0.5*(bin_edges[1]-bin_edges[0])

        # median = powerSort[idxm[0]]
        # level1m = powerSort[idx1m[0]]
        # level1p = powerSort[idx1p[0]]
        ##Use the original power values
        # median = powerSort_not_norm[idxm]
        # level1m = powerSort_not_norm[idx1m]
        # level1p = powerSort_not_norm[idx1p]

        ##write these values to arrays
        median_array[j] = median
        minus_array[j] = level1m
        plus_array[j] = level1p
    
    np.save('median_array', median_array)
    np.save('minus_array', minus_array)
    np.save('plus_array', plus_array)

##envelope cdf but no weights. For sensativity analysis
def envelope_cdf_no_weights(freqarray,powerarray):
    ##calculate cdf
    power_array = np.load(powerarray)
    freq_array = np.load(freqarray)

    ##create an array of median power values for each frequency
    median_array = np.zeros(len(freq_array))
    
    ##do same for +/- 1 sigma
    minus_array = np.zeros(len(freq_array))
    plus_array = np.zeros(len(freq_array))
    
    ##go through the power file one line at a time to make cdfs
    ##Each value in one row has a weight value attached to it as well
    for j in tqdm(range(len(freq_array))):
        ##each column is the power of a particular frequency. Read through columns
        col = power_array[:,j]
        ##want to take cdf of this column
        ##start take by making a histogram, weighting it by weights
        power,bin_edges = np.histogram(col,bins=10000,normed=False)
        ##start cdf process, normalize
        power_norm = np.array(power, dtype=float) / power.sum()
        
        # sid = (power_norm.argsort())[::-1] # indices for a reverse sort
        sid = (power_norm.argsort())
        powerSort = power_norm[sid]
        ##sort the original power array - should be the same as powerSort, but not normalized
        # powerSort_not_norm = power[sid]
        
        ##cdf
        cdf = np.cumsum(powerSort)
        
        ##Determine points for median, +/- 1 sigma
        idxm = (np.where(cdf > 0.5))[0] #median
        idx1m = (np.where(cdf > 0.3173))[0] #1 sigma minus
        idx1p = (np.where(cdf > 0.6827))[0] #1 sigma plus
        
        median = bin_edges[idxm[0]] + 0.5*(bin_edges[1]-bin_edges[0]) ##is this last part appropriate?
        level1m = bin_edges[idx1m[0]] + 0.5*(bin_edges[1]-bin_edges[0])
        level1p = bin_edges[idx1p[0]] + 0.5*(bin_edges[1]-bin_edges[0])

        ##write these values to arrays
        median_array[j] = median
        minus_array[j] = level1m
        plus_array[j] = level1p
    
    np.save('median_array', median_array)
    np.save('minus_array', minus_array)
    np.save('plus_array', plus_array)
    
def plot_env(freqarray,median,plus_env,minus_env,noise=False):
    ##make a plot of the Lomb Scargle, plotting median power, +/- 1 sigma
    frequency = np.load(freqarray)
    median = np.load(median)
    plus = np.load(plus_env)
    minus = np.load(minus_env)
    
    # if noise == True:
        ##manually put in noise files
        # noise_dir = 
        # noise_freq = np.load('/u/devinchu/efits_binary_investigation/efit_boehle_2016/rv_binary/Sensitivity_Analysis/freq_array_sa_all.npy')
        # noise = np.load('/u/devinchu/efits_binary_investigation/efit_boehle_2016/rv_binary/Sensitivity_Analysis/median_array_sa_all.npy')
        
    ##plot the function
    # plt.semilogx(1/frequency, plus, color ='gray',alpha=.5)
    plt.semilogx(1/frequency, median, color ='black')
    # plt.semilogx(1/frequency, minus, color ='gray',alpha=.5)
    # plt.plot(1/frequency,median,alpha=0)
    plt.fill_between(1/frequency,median,plus,facecolor='yellow', alpha=0.5)
    plt.fill_between(1/frequency,minus,median,facecolor='yellow', alpha=0.5)
    if noise == True:
        noise_freq = np.load('/u/devinchu/efits_binary_investigation/efit_boehle_2016/rv_binary/Sensitivity_Analysis/freq_array_sa_all.npy')
        noise = np.load('/u/devinchu/efits_binary_investigation/efit_boehle_2016/rv_binary/Sensitivity_Analysis/median_array_sa_all.npy')
        plt.fill_between(1/noise_freq,0.,noise,facecolor='red', alpha=0.5)
    # plt.set_xscale('log')
    plt.xlabel('Period (Days)')
    plt.ylabel('Power')
    # plt.ylim(0,1.5)
    #plt.xlim(0,30)
    plt.xlim(1,2400)    
    plt.show()

    # plt.semilogx(1/frequency, median - minus, color ='black')
    # plt.show()
    
    # plt.semilogx(1/frequency, plus - median, color ='black')
    # plt.show()

def plot_env_2(freqarray_1,median_1,plus_env_1,minus_env_1,freqarray_2,median_2,plus_env_2,minus_env_2):
    ##basically combining 2 plot env to expand frequency range
    frequency_1 = np.load(freqarray_1)
    median_1 = np.load(median_1)
    plus_1 = np.load(plus_env_1)
    minus_1 = np.load(minus_env_1)

    frequency_2 = np.load(freqarray_2)
    median_2 = np.load(median_2)
    plus_2 = np.load(plus_env_2)
    minus_2 = np.load(minus_env_2)

    plt.semilogx(1/frequency_1,median_1,alpha=0)
    plt.fill_between(1/frequency_1,median_1,plus_1,facecolor='yellow', alpha=0.5)
    plt.fill_between(1/frequency_1,minus_1,median_1,facecolor='yellow', alpha=0.5)

    plt.semilogx(1/frequency_2,median_2,alpha=0)
    plt.fill_between(1/frequency_2,median_2,plus_2,facecolor='yellow', alpha=0.5)
    plt.fill_between(1/frequency_2,minus_2,median_2,facecolor='yellow', alpha=0.5)

    # plt.set_xscale('log')
    plt.xlabel('Period (Days)')
    plt.ylabel('Power')
    # plt.ylim(0,1.5)
    plt.xlim(1,2400)
    plt.show()    

def fold_curve(freqarray,median,resid_file,plots=True):
    ##plot the rv to a folded period
    frequency = np.load(freqarray)
    median = np.load(median)
    ##look for the highest value
    best_freq_ind = np.argmax(median)
    best_freq = frequency[best_freq_ind]

    ##perform a cutoff - arbitrary at the moment
    # median_cut = np.where(median > 0.5) ##arbitrary, hard coded cut-off right now
    # print median_cut[0]
    peak_freq = np.zeros(0)
    peak_ind = np.zeros(0)
    peak_med = np.zeros(0)
    for i in range(1,len(median)-1):
        if median[i] <= 0.5: ##arbitrary, hard coded cut-off right now
            continue
        ##check to see if this frequency is a local max
        if median[i] <= median[i-1]:
            continue
            
        if median[i] > median[i+1]:
            freq_output = frequency[i] ##peak frequencies
            ind = i
            peak_ind = np.append(peak_ind,ind) ##peak indices
            peak_freq = np.append(peak_freq,freq_output)
            peak_med = np.append(peak_med,median[i]) ##peak power values

    # print peak_ind
    # print peak_freq
    # print 1/peak_freq
    # print peak_med
    # print 1/best_freq
    # print best_freq

    ##sorting through the peak frequencies
    sid = (peak_med.argsort())[::-1]
    medianSort = peak_med[sid]
    print medianSort
    freqSort = peak_freq[sid]
    print 1/freqSort

    ##below is to make the folded rv curve plots
    if plots == True:
        t_fit = np.linspace(0,1)
        ##read in residual rv file to plot
        data = np.genfromtxt(resid_file)
        mjd = data[:,0]
        resid = data[:,1]
        rverr = data[:,2]

        ##make folded rv curves
        ##for each peak frequency
        for i in range(len(medianSort)):
            y_fit = LombScargle(mjd,resid,rverr).model(t_fit/freqSort[i], freqSort[i])
            ##phase the data
            phase = (mjd * freqSort[i]) % 1    

            plt.figure()
            plt.errorbar(phase,resid,rverr,fmt='o',color='black')
            plt.plot(t_fit,y_fit,color='black')
            plt.title('Period in days={0:.3f}'.format(1/freqSort[i]))
            plt.xlabel('Phase')
            plt.ylabel('Residual (km/s)')
            plt.show()
    
def CL_vmax(resid_file,freq_array):
    ##find the amplitude and phase shift values for fitting phased residual curve
    ##first need to fold the RV data to a particular frequency
    ##then fit data to S sin(w*t) + C cos(w*t) + const
    ##w = 2*pi/Period
    ##test freq 0.922322232223
    freq_array = np.load(freq_array)
    ##data from file
    data = np.genfromtxt(resid_file)
    mjd = data[:,0]
    resid = data[:,1]
    rverr = data[:,2]
    
    ##given best frequency to phase:
    # frequency = freq
    period = 1./freq_array
    # w = 2.*np.pi/(1./freq)
    w = 2. * np.pi * freq_array

    CL_array = np.zeros(len(freq_array))
    
    for i in tqdm(range(len(freq_array))):
    	##phase data to the frequency
        # phase = (mjd * freq_array[i]) % 1
        
        def variance(t,a,b,const):
            ##sine function + cos function
            z = a * np.sin(w[i]*t) + b * np.cos(w[i]*t) + const
            return z
    
        # (x1,x2) = curve_fit(variance,phase,resid,p0=(0.,0.,0.),sigma=rverr) ##this was a typo
        (x1,x2) = curve_fit(variance,mjd,resid,p0=(0.,0.,0.),sigma=rverr)
        ##best fit parameters
        # print x1
        A = x1[0]
        # print A
        B = x1[1]
        # print B
        cons = x1[2]
        # print cons
        vmax = np.sqrt(A**2 + B**2)
        # print vmax
        #print x2
        ##want uncertainties from covariance matrix
        ##uncertainty in amplitude
        A_sig =  np.sqrt(x2.diagonal().item(0))
        # print A_sig
        B_sig =  np.sqrt(x2.diagonal().item(1))
        # print B_sig
        ##covariance matrix
        cov = x2
        # print cov
        # print x2[0]
        co_a = x2[:2][0][:2]
        co_b = x2[:2][1][:2]
        co_ab = np.vstack((co_a,co_b))
        # print co_ab
        ##start proess to calculate 95% confidence level
        n = 1000 ##select number of trials
        a = np.zeros(n)
        b = np.zeros(n)
        vmax_array = np.zeros(n)
        ##draw from a multivariate gaussian
        ##uses the fit parameters as mean, then uses covariance matrix
        ##to find A and B, which will then be used to find Vmax
        for j in range(n):
            sample = np.random.multivariate_normal((A,B),co_ab)
            a[j] = sample[0]
            b[j] = sample[1]
        vmax_array = np.sqrt(a**2 + b**2)
        vmax_n, vmax_minmax, vmax_mean, vmax_var, vmax_skew, vmax_kurt = scipy.stats.describe(vmax_array)
        vmax_std = np.sqrt(vmax_var)
        ##get confidence level
        CL_vmax = stats.norm.interval(0.95,loc=vmax_mean,scale=vmax_std/np.sqrt(n))
        # print CL_vmax
        CL_array[i] = CL_vmax[1]
    np.save('conf_lev',CL_array)
    plt.figure()
    plt.plot(1./freq_array,CL_array)
    plt.show()
    
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
        resid[i] = rv[i] - vz_model[idx[i]]
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
    vz = model_table[6].tonumpy()

    ##calculate the residuals from the fit
    idx = np.zeros(len(rv), dtype = int)
    resid = np.zeros(len(rv))
    chisq = np.zeros(len(rv))
    for i in range(len(rv)):
        minimum = (np.abs(date-daterv[i])).argmin()
        idx[i] = minimum 
    # print idx
    for i in range(len(rv)):
        resid[i] = rv[i] - vz[idx[i]]
        ##show the chi-square value as well
        chisq[i] = resid[i]**2/np.abs(vz[idx[i]])

    ##sum array chisq to get the chi-squared
    ##divide by datapoints - 1 for reduced chi-squared
    red_chisq = np.sum(chisq)/(len(rv) - 1)
    print red_chisq    
    

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
    frequency, power = LombScargle(mjd,resid,rverr).autopower(minimum_frequency=0.001,maximum_frequency=1.,samples_per_peak=2.,method='fast')
    print len(frequency)
    #plt.plot(1./frequency, power)
    plt.semilogx(1./frequency, power, color='black')
    plt.xlabel('Period (Days)')
    plt.ylabel('Power')
    plt.xlim(0,10)
    plt.show()

def lombscargle(mjd,resid,rverr,min_freq,max_freq):
    ##maximum frequency of .001 works out to about 1000 day period
    ##reducing number of samples at peak to help with calculations
    # frequency, power = LombScargle(mjd,rverr,rverr).autopower(minimum_frequency=0.001,maximum_frequency=1.,samples_per_peak=2.,method='fast')
    ##doing a uniform sample of frequency
    # frequency = np.linspace(0.1,1.,10000)
    frequency = np.linspace(min_freq,max_freq,10000)
    power = LombScargle(mjd,resid,rverr).power(frequency,method='fast')
    return power

##develop sensativity analysis
##start with a non-periodic signal
def sens_analysis(rv_file,min_freq,max_freq):
    rv_table = asciidata.open(rv_file)
    #read in RV data
    daterv = rv_table[0].tonumpy()
    # rv = rv_table[1].tonumpy()
    rverr = rv_table[2].tonumpy()
    mjdrv = rv_table[3].tonumpy()
    freq_array = np.linspace(min_freq,max_freq,10000)

    ##want to run this sensativity n times
    n = 10000 ##set this manually
    big_power_array = np.zeros((n,len(freq_array)))
    for k in tqdm(range(n)):
        
        ##generate a fake residual RV curve, pick from distribution based on rv error
        fake_resid = np.zeros(len(rverr))
        for i in range(len(fake_resid)):
            ##Gaussian centered at 0, with sigma being rv error
            fake_resid[i] = np.random.normal(0,rverr[i])
        ##plot this fake residual curve, as a test
        # plt.figure()
        # plt.axhline(color='black')
        # plt.scatter(daterv, fake_resid, color = 'black')
        # plt.errorbar(daterv, fake_resid, rverr, np.zeros(len(daterv)), color='black', linestyle='None')
        # plt.show()
        ##now run the fake residual curve through a lomb scargle
        frequency = np.linspace(min_freq,max_freq,10000)
        power = lombscargle(mjdrv,fake_resid,rverr,min_freq,max_freq)
        big_power_array[k] = power
        
        ##plot lomb_scargle as a test
        # plt.figure()
        # plt.semilogx(1./frequency, power, color='black')
        # plt.xlabel('Period (Days)')
        # plt.ylabel('Power')
        # plt.show()
    np.save('power_array',big_power_array)
    np.save('freq_array',freq_array)

##simple function to append the arrays together to make it easier for plotting
##these are hard-coded for now
def array_append(dir):
    freq_1 = np.load(dir + 'freq_array_sa.npy')
    freq_2 = np.load(dir + 'freq_array_sa_100day.npy')
    freq_3 = np.load(dir + 'freq_array_sa_1000day.npy')
    freq_4 = np.load(dir + 'freq_array_sa_2400day.npy')
    
    med_1 = np.load(dir + 'median_array_sa.npy')
    med_2 = np.load(dir + 'median_array_sa_100day.npy')
    med_3 = np.load(dir + 'median_array_sa_1000day.npy')
    med_4 = np.load(dir + 'median_array_sa_2400day.npy')
    
    plus_1 = np.load(dir + 'plus_array_sa.npy')
    plus_2 = np.load(dir + 'plus_array_sa_100day.npy')
    plus_3 = np.load(dir + 'plus_array_sa_1000day.npy')
    plus_4 = np.load(dir + 'plus_array_sa_2400day.npy')
    
    minus_1 = np.load(dir + 'minus_array_sa.npy')
    minus_2 = np.load(dir + 'minus_array_sa_100day.npy')
    minus_3 = np.load(dir + 'minus_array_sa_1000day.npy')
    minus_4 = np.load(dir + 'minus_array_sa_2400day.npy')
    
    ##flip the arrays for plotting
    freq_1_flip = freq_1[::-1]
    freq_2_flip = freq_2[::-1]
    freq_3_flip = freq_3[::-1]
    freq_4_flip = freq_4[::-1]
    
    med_1_flip = med_1[::-1]
    med_2_flip = med_2[::-1]
    med_3_flip = med_3[::-1]
    med_4_flip = med_4[::-1]
    
    plus_1_flip = plus_1[::-1]
    plus_2_flip = plus_2[::-1]
    plus_3_flip = plus_3[::-1]
    plus_4_flip = plus_4[::-1]
    
    minus_1_flip = minus_1[::-1]
    minus_2_flip = minus_2[::-1]
    minus_3_flip = minus_3[::-1]
    minus_4_flip = minus_4[::-1]    
    
    ##start appending arrays
    complete_freq = np.append(freq_1_flip, [freq_2_flip, freq_3_flip, freq_4_flip])
    complete_med = np.append(med_1_flip, [med_2_flip, med_3_flip, med_4_flip])
    complete_plus = np.append(plus_1_flip, [plus_2_flip, plus_3_flip, plus_4_flip])
    complete_minus = np.append(minus_1_flip, [minus_2_flip, minus_3_flip, minus_4_flip])
    
    ##save arrays
    np.save('freq_array_sa_all', complete_freq)
    np.save('median_array_sa_all', complete_med)
    np.save('plus_array_sa_all', complete_plus)
    np.save('minus_array_sa_all', complete_minus)
    
##turn an input of period and velocity max to mass of binary
##equation is binary mass equation

##define the binary mass function

def bm_equation(m_test,p_days,vmax_kms,mass_tot):
    ##define the equation first
    ##period in days, vmax in km/s
    ##sin^3 i is not included in equation now
    ##idea is to solve for m in solar masses
    
    G = const.G
    ##mass of S0-2 comes from input
    mtot = mass_tot * u.Msun
    
    ##period in days
    p = p_days * u.d
    
    ##vmax in km/s
    vmax = vmax_kms * (u.km/u.s)
    
    ##test value for m
    m = m_test * u.Msun
    
    value = m**3 / (mtot)**2 - p * vmax**3 / (2. * np.pi * G)
    
    ##I want to return this value and minimize it
    ##.value takes out the units
    return np.abs(value.value)
    
def bm_solve(period,vmaxkms,mass_tot):
    ##sample through a mass array to find companion
    m_array = np.arange(0.1,10.1,.1)
    test_values = np.zeros(len(m_array))

    ##loop through mass values to find where the minimum value is for function
    for i in range(len(m_array)):
        test_values[i] = bm_equation(m_array[i],period,vmaxkms,mass_tot)
    # print test_values
    min_arg = np.argmin(test_values)
    # print min_arg 
    # print test_values[min_arg]
    # print m_array[min_arg]
    ##return the mass that best minimizes the function
    return m_array[min_arg]

##make plot of binary mass vs period
##requires Vmax vs period file
def mass_period_plot(file_path,mass_tot):
    ##read in the file
    info = np.genfromtxt(file_path)
    period_array = info[:,0]
    vmax_array = info[:,1]
    binary_mass_array = np.zeros(len(period_array))
    
    ##feed these array information into the bm_solve function
    for i in tqdm(range(len(period_array))):
        binary_mass_array[i] = bm_solve(period_array[i], vmax_array[i], mass_tot)
    print binary_mass_array
    ##make plot of period vs companion mass
    plt.figure()
    plt.plot(period_array,binary_mass_array)
    plt.xlabel('Period (Days)')
    plt.ylabel('Companion Mass Sin i (Solar Masses)')
    plt.show()

##if given a period, calculate the vmax, given a companion mass
##gives up a limit for excluding periods with high power from aliasing
def vmax_find(m_test,mass_tot,period):
    G = const.G
    ##total mass (S02 + companion) comes from input
    mtot = mass_tot * u.Msun
    
    ##period in days
    p = period * u.d
    
    ##test value for m companion
    m = m_test * u.Msun

    ##binary mass equation, solving for vmax
    # z = (((m**3)/((m+ms02)**2))*((2. * np.pi)/p))
    z = (m**3./(mtot)**2.)*(2*np.pi * G / p)
    # print z
    x = z**(1./3.)
    # print x
    ##show this value in km/s
    x_kms = x.to(u.km/u.s)
    return x_kms.value
    # print x_kms
    # print x_kms.value

##uses the function vmax_find, but for array of periods (1/frequencies)
def vmax_array(m_test,mass_tot,freq_array):
    freq = np.load(freq_array)
    periods = 1./freq
    vmax_array = np.zeros(len(periods))
    for i in tqdm(range(len(periods))):
        vmax_array[i] = vmax_find(m_test,mass_tot,periods[i])
    print vmax_array[:30]
        