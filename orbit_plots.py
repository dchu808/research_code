##program to make different orbit plots from efit5 - D.Chu, 2016-09-29

import numpy as np
import matplotlib.pyplot as plt
import asciidata

##first plot - compare 2 points files with 1 orbit model
##DOES NOT USE ORBIT ENVELOPES RIGHT NOW

def get_astr_data(star,dir):
    # model_table = asciidata.open(dir + 'orbit.'+star+'.model')
	points_table = asciidata.open(dir + star+'.points')
	#read in astrometric data
	datedat = points_table[0].tonumpy()
	xdat = points_table[1].tonumpy()
	ydat = points_table[2].tonumpy()
	xerrdat = points_table[3].tonumpy()
	yerrdat = points_table[4].tonumpy()
	#read in orbital models:
    # date = model_table[0].tonumpy()
    # x = model_table[1].tonumpy()
    # y = model_table[2].tonumpy()
    # z = model_table[3].tonumpy()
    # vx = model_table[4].tonumpy()
    # vy = model_table[5].tonumpy()
	minx = np.min(xdat)
	maxx = np.max(xdat)
	miny = np.min(ydat)
	maxy = np.max(ydat)
	
    # return datedat, xdat, ydat, xerrdat, yerrdat, date, x, y, z, vx, vy, minx, maxx, miny, maxy
    return datedat, xdat, ydat, xerrdat, yerrdat, minx, maxx, miny, maxy
    
##want separate function to pull out astrometric model

def get_astr_model(star,dir):
	model_table = asciidata.open(dir + 'orbit.'+star+'.model')
	#read in orbital models:
	date = model_table[0].tonumpy()
	x = model_table[1].tonumpy()
	y = model_table[2].tonumpy()
	z = model_table[3].tonumpy()
	vx = model_table[4].tonumpy()
	vy = model_table[5].tonumpy()
	minx = np.min(xdat)
	maxx = np.max(xdat)
	miny = np.min(ydat)
	maxy = np.max(ydat)
    
    return date, x, y, z, vx, vy
    
def get_rv_data(star,dir):
    # model_table = asciidata.open(dir + 'orbit.'+star+'.model')
	rv_table = asciidata.open(dir + star + '.rv')
    # ## RV from model
    # vz = -model_table[6].tonumpy()
	#read in RV data
	daterv = rv_table[0].tonumpy()
	rv = rv_table[1].tonumpy()
	rverr = rv_table[2].tonumpy()
	
    # return vz, daterv, rv, rverr
    return daterv, rv, rverr

##want separate function to pull out astrometric model	
def get_rv_model()
    model_table = asciidata.open(dir + 'orbit.'+star+'.model')
	## RV from model
	vz = -model_table[6].tonumpy()
    return vz
    
##make plots
##potentially call multiple points files from multiple align directories
##only one model is plotted. Make sure that is the first directory
def make_plots(stars,dirs,include_rv=True):
    ##will only take model from the first directory and star
	date, x, y, z, vx, vy = get_astr_model(stars[0],dirs[0])
    if include_rv==True:
        vz = get_rv_model(stars[0],dirs[0])
    
    plt.figure(figsize = (16,10))
    ##some parts of this is hardcoded - choose the order of colors, for instance
    colors = ['black','green','blue']
    
    ##plot the different points files that want to be included
	for i in range(len(stars)):
		star = stars[i]
		dir = dirs[i]
        ##will pull out data information from list of stars/directories
        # datedat, xdat, ydat, xerrdat, yerrdat, date, x, y, z, vx, vy, minx, maxx, miny, maxy = get_astr_data(star,dir)
        datedat, xdat, ydat, xerrdat, yerrdat, minx, maxx, miny, maxy = get_astr_data(star,dir)
        
		if include_rv==True:
            # vz, daterv, rv, rverr = get_rv_data(star,dir)
            daterv, rv, rverr = get_rv_data(star,dir)
        
        
        
	
	    # plt.figure(figsize = (16,10))
	    ##joint figure
	    ##x vs time
	    plt.subplot(331)
	    #plt.figure(figsize = (5,5))
	    #plt.clf()
	    plt.subplots_adjust(wspace=0.25, right = 0.9, left = 0.1, top = 0.95, bottom = 0.1)
	    ##to follow convenction of RA
	    plt.plot(date, -x, color = colors[i])
	    plt.scatter(datedat, -xdat, color = colors[i])
	    plt.errorbar(datedat, -xdat, xerrdat, np.zeros(len(datedat)), color = colors[i], linestyle = 'None')
	    plt.xlabel('Date (years)')
	    plt.ylabel('RA Offset (arcsec)')
	    plt.xlim([1995, 2020])
	
	    ##y vs time
	    plt.subplot(332)
	    plt.subplots_adjust(wspace=0.25, right = 0.9, left = 0.1, top = 0.95, bottom = 0.1)
	    plt.plot(date, y, color = colors[i])
	    plt.scatter(datedat, ydat, color = colors[i])
	    plt.errorbar(datedat, ydat, yerrdat, np.zeros(len(datedat)), color = colors[i], linestyle = 'None')
	
	    ##plot x vs t residual
	    #plt.figure(figsize = (5,5))
	    #plt.clf()
	    plt.subplot(334)
	    plt.subplots_adjust(wspace=0.25, right = 0.9, left = 0.1, top = 0.95, bottom = 0.1)
	    idx_2 = np.zeros(len(xdat), dtype = int)
	    for i in range(len(xdat)):
	    	minimum_2 = (np.abs(date-datedat[i])).argmin()
	    	idx_2[i] = minimum_2
	    plt.scatter(datedat, -xdat + x[idx_2], color = colors[i])
	    plt.errorbar(datedat, -xdat + x[idx_2], xerrdat, np.zeros(len(datedat)), color = colors[i], linestyle = 'None')
	    plt.ylabel('RA Residual (arcsec)')
	    plt.xlim([1995, 2020])
	
	    ##plot y vs t residual
	    #plt.figure(figsize = (5,5))
	    #plt.clf()
	    plt.subplot(335)
	    plt.subplots_adjust(wspace=0.25, right = 0.9, left = 0.1, top = 0.95, bottom = 0.1)
	    idx_2 = np.zeros(len(ydat), dtype = int)
	    for i in range(len(ydat)):
	    	minimum_2 = (np.abs(date-datedat[i])).argmin()
	    	idx_2[i] = minimum_2
	    plt.scatter(datedat, ydat - y[idx_2], color = colors[i])
	    plt.errorbar(datedat, ydat - y[idx_2], yerrdat, np.zeros(len(datedat)), color = colors[i], linestyle = 'None')
	    plt.xlabel('Date (years)')
	    plt.ylabel('Dec Residual (arcsec)')
	    plt.xlim([1995, 2020])
	
	    ##plot the orbit in x and y
	    plt.subplot(338)
	    plt.subplots_adjust(wspace=0.25, right = 0.9, left = 0.1, top = 0.95, bottom = 0.1)
	    plt.plot(-x, y, color = colors[i])
	    plt.scatter(-xdat, ydat, color = colors[i])
	    plt.errorbar(-xdat, ydat, yerrdat, xerrdat, color = colors[i], linestyle = 'None')
	    plt.xlabel("Offset from Sgr A* (arcsec)")
	    plt.ylabel("Offset from Sgr A* (arcsec)")
	    plt.xlim(-np.array([min(x)-.05, max(x)+.05]))
	    plt.ylim(np.array([min(y)-.05, max(y)+.05]))
	
	    if include_rv==True:
	    	##make RV vs time
	    	plt.subplot(333)
	    	plt.subplots_adjust(wspace=0.25, right = 0.9, left = 0.1, top = 0.95, bottom = 0.1)
	    	plt.plot(date, vz, color = colors[i])
	    	plt.scatter(daterv, rv, color = colors[i])
	    	plt.errorbar(daterv, rv, rverr, np.zeros(len(daterv)), color = colors[i], linestyle = 'None')
	    	plt.xlabel('Date (years)')
	    	plt.ylabel('RV (km/sec)')
	    	plt.xlim([1995, 2020])
		
		    ##make rv vs time residual
		    plt.subplot(336)
		    plt.subplots_adjust(wspace=0.25, right = 0.9, left = 0.1, top = 0.95, bottom = 0.1)
		    idx_2 = np.zeros(len(rv), dtype = int)
		    for i in range(len(rv)):
		    	minimum_2 = (np.abs(date-daterv[i])).argmin()
		    	idx_2[i] = minimum_2
		    plt.scatter(daterv, rv + vz[idx_2], color = colors[i])
		    plt.errorbar(daterv, rv + vz[idx_2], rverr, np.zeros(len(daterv)), color = colors[i], linestyle = 'None')
		    plt.xlabel('Date (years)')
		    plt.ylabel('Dec Residual (arcsec)')
		    plt.xlim([1995, 2020])
		
	