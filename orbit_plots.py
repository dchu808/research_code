##program to make different orbit plots from efit5 - D.Chu, 2016-09-29

import numpy as np
import matplotlib.pyplot as plt
import asciidata

##first plot - compare 2 points files with 1 orbit model
##DOES NOT USE ORBIT ENVELOPES RIGHT NOW

def get_astr_data(star,dir):
	model_table = asciidata.open(dir + 'orbit.'+star+'.model')
	points_table = asciidata.open(dir + star+'.points')
	#read in astrometric data
	datedat = points_table[0].tonumpy()
	xdat = points_table[1].tonumpy()
	ydat = points_table[2].tonumpy()
	xerrdat = points_table[3].tonumpy()
	yerrdat = points_table[4].tonumpy()
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
	
	return datedat, xdat, ydat, xerrdat, yerrdat, date, x, y, z, vx, vy, minx, maxx, miny, maxy
	
def get_rv_data(star,dir):
	rv_table = asciidata.open(dir + star + '.rv')
	## RV from model
	vz = -model_table[6].tonumpy()
	#read in RV data
	daterv = rv_table[0].tonumpy()
	rv = rv_table[1].tonumpy()
	rverr = rv_table[2].tonumpy()
	
	return vz, daterv, rv, rverr
	
	
	