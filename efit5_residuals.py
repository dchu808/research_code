##Devin Chu
##2016-08-16
##Caluclate the residuals from an efit5 fit
##In this case, we are now not using the efit2summary step

##Step 1 - open the orbit.star file to determine number of stars used
## also, need to determine which bh_parameters are fixed

def make_resid(orbit_file_path):
	orbit_file = open(orbit_file_path,'r')
	##dictionary for black hole parameters
	bh_parameters = {}
	##dictionary of stars used in fit
	##later on in the code, a secondary dictionary will contain stellar parameters
	objects = {}
	isobject = False
	current_object = ""
	
	##read through the file and determine which lines correspond to stars
	##this will be used to determine number of stars and black hole parameters
	for line in orbit_file:
		##looking for stars in the orbit file
		words_space = line.strip().split(" ")
		if words_space[0] == 'object':
			isobject = True
			##the next line removed the quirk of having double quotes around star name
			current_object = words_space[1].replace('"','')
			##record name of star in objects dictionary
			objects[current_object] = {}
			continue
		
		##Going through the lines that are associated with the stars in the fit
		##this will create another dictionary for that particular star's parameters	
		if isobject == True:
			words = line.strip().split(" = ")
			##dealing with the } at the end of the object
			if words[0] == '}':
				continue
			##making the dictionary of the star's orbital parameters
			objects[current_object][words[0]]=words[1]
		
		
		##make the dictionary of only BH bh_parameters
		##These bh_parameters are not within the object brackets	
		if isobject == False:	
			words = line.strip().split(" = ")
			##making dictionary for bh_parameters and values
			bh_parameters[words[0]]=words[1]
		
	# print bh_parameters
	# print objects
	# print objects['S0-2']
	# print objects['S0-2']['period']
	
	##determine number of stars in fit
	# print len(objects)
	n_stars = len(objects)
		 