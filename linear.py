#!/usr/bin/python

import numpy as np
import random as rand

class Regression(object):
	def __init__(self):
		self.theta = 0 #weights factor

	def fit(self, X_train, y_train, steps, norm=None):
		self.Y = y_train
		self.norm = norm
		if not (self.norm == None): #normalize training features with given normalization function
			X_train, self.meus, self.stds = self.norm(X_train)
		self.X = np.ones( (X_train.shape[0], X_train.shape[1]+1) )
		self.X[:, :-1] = X_train
		#create predictor coefficients
		self.theta = np.random.rand( self.X.shape[1], 1 )
		self.gradientDescent( steps )


	def gradientDescent(self, steps):
		'minimize error predictors and expected values; descent uses adaptive alpha'
		prev_err = 1 + max(self.Y)[0] #set expected erros some large value
		cost_hst = [prev_err]
		alpha = 0.01 #changes as
		rho = 1.1 #how much to increase learning rate
		sig = 0.5 #how much to backstep learning rate
		#don't update original theta incase a backstep is required during descent
		t_theta = self.theta
		m = self.X.shape[0] #number of training examples
		#start gradient descent
		for i in range(steps):
			err = self.X.dot(t_theta)-self.Y #create hypothesis and compute error
			t_theta -= alpha * (1.0 / m) * self.X.T.dot(err)
			#alpha is increased while errors are minimized, otherwise backstep and continue
			cost_hst.append( self.computeCost(m, err)[0] )
			if cost_hst[-1:][0] < prev_err:
				prev_err = cost_hst[-1:][0]
				alpha = alpha * rho
				self.theta = t_theta #save optimized thetas
			elif cost_hst[-1:][0] >= prev_err*1.3:
				break
			else:
				alpha = alpha * sig #use curing sigma to fix learning rate
				t_theta = self.theta
		return cost_hst #to see how gradient descent works against dataset

	def computeCost(self, m, err):
		'compute least square error between expected and predicted value'
		sqerr = pow( err, 2)
		#print sqerr[:]
		sqerrsm = sum( sqerr )
		return (1.0 /(2 * m)) / sqerrsm

	def predict(self, dat):
		'predict outcome for a given datum'
		if type(self.theta) == int:
			#print '\n Please train predictor first!'
			return None
		elif not (self.norm == None): #if data was normalized
			dat = self.norm(dat, self.meus, self.stds)
		
		datum = np.ones( (1, self.theta.shape[0]) )
		datum[:,:-1] = dat
		return datum.dot( self.theta)[0][0] #return prediction

def readData(rows):
	dat = []
	for i in range(rows):
		tmp = raw_input().split()
		tmp[1] = int(tmp[1])
		dat.append(tmp)
	return dat

def getRises(dat):
	ris = dict()
	i = 0
	mxsz = len(dat)-1
	negcnt = 0.0
	while i < mxsz:
		#descretize negative values with sigmoid so regression model can predict
		val = dat[i][1] - dat[i+1][1]
		if val < 0:
			negcnt += 1
			ris[ dat[i][0] ] = [ ( negcnt/ (mxsz+1) ), abs(val) ]
		else:
			ris[ dat[i][0] ] = [ 1-( negcnt/ (mxsz+1) ), abs(val) ]
		i += 1
	ris[ dat[mxsz][0] ] = [2, 1]
	return ris

def normalize(dat, meus=None, stds=None):
	if not (meus == None):
		return (dat - [meus] ) / [stds] + 3

	meus, stds = [], []
	#compute each column's mean and standard derivation
	for i in range( dat.shape[1] ):
		meus.append( np.mean( dat[:, i] ) )
		stds.append( np.std( dat[:, i] ) )
	return ( dat - [meus] ) / [stds], meus, stds


################# SPECIFIC FOR PROBLEM CHALLENGE

def main():
	rows = int( raw_input() )
	#print "\t>>> reading DATA.."
	data = readData(rows)
	#print "\t>>> expanding feature space.."
	dicrises = getRises(data) #increase feature space by identifying increases of travellers
	
	#print "\t>>> Prepping training data.."
	tmpris, posprob = [], 0.0
	for i in range(rows):
		if dicrises[ data[i][0] ][:-1][0] == 2: #keep track of possitive bits to compute probability
			posprob += 1
		#mnth, neg/pos bit, bit pair val, num of pass
		tmpris.append( [i%12+1] + dicrises[data[i][0]] + [data[i][1]] )
	posprob /= rows #probability that there is a possitive increase in flights
	training_data = np.array(tmpris)
	train_y = training_data[:, -1:]
	train_X = training_data[:, :-1]
	trn_y_r = training_data[:, -2:-1]
	trn_X_r = training_data[:, :-2]

	#print "\t>>> Creating learners"
	itrstps = 400
	r_lrnr, lrnr = Regression(), Regression()
	#print "\t>>> Training learners.."
	# r_lrnr.fit( trn_X_r, trn_y_r, itrstps, normalize)
	# lrnr.fit( train_X, train_y, itrstps, normalize)
	lrnr.fit( trn_X_r, trn_y_r, itrstps)
	avg = sum(train_y)/rows
	#print "\t>>> Making Predictions.."
	for i in range(12):
		datum = 0
		bit = 2
		if rand.uniform(0,1) < posprob:
			bit = 1
		# datum = np.array([ [ i%12+1, bit, r_lrnr.predict( np.array([ [i%12+1, bit] ]) ) ] ])
		datum = np.array([ [ i%12+1, bit ] ])
		print int( lrnr.predict( datum )+avg )

if __name__ == '__main__':
	main()