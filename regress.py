#!/usr/bin/python

import numpy as np
import random as rand

class LinearRegress(object):
	def __init__(self):
		self.theta = 0 #weights factor

	def train(self, train_dat, steps):
		self.Y = train_dat[:, -1:] #last column is treated as output set
		self.X = np.ones( train_dat.shape ) #include bias
		self.theta = np.random.rand( train_dat.shape[1], 1 ) #predictors
		self.X[:, :-1] = train_dat[:, :-1] #save training
		self.gradientDescent(steps) #start descent

	def gradientDescent(self, steps):
		'minimize error predictors and expected values; descent uses adaptive alpha'
		prev_err = 1 + max(self.Y) #set expected erros some large value
		cost_hst = []
		alpha = 0.01 #changes as
		#descent constants
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
			cost_hst.append( self.computeCost(m, err) )
			#print error, '|', prev_err;			
			if cost_hst[-1:] < prev_err:
				print ">>>updating"
				prev_err = cost_hst[-1:]
				alpha = alpha * rho
				self.theta = t_theta #save optimized thetas
			else:
				alpha = alpha * sig #use curing sigma to fix learning rate
				t_theta = self.theta
		return cost_hst #to see how gradient descent works against dataset

	def computeCost(self, m, err):
		'compute least square error between expected and predicted value'
		return (1.0 /(2 * m)) / sum( pow( err, 2) )

	def predict(self, dat):
		'predict outcome for a given datum'
		if type(self.theta) == int:
			print '\n Please train predictor first!'
			return None
		else:
			datum = np.ones( (1, self.theta.shape[0]) )
			datum[:, 1:] = dat
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
	while i < mxsz:
		#descretize negative values with sigmoid so regression model can predict
		val = dat[i][1] - dat[i+1][1]
		if val < 0:
			ris[ dat[i][0] ] = [1, abs(val) ]
		else:
			ris[ dat[i][0] ] = [2, abs(val) ]
		i += 1
	ris[ dat[mxsz][0] ] = [2, 1]
	return ris

def normalize(dat):
	#compute each column's mean and standard derivation
	for i in range( dat.shape[1] ):
		dat[:, i] -= np.mean( dat[:, i] )
		dat[:, i] /= np.std( dat[:, i] )
	return dat

################# SPECIFIC FOR PROBLEM CHALLENGE

def main():
	print "\t>>> reading rows.."
	rows = int( raw_input() )
	print "\t>>> reading DATA.."
	data = readData(rows)
	#try to increase feature space by identifying high travel months
	print "\t>>> expanding feature space.."
	dicrises = getRises(data)

	#generate learner's training data as numpy data structure
	print "\t>>> Prepping training data.."
	tmpris, posprob = [], 0.0
	for i in range(rows):
		if dicrises[ data[i][0] ][:-1][0] == 2:
			posprob += 1
		tmpris.append(
		 	[i%12+1] #month
		 	+ dicrises[data[i][0]] #increasing/decreasing bit and value
		 	+ [data[i][1]] #number of passangers
		 )
	posprob /= rows #probability that there is a possitive increase in flights
	training_data = np.array(tmpris)
	print "\t>>> Creating learners.."
	predictor = LinearRegress()
	#train
	itrstps = 400
	print "\t>>> Training learners.."
	predictor.train(training_data, itrstps)
	print "\t>>> Making Predictions.."

	# for i in range(12):
	# 	datum = []
	# 	if rand.uniform(0,1) < posprob:
	# 		datum = [ risesregre.predict( [i%12, 1] ), peeksregre.predict(i%12) ]
	# 	else:
	# 		datum = [ risesregre.predict( [i%12, 2] ), peeksregre.predict(i%12) ]
	# 	# print predictor.predict( datum )
	# 	print datum
	return predictor, posprob



# if __name__ == '__main__':
# 	main()