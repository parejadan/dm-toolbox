#!/usr/bin/python

import numpy as np
import random as rand

class LinearRegress(object):
	def __init__(self):
		self.theta = 0 #weights factor

	def train(self, steps, train_dat, init_ran):
		self.Y = train_dat[:, -1:] #last column is treated as output set
		#determine where in training to begin, at zero, or some random location
		if init_ran == 'r':
			self.X = np.random.rand( train_dat.shape ) #include bias
			self.theta = np.random.rand( shape=( train_dat.shape[1], 1 ) ) #feature space
		else:
			self.X = np.zeros( train_dat.shape ) #include bias
			self.theta = np.ones( shape=( train_dat.shape[1], 1 ) ) #feature space
		#save training
		self.X[:, :-1] = train_dat[:, :-1]
		#start descent
		self.gradientDescent(steps)

	def gradientDescent(self, steps):
		'minimize error predictors and expected values; descent uses adaptive alpha'
		cur_err = 1 + max(self.Y) #set expected erros some large value
		x_trans = self.X.T
		alpha = 0.01 #changes as
		#descent constants
		rho = 1.1 #how much to increase learning rate
		sig = 0.5 #how much to backstep learning rate
		#don't update original theta incase a backstep is required during descent
		t_theta = self.theta
		m = self.X.shape[0] #number of training examples
		#start gradient descent
		for i in range(steps):
			hypo = self.X.dot(t_theta)
			t_theta -= alpha * (1.0 / m) * (x_trans.dot(hypo - self.Y))
			#alpha is increased while errors are minimized, otherwise backstep and continue
			error = self.computeCost(m, t_theta)
			#print error, '|', cur_err;			
			if error < cur_err:
				cur_err = error
				alpha = alpha * rho
				self.theta = t_theta #save optimized thetas
			else:
				alpha = alpha * sig #use curing sigma to fix learning rate
				t_theta = self.theta

	def computeCost(self, m, theta):
		'compute least square error between expected and predicted value'
		hyp = self.X.dot(theta)
		sq_err = hyp - self.Y
		print sq_err, '\n'
		return (1.0 /(2 * m)) / sum(sq_err)

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

################# SPECIFIC FOR PROBLEM CHALLENGE

def main():
	print "\t>>> reading rows.."
	rows = int( raw_input() )
	print "\t>>> reading DATA.."
	rawdat = readData(rows)
	#try to increase feature space by identifying high travel months
	print "\t>>> expanding feature space.."
	dicrises = getRises(rawdat)

	#generate learner's training data as numpy data structure
	print "\t>>> Prepping training data.."
	tmpris, posprob = [], 0.0
	for i in range(rows):
		if dicrises[ rawdat[i][0] ][:-1][0] == 2:
			posprob += 1
		tmpris.append( [
		 	i%12+1, #month
		 	dicrises[ rawdat[i][0] ][0], # if increases or decreases bit
		 	dicrises[ rawdat[i][0] ][1], # value it changes
		 	rawdat[i][1] #number of passangers
		 ] )
	posprob /= rows #probability that there is a possitive increase in flights
	training_data = np.array(tmpris)
	print "\t>>> Creating learners.."
	predictor = LinearRegress()
	#train
	itrstps = 400
	print "\t>>> Training learners.."
	predictor.train(itrstps, training_data, 'z')
	print "\t>>> Making Predictions.."

	# for i in range(12):
	# 	datum = []
	# 	if rand.uniform(0,1) < posprob:
	# 		datum = [ risesregre.predict( [i%12, 1] ), peeksregre.predict(i%12) ]
	# 	else:
	# 		datum = [ risesregre.predict( [i%12, 2] ), peeksregre.predict(i%12) ]
	# 	# print predictor.predict( datum )
	# 	print datum



if __name__ == '__main__':
	main()