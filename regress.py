#!/usr/bin/python

import numpy as np

class LinearRegress(object):
	def __init__(self):
		self.theta = 0 #weights factor

	def train(self, steps, train_dat, init_ran):
		self.Y = train_dat[:, 1:] #last column is treated as output set
		#determine where in training to begin, at zero, or some random location
		if init_ran == 'r':
			self.X = np.random.rand( train_dat.shape ) #include bias
			self.theta = np.random.rand( shape=( train_dat.shape[1]+1, 1 ) ) #feature space
		else:
			self.X = np.zeros( train_dat.shape ) #include bias
			self.theta = np.ones( shape=( train_dat.shape[1]+1, 1 ) ) #feature space
		#save training
		self.X[:, :-1] = train_dat[:, :-1]
		#start descent
		self.gradientDescent(self, steps)

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
			if error < cur_err[0]:
				cur_err = error
				alpha = alpha * rho
				self.theta = t_theta #save optimized thetas
			else:
				alpha = alpha * sig #use curing sigma to fix learning rate
				t_theta = self.theta

	def computeCost(self, m, theta):
		'compute least square error between expected and predicted value'
		hyp = self.X.dot(theta)
		sq_err = (hyp - self.Y)**2

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

def getPeeks(dat, dic):
	mxs = []
	sz = len(dat)
	i = 0
	mxSZ = len(mxs)
	while i < sz:
		if i == 0 and dat[i][1] > dat[i+1][1]:
			mxs.append(dat[i])
		elif i == sz-1 and dat[i][1] > dat[i-1][1]:
			mxs.append(dat[i])
		elif dat[i-1][1] <= dat[i][1] and dat[i][1] >= dat[i+1][1]:
			mxs.append(dat[i])
		#descretize peeks hiarchy to dictonary, the higher the number the bigger the value 
		if mxSZ != len(mxs):
			mxSZ = len(mxs)
			if dat[i][0] in dic:
				dic[ dat[i][0] ] += 1
			else:
				dic[ dat[i][0] ] = 1
		i += 1
	return mxs

def getRises(dat):
	ris = dict()
	i = 1
	mxsz = len(dat)
	ris[ dat[0][0] ] = 0
	while i < mxsz:
		#if possotive then increase; otherwise decrease
		ris[ dat[i][0] ] = dat[i][1] - dat[i-1][1]
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
	dicpeeks = dict() #descretized peeks to then used for increasing data feature space
	peeks = [ getPeeks(rawdat, dicpeeks) ]
	while len(peeks[-1]) > round(rows/12): #try to determine the holiday months in dataset
		peeks.append( getPeeks( peeks[-1], dicpeeks ) )

	print "\t>>> Allocating space for training data.."
	training_data = np.zeros( (rows, 3) ) #rises, peeks, total passangers
	peek_dim = np.zeros( (rows, 2) ) #training data for peeks predictor
	rise_dim = np.zeros( (rows, 2) ) #training data for rises predictor
	i = 0
	#generate learner's training data as numpy data structure
	print "\t>>> Prepping training data.."
	while i < rows:
		training_data[i] = [ dicrises[ rawdat[i][0] ], dicpeeks[ rawdat[i][0] ], rawdat[i][1] ]
		rise_dim[i] = [i%12, dicrises[ rawdat[i][0] ] ]
		peek_dim[i] = [i%12, dicpeeks[ rawdat[i][0] ] ]
		i += 1

	#get learners, two for expanding feature space the other for the predictor
	print "\t>>> Creating learners.."
	risesregre = LinearRegress()
	peeksregre = LinearRegress()
	predictor = LinearRegress()
	#train
	itrstps = 400
	print "\t>>> Training learners.."
	risesregre.train(itrstps, rise_dim, 'z')
	peeksregre.train(itrstps, peek_dim, 'z')
	predictor.train(itrstps, training_data, 'z')

	print "\t>>> Making Predictions.."
	while k in range(12):
		print predictor.predict( [risesregre.predict(i%12), peeksregre.predict(i%12) ] )
		i += 1



if __name__ == '__main__':
	main()