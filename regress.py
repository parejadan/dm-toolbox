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

def getpeeks(dat, dic):
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


################# SPECIFIC FOR PROBLEM CHALLENGE
def main():
	rows = int( raw_input() )
	rawdat = readData(rows)
	#try to increase feature space by identifying high travel months
	dicpeeks = dict() #descretized peeks to then used for increasing data feature space
	peeks = [ getpeeks(rawdat, dicpeeks) ]
	while len(peeks[-1]) > round(rows/12): #try to determine the holiday months in dataset
		peeks.append( getpeeks( peeks[-1] ) )
		
	#create numpy array for faster processing

if __name__ == '__main__':
	main()