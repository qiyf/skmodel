import numpy as np
import pickle

with open('../param/params.pkl','rb') as f:
	param = pickle.load(f)
	numSpin = param['numSpin']
	beta = param['beta']

np.random.seed(1234)
h = np.zeros(numSpin)
J = np.zeros((numSpin,numSpin))
J[np.triu_indices(numSpin,1)] = np.random.normal(0,1/np.sqrt(numSpin),numSpin*(numSpin-1)/2)
J = J+J.T

with open('../param/spin%d_beta%.1f.pkl'%(numSpin,beta),'wb') as f:
	pickle.dump({'h':h, 'J':J}, f)