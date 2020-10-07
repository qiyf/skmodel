import numpy as np
import pickle
import itertools

with open('../param/params.pkl','r') as f:
	param = pickle.load(f)
	numSpin = param['numSpin']
	beta = param['beta']

with open('../param/spin%d_beta%.1f.pkl'%(numSpin,beta),'rb') as f:
	model = pickle.load(f)
	h = model['h']
	J = model['J']

spin = [-1,1]
Q = list(itertools.product(spin,repeat=numSpin))
E = -(np.sum(np.matmul(Q,h),-1)+np.sum(Q*np.matmul(Q,J),-1))
p = np.exp(-beta*E)
Z = np.sum(p)
p = p/Z
F = -1/beta*np.log(Z)

with open('../output/solve_model_numerical_spin%d_beta%.1f.pkl'%(numSpin,beta),'wb') as f:
	pickle.dump({'Q':Q,'E':E,'p':p,'F':F}, f)
