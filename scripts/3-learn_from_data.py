import numpy as np
import pickle
import itertools
import torch.nn as nn
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.double)
from NADE import *
import torch.optim as optim

# ####
# load basic param
# ####
with open('../param/params.pkl','r') as f:
	param = pickle.load(f)
	numSpin = param['numSpin']
	beta = param['beta']

with open('../param/spin%d_beta%.1f.pkl'%(numSpin,beta),'rb') as f:
	model = pickle.load(f)
	h = model['h']
	J = model['J']

with open('../output/solve_model_numerical_spin%d_beta%.1f.pkl'%(numSpin,beta),'rb') as f:
	data = pickle.load(f)
	Q = data['Q']
	E = data['E']
	p = data['p']
Q = np.array(Q)
Q[Q==-1]=0

# ####
# learning param
# ####
hid_dim = numSpin
x_dim = numSpin
nSteps = 10000
lr = 1E-3
ntrain = 10000 #2**(numSpin-1) # use half of the data 

# idx_rdm = np.random.choice(range(len(Q)),ntrain,replace=False,p=p)
idx_rdm = np.random.choice(range(len(Q)),ntrain,replace=True,p=p)
with open('../output/learn_with_data_spin%d_beta%.1f_index_select.pkl'%(numSpin,beta),'wb') as f:
	pickle.dump({'index_select':idx_rdm},f)

Q_batch = torch.from_numpy(Q[idx_rdm]).double()
loss_record = []

# ####
# define optimizer
# ####
nade = NADE(hid_dim,x_dim)
optimizer = optim.Adam(nade.parameters(),lr=lr)

# ####
# training
# ####
for istep in xrange(nSteps):
	# loss = torch.mean(nade.cal_neg_log_llh(Q_batch)) # cross entropy, positive
	loss = torch.mean(nade.cal_neg_log_llh(Q_batch)) # cross entropy, positive

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	loss_record.append(loss.item())
	if istep % 50 == 0:
		print('step: %d, loss: %.5f'%(istep,loss.item()))
		torch.save({'state_dict':nade.state_dict()},
			'../output/learn_with_data_spin%d_beta%.1f_step%d.pt'%(numSpin,beta,istep))

with open('../output/loss_record_spin%d_beta%.1f.pkl','wb') as f:
	pickle.dump({'loss_record':loss_record},f)

