import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt

with open('../param/params.pkl','rb') as f:
	param = pickle.load(f)
	numSpin = param['numSpin']
	beta = param['beta']

with open('../output/solve_model_numerical_spin%d_beta%.1f.pkl'%(numSpin,beta),'rb') as f:
	model = pickle.load(f)
	F_true = model['F']

num_steps = 5000
idx_step_list = range(0, num_steps, 50)

with open('../output/DKL_pq_record_spin%d_beta%.1f.pkl'%(numSpin,beta),'rb') as f:
	data = pickle.load(f)
	DKL_pq = data['DKL_record']
with open('../output/DKL_qp_record_spin%d_beta%.1f.pkl'%(numSpin,beta),'rb') as f:
	data = pickle.load(f)
	DKL_qp = data['DKL_record']

print 'Begin making the figure...'
fig = plt.figure(0)
fig.clf()
plt.plot(F_true*np.ones(len(DKL_pq)),color='k')
plt.plot(-DKL_pq/beta,color='r')
plt.plot(DKL_qp/beta,color='g')
plt.xscale('log')
plt.savefig('../output/DKL_F.pdf')

# plot loss
with open('../output/loss_record_spin%d_beta%.1f.pkl','rb') as f:
	data = pickle.load(f)
	loss = data['loss_record']
fig = plt.figure(1)
fig.clf()
plt.plot(loss,color='k')
plt.savefig('../output/loss.pdf')
