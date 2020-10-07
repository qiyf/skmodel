
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
# param
# ####
hid_dim = numSpin
x_dim = numSpin
nSteps = 10000
DKL_record = []
E_q_record = []

# ####
# p
# ####
with open('../output/learn_with_data_spin%d_beta%.1f_index_select.pkl'%(numSpin,beta),'rb') as f:
    data = pickle.load(f)
    index_select = data['index_select']

Q_sel = torch.from_numpy(Q[index_select]).double()
num_samples = len(Q_sel)

# ####
# ####
nade = NADE(hid_dim,x_dim)
for istep in xrange(0,nSteps,50):

    print "istep %d"%istep

    data = torch.load('../output/learn_with_data_spin%d_beta%.1f_step%d.pt'%(numSpin,beta,istep))
    nade.load_state_dict(data['state_dict'])

    with torch.no_grad():
        Q_sel = nade.sampling(num_samples)
        logq_sel = -nade.cal_neg_log_llh(Q_sel)

    Q_sel = Q_sel.numpy()
    Q_sel[Q_sel==0] = -1
    E_sel = -(np.sum(np.matmul(Q_sel,h),-1)+np.sum(Q_sel*np.matmul(Q_sel,J),-1))
    logp_sel = -beta*E_sel

    DKL = torch.mean((logq_sel-logp_sel)) # since q is a normalized distribution
    DKL_record.append(DKL.numpy())

    # E_q = -1./beta*logq_sel
    # E_q_record.append(E_q.numpy())

DKL_record = np.array(DKL_record)
E_q_record = np.array(E_q_record)
E_p_record = np.array(E_sel)
with open('../output/DKL_qp_record_spin%d_beta%.1f.pkl'%(numSpin,beta),'wb') as f:
    # pickle.dump({'DKL_record':DKL_record,'E_p':E_p_record,'E_q':E_q_record},f)
    pickle.dump({'DKL_record':DKL_record},f)

