
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
p_sel = p[index_select]
p_sel = p_sel / np.sum(p_sel)
E_sel = E[index_select]
logp_sel = -beta*E_sel
print logp_sel

# ####
# ####
nade = NADE(hid_dim,x_dim)
for istep in xrange(0,nSteps,50):

    print "istep %d"%istep

    data = torch.load('../output/learn_with_data_spin%d_beta%.1f_step%d.pt'%(numSpin,beta,istep))
    nade.load_state_dict(data['state_dict'])

    with torch.no_grad():
        logq_sel = -nade.cal_neg_log_llh(Q_sel)
        logq_sel = logq_sel.numpy()

    DKL = np.sum((logp_sel-logq_sel)*p_sel) # this is actually beta*W_A->A'
    DKL_record.append(DKL)

    # E_q = -1./beta*logq_sel
    # E_q = -logq_sel
    # E_q_record.append(E_q)

DKL_record = np.array(DKL_record)
E_q_record = np.array(E_q_record)
E_p_record = np.array(E_sel)
with open('../output/DKL_pq_record_spin%d_beta%.1f.pkl'%(numSpin,beta),'wb') as f:
    # pickle.dump({'DKL_record':DKL_record,'E_p':E_p_record,'E_q':E_q_record},f)
    pickle.dump({'DKL_record':DKL_record},f)

