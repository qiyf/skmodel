## NADE implementation
## Original Paper: Uria et al. arXiv:1605.02226v3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse
torch.set_default_dtype(torch.double)

class NADE(nn.Module):
    def __init__(self,hid_dim,x_dim):
        super(NADE, self).__init__()
        self.hid_dim = hid_dim
        self.x_dim = x_dim
        
        self.W_hx = nn.Parameter(torch.randn(hid_dim,x_dim),requires_grad=True)
        self.c = nn.Parameter(torch.zeros(hid_dim),requires_grad=True)
        self.mask_W_hx = nn.Parameter(torch.tril(torch.ones(hid_dim,x_dim),diagonal=-1),requires_grad=False)

        self.W_xh = nn.Parameter(torch.randn(x_dim,hid_dim),requires_grad=True)
        self.b = nn.Parameter(torch.zeros(x_dim),requires_grad=True)
        self.mask_W_xh = nn.Parameter(torch.tril(torch.ones(x_dim,hid_dim),diagonal=0),requires_grad=False)


    def forward(self,inputs):
        h = torch.matmul(inputs, (self.W_hx*self.mask_W_hx).T) + self.c
        h = torch.sigmoid(h)
        p = torch.matmul(h, (self.W_xh*self.mask_W_xh).T) + self.b
        return p

    def cal_neg_log_llh(self, x):
        res = self.forward(x)
        lossf = nn.BCEWithLogitsLoss(reduction = "none")
        llh = lossf(res,x)
        llh = torch.sum(llh, -1)
        return llh
        
    def sampling(self, num_samples):
        x = torch.randn(num_samples,self.x_dim)
        for d in xrange(self.x_dim):
            hd = torch.matmul(x[:,:d+1], (self.W_hx[:,:d+1]*self.mask_W_hx[:,:d+1]).T)+self.c
            hd = torch.sigmoid(hd)
            pd = torch.matmul(hd,(self.W_xh[d,:]*self.mask_W_xh[d,:]).T)+self.b[d]
            pd = torch.sigmoid(pd)
            dist = torch.distributions.Bernoulli(probs=pd)
            x_d = dist.sample()
            x[:,d] = x_d
        return x
