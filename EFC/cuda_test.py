# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:03:44 2024

@author: rfrazin
"""

import torch
avail = torch.cuda.is_available()
ver = torch.version.cuda
print('cuda available =', avail)
print('torch cuda version = ', ver)
# %%
if avail:
    device = torch.device('cuda')
    matrix1 = torch.rand(4, 4, device=device)
    matrix2 = torch.rand(4, 4, device=device)
    result = torch.matmul(matrix1, matrix2)
    print(result)