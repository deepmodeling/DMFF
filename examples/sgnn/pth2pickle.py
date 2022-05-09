#!/usr/bin/env python
import sys
import torch
import pickle
import re

# load pth
pth = sys.argv[1]
state_dict = torch.load(sys.argv[1])

for k in state_dict:
    state_dict[k] = state_dict[k].numpy()

ofn = re.sub('\.pth$', '.pickle', pth)
with open(ofn, 'wb') as ofile:
    pickle.dump(state_dict, ofile)

