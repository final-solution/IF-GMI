import pickle
import torch
import sys

def change(path, name):
    with open(path, 'rb') as f:
        G = pickle.load(f)['G_ema']
        torch.save({'state_dict': G.state_dict()}, name)


if __name__ == '__main__':
    sys.path.append('stylegan2_intermediate')
    path = 'stylegan2_intermediate/afhqdog.pkl'
    name = 'afhq_G.pth'
    change(path=path, name=name)
