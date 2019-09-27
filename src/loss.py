import os
import torch
import torch.nn as nn
import numpy as np

def distance_matrix(x):
    x_norm = (x**2).sum(1).view(-1, 1)
    dist = x_norm + x_norm.view(1, -1) - 2.0 * torch.mm(x, torch.transpose(x, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)

def kronecker(matrix1, matrix2):
    # Made by OCY, from the PyTorch forums
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))

def get_hardest_points(D, M, N):
    '''
        Find hardest points (D, M, N)
    '''
    really_big_number = 1e10

    same_user_mask = kronecker(
        kronecker(torch.eye(N,N), torch.ones(1,M)),
        torch.ones(M,1)
    )

    # Risky, but fast
    #different_user_mask = really_big_number*same_user_mask+1 

    hardest_positives = (same_user_mask*D).max(1)
    negatives = D[(1-same_user_mask)==1].view(M*N, M*(N-1))
    hardest_negatives = negatives.min(1)
    
    #hardest_negatives = ((different_user_mask*(D+1)-1)).min(1)

    # safer (but slower) alternative:
    # > negatives = D[(1-same_user_mask)==1].view(M*N, M*(N-1))
    # > hardest_negatives = negatives.min(1)
    
    return hardest_positives[0], hardest_negatives[0]

def topk_hardest_mean(D, M, N, k_positives = 1, k_negatives = 10, use_cuda = False):
    same_user_mask = kronecker(
        kronecker(torch.eye(N,N), torch.ones(1,M)),
        torch.ones(M,1)
    )
    
    if use_cuda:
        same_user_mask = same_user_mask.cuda()
    
    hardest_positives = ((same_user_mask*D).topk(k_positives, 1)[0]).mean(1)
    negatives = D[(1-same_user_mask)==1].view(M*N, M*(N-1))
    hardest_negatives = (-((-negatives).topk(k_negatives, 1)[0])).mean(1)

    return hardest_positives, hardest_negatives


def batch_hard_loss(hardest_positives, hardest_negatives, margin = 0.1):
    relu_s = nn.ReLU()(hardest_positives - hardest_negatives + margin)
    return relu_s.mean()
