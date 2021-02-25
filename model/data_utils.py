import numpy as np
import torch

def pad_nt_matrix_roberta(nt_idx_matrix, max_nt_len, max_length):
    n_nt, n_tokens = nt_idx_matrix.size()
    padded_matrix = torch.zeros((max_nt_len, max_length))
    # account for the [CLS] / <s> token - offset by 1
    padded_matrix[:n_nt, 1:n_tokens+1] = nt_idx_matrix
    return padded_matrix

def pad_nt_matrix_xlnet(nt_idx_matrix, max_nt_len, max_length):
    n_nt, n_tokens = nt_idx_matrix.size()
    padded_matrix = torch.zeros((max_nt_len, max_length))
    # account for the [CLS] / <s> token - offset by 1
    # print(max_nt_len, nt_idx_matrix.size(), max_length)
    padded_matrix[:n_nt, :n_tokens] = nt_idx_matrix
    # padded_matrix = np.flip(padded_matrix, axis=1)
    return padded_matrix