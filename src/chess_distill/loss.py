import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
        
    def forward(self, p_logits, v_pred, p_target, v_target):
        # p_logits: (N, 4672)
        # v_pred: (N, 1)
        # p_target: (N,) indices
        # v_target: (N, 1) scalars
        
        l_p = self.policy_loss(p_logits, p_target)
        l_v = self.value_loss(v_pred, v_target)
        
        return l_p + l_v, l_p, l_v
