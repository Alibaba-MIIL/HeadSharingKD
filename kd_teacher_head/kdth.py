from __future__ import print_function

from crd.criterion import CRDLoss

import torch.nn as nn
import torch.nn.functional as F


class ModelTH(nn.Module):
    """KD + teacher classifier"""
    def __init__(self, model_s, model_t, do_init_s_from_th=False):
        super(ModelTH, self).__init__()

        self.model_s = model_s
        self.t_head = nn.Linear(model_t.fc.in_features, model_t.fc.out_features)
        self.t_head.weight.requires_grad, self.t_head.bias.requires_grad = False, False
        model_t.fc.weight.requires_grad, model_t.fc.bias.requires_grad = False, False
        self.t_head.weight.copy_(model_t.fc.weight)
        self.t_head.bias.copy_(model_t.fc.bias)

        self.do_init_s_from_th = do_init_s_from_th
        if self.do_init_s_from_th:
            value_weight = self.model_s.fc.weight.requires_grad
            value_bias = self.model_s.fc.bias.requires_grad
            self.model_s.fc.weight.requires_grad = False
            self.model_s.fc.bias.requires_grad = False
            self.model_s.fc.weight.copy_(model_t.fc.weight)
            self.model_s.fc.bias.copy_(model_t.fc.bias)
            self.model_s.fc.weight.requires_grad = value_weight
            self.model_s.fc.bias.requires_grad = value_bias

    def forward(self, x, is_feat=False, preact=False):

        if not is_feat:
            logits = self.model_s(x)
            return logits
        else:
            feat, logits = self.model_s(x, is_feat=is_feat, preact=preact)
            logits_th = self.t_head(feat[-1])
        return feat, [logits, logits_th]


class KDTH(nn.Module):
    """KD with a Teacher Head auxiliary loss"""
    def __init__(self, T=4):
        super(KDTH, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        y_s_th = y_s[1]
        y_s = y_s[0]

        # Teacher activation
        p_t = F.softmax(y_t/self.T, dim=1)

        # # Student loss
        # p_s = F.log_softmax(y_s/self.T, dim=1)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]

        # Student with teacher-head loss
        p_s_th = F.log_softmax(y_s_th/self.T, dim=1)
        loss_th = F.kl_div(p_s_th, p_t, size_average=False) * (self.T**2) / y_s.shape[0]

        # # Combine loss
        # loss = loss + self.alpha_th * loss_th
        # loss = loss / 2

        return loss_th


class KDE(nn.Module):
    """KD on embeddings - KDE"""

    def __init__(self):
        super(KDE, self).__init__()

    def forward(self, embedding_s, embedding_t):
        # KDE loss
        inputs_embed = F.normalize(embedding_s, p=2.0, dim=1)
        targets_embed = F.normalize(embedding_t, p=2.0, dim=1)
        loss_kde = nn.MSELoss(reduction='sum')(inputs_embed, targets_embed)
        return loss_kde


class KDETH(nn.Module):
    """Combination of KDE and TH"""
    def __init__(self, T=4, th_weight=1.0):
        super(KDETH, self).__init__()
        self.kde = KDE()
        self.kdth = KDTH(T=T)

        self.th_weight = th_weight  # delta/gamma, because this loss is multiplied by gamma

    def forward(self, y_s, y_t, embedding_s, embedding_t):
        loss_kde = self.kde(embedding_s, embedding_t)
        loss_kdth = self.kdth(y_s, y_t)
        loss = loss_kde + self.th_weight * loss_kdth
        return loss


class CRDTH(nn.Module):
    """Combination of CRD and TH"""
    def __init__(self, opt, T=4.0, th_weight=1.0):
        super(CRDTH, self).__init__()
        self.crd = CRDLoss(opt)
        self.kdth = KDTH(T=T)

        self.th_weight = th_weight  # delta/gamma, because this loss is multiplied by gamma

    def forward(self, y_s, y_t, f_s, f_t, index, contrast_idx):
        loss_kdth = self.kdth(y_s, y_t)
        loss_crd = self.crd(f_s, f_t, index, contrast_idx)
        loss = loss_crd + self.th_weight * loss_kdth
        return loss