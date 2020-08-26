import numpy as np
from functions.utils import cuda2np
from parameters import *

import torch

class Evaluator:
    def __init__(self, mode='binary'):
        self.mode=mode
        self.precs = []
        self.recas = []
        self.accus = []
        self.confidence = torch.linspace(0, 1., 256)

    def add(self, pred, mask):
        if self.mode == 'binary':
            prec = torch.zeros(mask.shape[0], len(self.confidence))
            reca = torch.zeros(mask.shape[0], len(self.confidence))
            accu = torch.zeros(mask.shape[0], len(self.confidence))
            pred_tmp = pred.squeeze(dim=1).cpu()    # batch X height X width
            mask_tmp = mask.squeeze(dim=1).cpu()    # batch X height X width
            all_pixels = (mask.shape[0] * mask.shape[1] * mask.shape[2])

            for j in range(len(self.confidence)):
                pred_cls = (pred_tmp > self.confidence[j]).float()
                TP = (pred_cls * mask_tmp).sum(dim=-1).sum(dim=-1)  # [Batch X Height X Width] -> [Batch]
                TN = ((pred_cls == 0) & (mask_tmp == 0)).sum(dim=-1).sum(dim=-1)
                prec[:, j] = (TP + 1e-10) / (pred_cls.sum(dim=-1).sum(dim=-1) + 1e-10) # [Batch]
                reca[:, j] = (TP + 1e-10) / (mask_tmp.sum(dim=-1).sum(dim=-1) + 1e-10)
                accu[:, j] = (TP + TN) / all_pixels

            self.precs.append(prec) # [Iter X Batch]
            self.recas.append(reca)
            self.accus.append(accu)

        elif self.mode == 'pallete':
            pass

        else:
            print('wrong evaluator mode.')

    def view(self):
        precision = torch.cat(self.precs, dim=0).mean(dim=0).view(-1)
        recall = torch.cat(self.recas, dim=0).mean(dim=0).view(-1)
        accuracy = torch.cat(self.accus, dim=0).mean(dim=0).view(-1)
        f1_score = (2 * precision * recall)/(precision + recall)

        return list(precision), list(recall), list(accuracy), list(f1_score), list(self.confidence)