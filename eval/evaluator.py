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
        self.confidence = np.linspace(0, 1., 256)
        self.size = 0

    def add(self, pred, mask):
        if self.mode == 'binary':
            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            prec = np.zeros((batch_size, len(self.confidence)))
            reca = np.zeros((batch_size, len(self.confidence)))
            accu = np.zeros((batch_size, len(self.confidence)))
            pred_tmp = np.array(pred.squeeze(dim=1).cpu())    # batch X height X width
            mask_tmp = np.array(mask.squeeze(dim=1).cpu())    # batch X height X width
            self.size = batch_size * height * width

            for j in range(len(self.confidence)):
                pred_cls = np.zeros(tuple(pred_tmp.shape))
                mask_cls = np.zeros(tuple(mask_tmp.shape))
                pred_cls[pred_tmp > self.confidence[j]] = 1.0
                mask_cls[mask_tmp > 0.2] = 1.0
                # pred_cls = (pred_tmp > self.confidence[j]).float()
                TP = (pred_cls * mask_cls).sum() # [Batch X Height X Width] -> [Batch]
                TN = ((pred_cls == 0.0) & (mask_tmp == 0.0)).sum()
                prec[:, j] = (TP + 1e-10) / (pred_cls.sum() + 1e-10) # [Batch]
                reca[:, j] = (TP + 1e-10) / (mask_cls.sum() + 1e-10)
                accu[:, j] = (TP + TN) / self.size

            self.precs.append(prec) # [Iter X Batch]
            self.recas.append(reca)
            self.accus.append(accu)

        elif self.mode == 'pallete':
            pass

        else:
            print('wrong evaluator mode.')

    def value(self):
        precision = np.concatenate(self.precs, axis=0).mean(axis=0).reshape(-1)
        recall = np.concatenate(self.recas, axis=0).mean(axis=0).reshape(-1)
        accuracy = np.concatenate(self.accus, axis=0).mean(axis=0).reshape(-1)
        f1_score = (2 * precision * recall) / (precision + recall)
        return precision, recall, accuracy, f1_score

    def mean_data(self):
        precision, recall, accuracy, f1_score = self.value()
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        mean_accuracy = np.mean(accuracy)
        mean_f1 = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall)
        return mean_precision, mean_recall, mean_accuracy, mean_f1

    def plot_data(self):
        precision, recall, accuracy, f1_score = self.value()
        return list(precision), list(recall), list(accuracy), list(f1_score), list(self.confidence)
