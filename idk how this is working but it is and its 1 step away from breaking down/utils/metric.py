import time
import torch
import numpy as np
from torchmetrics.regression import ConcordanceCorrCoef

def accuracy_av(output, target):
    with torch.no_grad():
        concordance = ConcordanceCorrCoef(num_outputs=2)
        # Assuming output and target are both torch tensors
        batch_size = target.size(0)

        # Calculate the Concordance Correlation Coefficient (CCC)
        cc = concordance(output, target)

        # Calculate the average of the Arousal and Valence CCCs
        avg_cc = cc.mean() * 100.0 / batch_size

        #print("Returning average CCC = ", avg_cc.item())

        # Convert the average CCC to a scalar and put it into a list
        return avg_cc.item()



def accuracy_au(output, target, topk=(1,)): 


    with torch.no_grad():
        
        batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()

        pred = output

        #print("PREDS = ", pred, " Targets: ",target)
        correct = pred.eq(target)

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum().item()
            res.append(correct_k*100.0 / batch_size)

        if len(res)==1:
            #print("returning = ",res)
            return res[0]
        else:
            #print("returning = ",res)
            return res
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum().item()
            res.append(correct_k*100.0 / batch_size)

        if len(res)==1:
            #print("returning = ",res)
            return res[0]
        else:
            #print("returning = ",res)
            return res





class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval
    


    