import torch
from utils.LieGroup import *


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def getPosError(output, target):
    nBatch = len(target)
    if output.shape != (nBatch, 4, 4) or target.shape != (nBatch, 4, 4):
        print(f'[ERROR] getPosError : output.shape = {output.shape}, target.shape = {target.shape}')
        exit(1)
    return output[:, :3, 3] - target[:, :3, 3]


def getOriError(output, target):
    nBatch = len(target)
    if output.shape != (nBatch, 4, 4) or target.shape != (nBatch, 4, 4):
        print(f'[ERROR] getOriError : output.shape = {output.shape}, target.shape = {target.shape}')
        exit(1)
    SO3error = invSO3(output[:, :3, :3]) @ target[:, :3, :3]
    return skew_so3(SO3error)


def getSE3PosRmseError(output, target):
    nBatch = len(target)
    posError = getPosError(output, target)
    return (posError.pow(2).sum() / nBatch).item()


def getSE3OriRmseError(output, target):
    nBatch = len(target)
    oriError = getOriError(output, target)
    return (oriError.pow(2).sum() / nBatch).item()
