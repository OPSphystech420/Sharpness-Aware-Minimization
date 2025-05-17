import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, gold, smoothing=0.1):
    """
    Вычисляет кросс-энтропию с label smoothing.
    pred: тензор логитов формы [batch, classes]
    gold: тензор целевых меток [batch]
    smoothing: степень размытия меток
    """
    n_class = pred.size(1)
    # создаём one-hot с небольшим заполнением для всех классов
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(1, gold.unsqueeze(1), 1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)
    return F.kl_div(log_prob, one_hot, reduction='none').sum(-1)
