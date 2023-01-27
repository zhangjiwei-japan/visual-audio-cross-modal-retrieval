# coding: utf8
import torch
import torch.nn as nn
from torch.autograd import Variable

def center_loss(feature, label, lambdas):
    center = nn.Parameter(torch.randn(int(max(label).item() + 1), feature.shape[1]), requires_grad=True)
    # print(center.shape)  # torch.Size([2, 2])
    # print(label.shape)  # torch.Size([5])

    center_exp = center.index_select(dim=0, index=label.long())
    # print(center_exp.shape)  # torch.Size([5, 2])

    count = torch.histc(label, bins=int(max(label).item() + 1), min=0, max=int(max(label).item()))
    # print(count)  # tensor([3., 2.], device='cuda:0')

    count_exp = count.index_select(dim=0, index=label.long())
    # print(count_exp)  # tensor([3., 3., 2., 3., 2.], device='cuda:0')

    loss = lambdas / 2 * torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp))
    return loss

if __name__ == '__main__':
    data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32)
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)
    loss = center_loss(data, label, 2)
    print(loss)

