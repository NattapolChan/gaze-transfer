import torch
import math

def find_abs_angle_difference(a, b):
    # a/b = angle difference between reference and prediction

    assert a.size() == b.size()

    cos_theta = torch.cos(a/180 * math.pi) * torch.cos(b/180 * math.pi) 
    theta = torch.acos(cos_theta)
    return torch.sum(torch.abs(theta * 180 / math.pi))/theta.size(0)
