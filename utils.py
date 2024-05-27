import numpy as np
import torch 

def stand(data, clip = None):

    if np.iscomplexobj(data):
        data.real, data.imag = __stand__(data.real, clip.real), __stand__(data.imag, clip.imag)
    elif data.shape[-1] == 2:
        data[...,0], data[...,1] = __stand__(data[...,0], clip.real), __stand__(data[...,1], clip.imag)
    else:
        data = __stand__(data, clip.real)
    return data
            
def __stand__(data, clip = None):

    if clip is not None:
        data = np.clip(data, a_min = clip[0], a_max = clip[1])
    a,b = data.min(), data.max()
    return (data - a) / (b - a)

def check_type(obj):
    if isinstance(obj, torch.Tensor):
        return "Tensor"
    elif isinstance(obj, np.ndarray):
        return "Array"
    else:
        return "Unknown Type"
    
class Clip:

    def __init__(self, real = None, imag = None):
        self.real = real 
        self.imag = imag

    def quantile(self, data, q):
        if check_type(data) == 'Array':
            data = torch.Tensor(data)
        if np.iscomplexobj(data):
            self.real = (torch.quantile(data.real, 1-q), torch.quantile(data.real, q))
            self.imag = (torch.quantile(data.imag, 1-q), torch.quantile(data.imag, q))
        elif data.shape[-1] == 2:
            self.real = (torch.quantile(data[...,0], 1-q), torch.quantile(data[...,0], q))
            self.imag = (torch.quantile(data[...,1], 1-q), torch.quantile(data[...,1], q))
        else:
            self.real = (torch.quantile(data, 1-q), torch.quantile(data, q))
