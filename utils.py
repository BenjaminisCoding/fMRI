import numpy as np

def stand(data):
    if np.iscomplexobj(data):
        data = data.abs()
    a,b = data.min(), data.max()
    return (data - a) / (b - a)