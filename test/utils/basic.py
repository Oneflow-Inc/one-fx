import oneflow

def add_n(x, n):
    y = oneflow.relu(x)
    return y + n + len(x.shape)