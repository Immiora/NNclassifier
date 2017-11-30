import numpy as np

def shuffle_list(l):
    ind = np.random.permutation(range(len(l)))
    return [l[i] for i in ind]

def split_list(l, wanted_parts=1):
    length = len(l)
    return [ l[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

def flatten_list(l):
    return [val for sublist in l for val in sublist]