import numpy as np
import list_utils as jb_utils
import cPickle as pickle
import copy
from models import CNN_ND, MLP


def zscore_dataset(Train, Val, Test, z_train=True, zscore_x=True, zscore_y=True, verbose=True):

    '''
    Z-score Train, Val and Test to mu and sigma of Train. Z-scoring is done per k_fold.
    If z_train is False, Val and Test are z-scored using own mu and sigma.
    '''
    to_zscore = [zscore_x, zscore_y]
    for k_fold in range(len(Train)):
        for i in range(2):
            if to_zscore[i]:
                Train[k_fold][i], m, s = zscore_2dfrom3d(Train[k_fold][i])

                if z_train == False:
                    m, s = None, None

                if Val is not None: Val[k_fold][i] = zscore_2dfrom3d(Val[k_fold][i], axis=0, m=m, s=s)[0]
                Test[k_fold][i] = zscore_2dfrom3d(Test[k_fold][i], axis=0, m=m, s=s)[0]

    if verbose: print "Z-scored"
    return Train, Val, Test


def zscore_2dfrom3d(x, axis=0, m=None, s=None):

    '''
    Function standartizes 3d x data by reshaping to 2d and normalizing along 1st dimension
    :param x: 3d input array: ntrials x ntimepoints x nfeatures
    :return: zscored x
    '''
	
    def z_score(x, axis=0):
        m = np.mean(x, axis=axis, keepdims=True)
        s = np.std(x, axis=axis, keepdims=True)
        return (x - m) / s, m, s

    dims = x.shape
    x0 = x.reshape((-1, dims[-1]))
    if (m is None) & (s is None):
        x0, m, s = z_score(x0, axis=axis)
    else:
        x0 -= m
        x0 /= s

    x0 = x0.reshape(dims)
    return x0, m, s


def make_kcrossvalidation(x, y, n_folds, shuffle=True):

    '''
    Function creates cross-validation folds for x and y inputs.
    :param x: 3d input array: ntrials x ntimepoints x nfeatures
    :param y: vector of labels: ntrials,
    :param n_folds: number of splits of the data
    :param n_val: number of trials in validation set
    :return: Train, Val and Test lists of nfolds length, each item is a tuple of x and y data
    '''

    n = x.shape[0]
    shape = x.shape[1:]
    x = x.reshape(n, -1)
    if shuffle:
        inx = np.random.permutation(range(n))
        x = x[inx]
        y = y[inx]

    l = range(n)
    test_folds = jb_utils.split_list(l, n_folds)

    Train, Val, Test = [], [], []

    for i, t in enumerate(test_folds):
        xc = x.copy()
        yc = y.copy()
        Test.append([xc[t].reshape([-1] + [shape[s] for s in range(len(shape))]), y[t]])

        xc = np.delete(xc, t, axis=0)
        yc = np.delete(yc, t, axis=0)

        xc = np.roll(xc, -len(t) * i, axis = 0)
        yc = np.roll(yc, -len(t) * i, axis = 0)

        Val.append([xc[:len(t)].reshape([-1] + [shape[s] for s in range(len(shape))]), yc[:len(t)]])
        Train.append([xc[len(t):].reshape([-1] + [shape[s] for s in range(len(shape))]), yc[len(t):]])

    return Train, Val, Test


def make_batches(ktrain, batch_size, shuffle=True):

    '''Takes in a tuple of x,y, returns list of batches, each is batch_size x other_dims.
    Shuffle for independent trials. First dimension is trials'''

    def make_batch_indices(L, batch_size):
        s = int(np.round(len(L) / float(batch_size)))
        batch_size1 = len(L) / s + 1
        return jb_utils.split_list(L, batch_size1)

    if shuffle:
        idx = np.random.permutation(ktrain[0].shape[0])
        ktrain = [ktrain[i][idx] for i in range(len(ktrain))]

    batch_indices = make_batch_indices(L=range(ktrain[0].shape[0]), batch_size=batch_size)
    n_batches = max([len(i) for i in batch_indices])

    kbtrain = []

    for i_batch in range(n_batches):
        b = np.array([i[i_batch] for i in batch_indices if i_batch < len(i)])
        kbtrain.append([ktrain[i][b] for i in range(len(ktrain))])

    return kbtrain


def augment_n_times(data, n=2):
    '''
    Augment dataset n times
    :param n: factor by which to augment
    :return: stacked augmented dataset
    '''
    stacked = data
    for _ in range(n-1):
        temp = np.zeros(data.shape)
        for im in range(data.shape[0]):
            s = np.random.randint(-3, 3, data.shape[-1])
            for i, si in enumerate(s):
                temp[im, :, :, i] = np.roll(data[im,:,:,i], shift=si, axis=-1)
        stacked = np.append(stacked, temp, axis=0)
    return stacked


def smooth_signal(y, n):
    box = np.ones(n)/n
    ys = np.convolve(y, box, mode='same')
    return ys


def smooth_ends(data, n_points, axis=0):
    '''
    Trim axis by n_points on both sides
    :param data: n-dim data
    :param n_points: number of points
    :param axis:
    :return:
    '''
    def smooth_along_dim(d, n):
        d[:n] = d[n]
        d[-n:] = d[-n]
        return d
    return np.apply_along_axis(smooth_along_dim, axis=axis, arr=data, n=n_points)


def augment(data, n_times):
    '''
    Augment dataset
    :param data: list of data: [X, y]
    :param n_times: augment by n_times
    :return: new ktrain, unshaffled
    '''
    X_new = augment_n_times(data=data[0], n=n_times)
    X_new = smooth_ends(X_new, n_points=3, axis=2)
    return [X_new.astype(np.float32), np.repeat(np.atleast_2d(data[1]), repeats=n_times, axis=0).flatten()]


# def augment(ktrain):
#     ktrain_new = []
#     ktrain_new.append(np.vstack([ktrain[0], np.roll(ktrain[0], shift=3, axis=2), np.roll(ktrain[0], shift=-3, axis=2)]))
#     ktrain_new.append(np.hstack([ktrain[1], ktrain[1], ktrain[1]]))
#
#     idx = np.random.permutation(ktrain_new[0].shape[0])
#     return [ktrain_new[i][idx] for i in range(len(ktrain_new))]


def make_NN(n_classes, params):
    if params.nn_type == 'CNN_ND':
        NN = CNN_ND(n_layers=params.n_layers,
                 n_dim = params.n_dim,
                 n_filters=params.n_filters,
                 filter_size=params.filter_size,
                 pad_size=(params.filter_size - 1) / 2 if type(params.filter_size) is int
                     else tuple([(i - 1) / 2 for i in params.filter_size]),
                 n_output=n_classes,
                 use_bn=params.use_bn)
    elif params.nn_type == 'MLP':
        NN = MLP(n_layers=params.n_layers,
                 n_hidden=params.n_hidden,
                 n_output=n_classes)
    return NN


def dim_check(Train, Val, Test, nn_type, nn_dim):
    '''
    Function checks input dimensionality. Mostly necessary to add color dimension for CNN
    :param Train: Train set, list of folds, each is a tuple of x and y data
    :param Val: Val set, same as above
    :param Test: Test set, same as above
    :param nn_type: CNN or others
    :return:
    '''
    if (nn_type in ['CNN', 'CNN_ND']) & (Train[0][0].ndim != nn_dim + 2):
        Train = expand_x_dims(Train)
        Val = expand_x_dims(Val)
        Test = expand_x_dims(Test)

    elif (nn_type == 'MLP') & (Train[0][0].ndim != 2):
        Train = reduce_x_dims(Train)
        Val = reduce_x_dims(Val)
        Test = reduce_x_dims(Test)

    return Train, Val, Test


def expand_x_dims(Data):

    '''
    Function expands second dimension of x data in Train, Test and Val (for 2D CNN architectures)
    Should make it better somehow because now its purpose is basically to modify tuple Train which is conceptually wrong
    :param Data: list of tuples x and y data
    :return: new Train with expanded second dimension
    '''

    xData = [Data[i][0] for i in range(len(Data))]
    xData = [np.expand_dims(xData[i], axis=1) for i in range(len(xData))]

    New = []
    [New.append([xData[i], Data[i][1]]) for i in range(len(Data))]

    return New


def reduce_x_dims(Data):

    '''
    Function reduces 1: dimensions of x data in Train, Test and Val (for MPL architectures)
    Should make it better somehow because now its purpose is basically to modify tuple Train which is conceptually wrong
    :param Data: list of tuples x and y data
    :return: new Train with expanded second dimension
    '''

    xData = [Data[i][0] for i in range(len(Data))]
    xData = [np.reshape(xData[i], (xData[i].shape[0], -1)) for i in range(len(xData))]

    New = []
    [New.append([xData[i], Data[i][1]]) for i in range(len(Data))]

    return New


def copy_model(M, copyall = False):
    pM = copy.deepcopy(M)
    if hasattr(M, 'optimizer'): del (pM.optimizer)
    if copyall == False:
        del(pM.model)
    return pM


def concat_models(Models):
    dicts = {}
    for key, value in Models[0].__dict__.items():

        dicts[key] = []
        [dicts[key].append(Models[i].__dict__[key]) for i in range(len(Models))]
    dicts['n_folds'] = len(Models)
    all_models = ConcatModel(**dicts)
    return all_models


def save_model(pM, fname):
    pfile = open(fname, mode='wb')
    pickle.dump(pM, pfile)
    pfile.close()


def load_model(fname):
    pfile = open(fname, mode='rb')
    out = pickle.load(pfile)
    pfile.close()
    return out


class ConcatModel(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)