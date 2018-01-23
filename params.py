class Params(object):
    def __init__(self, **kwargs):
        self.x_file = None
        self.y_file = None
        self.test_pcnt = 5
        self.n_filters = 50
        self.filter_size = (3, 3)
        self.w_decay = 40
        self.n_epochs = 500
        self.nn_type = 'CNN_ND'
        self.n_layers = 1
        self.n_dim = 2
        self.batch_size = 100
        self.lr = 1e-04
        self.zscore = True
        self.augment = False
        self.augment_times = 2
        self.use_bn = False
        self.save_weights = False
        self.out_dir = None
        self.__dict__.update(kwargs)