import argparse
import sys
import os
import utils


##
class NNClassifier:

    def __init__(self, NN, lr=1e-04, w_decay=100):
        self.model = NN
        self.optimizer = optimizers.Adam(alpha=lr)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(rate=w_decay))

    def train(self, ktrain, kval, n_epochs=100, batch_size=1, no_improve_lim =50):
        self.tloss, self.tacc, self.vloss, self.vacc = [], [], [], []
        self.min_val_loss = 100000
        self.no_improve_lim = no_improve_lim
        self.no_improve = 0
        self.max_val_acc = 0
        best_model = self.optimizer.target.copy()

        for epoch in range(n_epochs):
            kbtrain = make_batches(ktrain, batch_size=batch_size, shuffle=True)
            self.n_batches = len(kbtrain)
            self.epoch = epoch
            self.tloss.append([])
            self.tacc.append([])
            for i_batch in range(self.n_batches):
                loss = self.model(Variable(kbtrain[i_batch][0]), Variable(kbtrain[i_batch][1])).loss
                self.tloss[epoch].append(loss.data[()])
                self.tacc[epoch].append(F.accuracy(self.model.y, kbtrain[i_batch][1]).data[()])
                self.print_report('train')
                self.model.cleargrads()
                loss.backward()
                self.optimizer.update()

            model_val = self.model.copy()
            self.vloss.append(model_val(Variable(kval[0]), Variable(kval[1])).loss.data[()])
            self.vacc.append(F.accuracy(model_val.y, kval[1]).data[()])
            self.vcorrect = np.sum(F.argmax(model_val.y, axis=1).data == kval[1])
            self.vall = len(kval[1])

            if (self.vloss[-1] < self.min_val_loss): ## | (self.vacc[-1] > self.max_val_acc):
                best_model = self.optimizer.target.copy()
                self.min_val_loss = self.vloss[-1]
                self.max_val_acc = self.vacc[-1]
                self.no_improve = 0

            self.print_report('val')
            if self.no_improve >= self.no_improve_lim:
                print
                print "Validation loss did not reduce in " + str(self.no_improve_lim) + " iterations"
                print "Quit iteration loop"
                break

            self.no_improve += 1
            self.optimizer.new_epoch()

        self.model = best_model


    def test(self, ktest):
        self.test_loss = []
        self.test_acc = []
        model_test = self.model.copy()
        self.test_loss.append(model_test(Variable(ktest[0]), Variable(ktest[1])).loss.data[()])
        self.test_acc.append(F.accuracy(model_test.y, ktest[1]).data[()])
        self.test_correct = np.sum(F.argmax(model_test.y, axis=1).data == ktest[1])
        self.test_n = len(ktest[1])
        self.test_targets = ktest[1]
        self.test_predictions = F.argmax(model_test.y, axis=1).data
        self.test_proba = F.softmax(model_test.y).data
        self.test_maxproba = F.max(Variable(self.test_proba), axis=1).data
        self.print_report('test')


    def print_report(self, report='train'):
        if report == 'train':
            print "Epoch: " + str(self.epoch + 1) + \
                  ", train loss: " + str(round(self.tloss[-1][-1], 4)) + ", train acc: " + str(round(self.tacc[-1][-1], 4))
        elif report == 'val':
            print "\t\t\t val loss: " + str(round(self.vloss[-1], 4)) + ", val acc: " + str(round(self.vacc[-1], 4)) + \
                  " (" + str(self.vcorrect) + "/" + str(self.vall) + ")"
            print "\t\t\t min. val loss: " + str(round(self.min_val_loss, 4)) +\
                                        " (" + str(self.no_improve) + "/" + str(self.no_improve_lim) + ")"
        elif report == 'test':
            print "Test loss: " + str(round(self.test_loss[-1], 4)) + ", test acc: " + str(round(self.test_acc[-1], 4))\
                  + " (" + str(self.test_correct) + "/" + str(self.test_n) + ")"

            print "target: " + str(self.test_targets) + ", prediction: " + str(self.test_predictions) +\
                  ", probability: " + str(self.test_maxproba)


def run_NNclassifier(params):

    if params.out_dir is not None:
        if not os.path.exists(params.out_dir): os.makedirs(params.out_dir)

    for key, value in params.__dict__.items():
        print str(key) + ': \t\t\t' + str(value)

    # load data: x: ntrials x ntimepoints x nfeatures, y: ntrials
    x = np.load(params.x_file)
    y = np.load(params.y_file)
    n_classes = len(np.unique(y))

    # get train, val and test sets
    n_folds = int(100 / params.test_pcnt)
    Train, Val, Test = utils.make_kcrossvalidation(x, y, n_folds, shuffle=True)
    Train, Val, Test = utils.zscore_dataset(Train, Val, Test, z_train=True, zscore_x=params.zscore, zscore_y=False)
    Train, Val, Test = utils.dim_check(Train, Val, Test, nn_type=params.nn_type, nn_dim=params.n_dim)

    # train model
    Models = []
    for kfold in range(n_folds):
        print "Fold " + str(kfold)
        NN = make_NN(n_classes=n_classes, params=params)

        M = NNClassifier(NN, lr = params.lr, w_decay=params.w_decay)
        ktrain = utils.augment(Train[kfold], n_times=params.augment_times) if params.augment else Train[kfold]
        M.train(ktrain, Val[kfold], n_epochs=params.n_epochs, batch_size=params.batch_size)
        M.test(Test[kfold])

        Models.append(utils.copy_model(M, copyall=params.save_weights))

    # save models
    pM = utils.concat_models(Models)
    if params.out_dir is not None: utils.save_model(pM, params.out_dir + '/model' + str(n_folds) + '.p')

    print "Total performance: " + \
        str(round(sum(pM.test_correct) / float(sum(pM.test_n)), 4)) + " (" +\
            str(sum(pM.test_correct)) + '/' + str(sum(pM.test_n)) + ")"


##
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_file', '-x', type=str)
    parser.add_argument('--y_file', '-y', type=str)
    parser.add_argument('--test_pcnt', '-t', type=int, default=5)
    parser.add_argument('--n_filters', '-f', type=str, default=32)
    parser.add_argument('--filter_size', '-s', default=9)
    parser.add_argument('--n_epochs', '-e', type=int, default=100)
    parser.add_argument('--w_decay', '-w', default = 100)
    parser.add_argument('--out_dir', '-o', type=str, default = './')
    args = parser.parse_args(args)
    run_NNclassifier(args)


if __name__ == '__main__':
    main(sys.argv[1:])


