<h3>Neural network classifier implemented in Chainer</h3>
By default trains a 2d CNN on X to predict categorical labels in Y. During training Adam optimizer minimizes cross-entropy loss.

Different architectures can be specified: 
  - varying dimensions for convolution (input x should be of corresponding shape): ```--n_dim``` or ```-d``` (default=2)
  - number of convolutional layers: ```--n_layers``` or ```-l``` (default=1)
  - kernel size (can be tuple or int; if int, then makes filter of size int along all dims): ```--filter_size``` or ```-s``` (default=9)
  - number of kernels: ```--n_filters``` or ```-f``` (default=32)
  - use batch normalization or not: ```--use_bn``` or ```-n``` (default=True)
  - amount of weight decay: ```--w_decay``` (default=0)
  - batch size: ```--batch_size``` or  ```-b``` (default=128)
  - max number of epochs: ```--n_epochs``` (default=100)
  - no improvement limit for early stopping (on validation set): ```--early_stop``` (default=10)
  - learning rate: ```--lr``` (default=1e-04)
  - save trained weights: ```--save_weights``` (default=True)
  
Instead of a CNN a MLP can be trained by setting ```--nn_type MLP``` and adjusting ```--n_hidden_fc``` (default=100)
  
Code used for classification problems using neural data

<h4>Help parameters:</h4>

```python train.py -help```

<h4>Usage:</h4>

```python train.py ../data/x_train_m.npy ../data/t_train.npy -t 10 -f 64 -b 200 -l 2```

```python train.py ../data/x_train_m.npy ../data/t_train.npy --test_pcnt 10 --batch_size 200 --save_weights False```



<h4>Alternatively, a MLP can be trained:</h4>

```python train.py ../data/x_train_m.npy ../data/t_train.npy --test_pcnt 10 --nn_type MLP --batch_size 200```

