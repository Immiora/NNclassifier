<h3>Neural network classifier implemented in Chainer</h3>
By default trains a 2d CNN on X to predict categorical labels in Y.

Different architectures can be specified: 
  - varying dimensions for convolution (input x should be of corresponding shape), 
  - number of convolutional layers, 
  - kernel size,
  - number of kernels, 
  - use batch normalization or not, 
  - amount of weight decay
  - details of training.

Alternatively, a MLP can be trained.

Code used for classification problems using neural data


<h4>Help parameters:<h4>
```python train.py -help```

<h4>Usage:<h4>
```python train.py ../data/x_train_m.npy ../data/t_train.npy -t 10 -f 64 -b 200 -l 2```

```python train.py ../data/x_train_m.npy ../data/t_train.npy --test_pcnt 10 --batch_size 200 --save_weights False```

```python train.py ../data/x_train_m.npy ../data/t_train.npy --test_pcnt 10 --nn_type MLP --batch_size 200 --save_weights False```
