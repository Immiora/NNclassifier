<h3>Neural network classifier implemented in Chainer</h3>
(Draft)

Different architectures can be specified: 
  - varying dimensions for convolution, 
  - number of convolutional layers, 
  - kernel size,
  - number of kernels, 
  - use batch normalization or not, 
  - amount of weight decay
  - details of training.
  
Code used for classification problems using neural data


Examples:
```python train.py -help```

```python train.py ../data/x_train_m.npy ../data/t_train.npy -t 10 -f 64 -b 200 -l 2```

```python train.py ../data/x_train_m.npy ../data/t_train.npy --test_pcnt 10 --batch_size 200 --save_weights False```
