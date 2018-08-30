import mxnet as mx
import numpy as np
import os

def get_data(name):
  if name=='mnist':
    return get_mnist()
  if name=='cifar10':
    return get_cifar10()

def get_mnist():
  mnist = mx.test_utils.get_mnist()
  X_train = mnist['train_data']
  y_train = mnist['train_label']
  X_val = mnist['test_data']
  y_val = mnist['test_label']
  os.system('rm *.gz')
  return X_train, y_train, X_val, y_val
# train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
# val_iter = mx.io.NDArrayIter(X_val, y_val, batch_size)
# return train_iter, val_iter
  
def get_cifar10():
  def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
  X_train = np.empty(shape=[0, 3, 32, 32])
  y_train = []
  for i in range(5):
    data = unpickle('../data/cifar-10-batches-py/data_batch_'+str(i+1))
    data['data'] = np.reshape(data['data'], (data['data'].shape[0],3,32,32))
    X_train = np.concatenate((X_train, data['data']), axis=0)
    y_train = y_train + data['labels']
  y_train = np.array(y_train)
  data = unpickle('../data/cifar-10-batches-py/test_batch')
  X_val = np.reshape(data['data'], (data['data'].shape[0],3,32,32))
  y_val = np.array(data['labels'])
  return X_train, y_train, X_val, y_val
 
def download_cifar10():
  if not os.path.exists('./data'): os.mkdir('./data')
  cmd = 'curl -o ./data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  subprocess.call(cmd.split())
  cmd = 'tar -xvzf ./data/cifar-10-python.tar.gz -C ./data/'
  subprocess.call(cmd.split())

def download_cifar100():
  if not os.path.exists('../data'): os.mkdir('../data')
  if not os.path.exists('../data/cifar100'): os.mkdir('../data/cifar100')
  cmd = 'curl -o ../data/cifar100/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
  subprocess.call(cmd.split())
  cmd = 'tar -xvzf ../data/cifar100/cifar-100-python.tar.gz -C ../data/cifar100/'
  subprocess.call(cmd.split())
