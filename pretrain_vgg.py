import mxnet as mx
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_rcv1
from scipy.sparse import csr_matrix
import NN
import Data

# hyper parameters
X_SHAPE = (3, 32, 32) # x dimension, for mnist is 784
LAYER = 15
OUTPUTN = 10
STEP1 = 10000
epsilon = 0.0001
STEPSIZE = 0.001
batch_size = 100

X_train, y_train, X_val, y_val = Data.get_cifar10()
train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(X_val, y_val, batch_size)

# network symbols
data = mx.sym.var('data')
#data = mx.sym.flatten(data=data)
w = [None]*LAYER
b = [None]*LAYER
for i in range(LAYER):
  w[i] = mx.sym.var('w'+str(i), init=mx.initializer.Xavier())
  b[i] = mx.sym.var('b'+str(i), init=mx.initializer.Uniform())

# build network
fc = NN.vgg_like_dropout_K(data, w, b, OUTPUTN, 1.25)
loss = mx.sym.SoftmaxOutput(data=fc, name='softmax')
mlp_model = mx.mod.Module(symbol=loss, context=mx.gpu(0))

# sym, arg_params, aux_params = mx.model.load_checkpoint('pretrained/cifar10/vgg_like_K025', 600)
# mlp_model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# mlp_model.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

sgd_opt = mx.optimizer.SGD(learning_rate=1e-1, momentum=0.5, wd=0.0001, rescale_grad=(1.0/batch_size))
#sgd_opt = mx.optimizer.DCASGD(momentum=0.0, lamda=0.04 )
#sgd_opt = mx.optimizer.Adam(learning_rate=1e-3, rescale_grad=(1.0/batch_size))#, momentum=0.9, wd=0.0001, rescale_grad=(1.0/batch_size))
def lr_callback(epoch, *args):
   if epoch % 50 == 0 and epoch>0 and sgd_opt.lr >= 1e-5:
     sgd_opt.lr /= 2 # decrease learning rate by a factor of 10 every 10 batches
   logging.debug('----------------> epoch:%d, learning rate:%f' % (epoch, sgd_opt.lr))

mlp_model.fit(train_iter,  # train data
              eval_data=val_iter, # validation data
              optimizer=sgd_opt,  # use SGD to train
              eval_metric='acc',  # report accuracy during training
              epoch_end_callback = [mx.callback.do_checkpoint('pretrained/cifar10/vgg_like_dropout_K125', period=50), lr_callback],
              begin_epoch=1,
              num_epoch=STEP1)  # train for at most 10 dataset passes
print mlp_model.score(val_iter, mx.metric.Accuracy())

