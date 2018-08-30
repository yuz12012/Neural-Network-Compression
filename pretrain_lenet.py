import mxnet as mx
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_rcv1
from scipy.sparse import csr_matrix
import NN
import Data

# hyper parameters
X_SHAPE = (1,784) # x dimension, for mnist is 784
LAYER = 4
OUTPUTN = 10
STEP1 = 300
epsilon = 0.0001
STEPSIZE = 0.001
batch_size = 100

X_train, y_train, X_val, y_val = Data.get_mnist()
train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(X_val, y_val, batch_size)

# load and init all essential params, and train
#print arg_params['mw0']
#print mx.ndarray.sum(arg_params['mw0'])

# network symbols
data = mx.sym.var('data')
#data = mx.sym.flatten(data=data)
w = [None]*LAYER
b = [None]*LAYER
mw = [None]*LAYER
for i in range(LAYER):
  w[i] = mx.sym.var('w'+str(i), init=mx.initializer.Xavier())
  b[i] = mx.sym.var('b'+str(i), init=mx.initializer.Uniform())

# build network
fc = NN.lenet_5(data, w, b, OUTPUTN)
loss = mx.sym.SoftmaxOutput(data=fc, name='softmax')
mlp_model = mx.mod.Module(symbol=loss, context=mx.gpu(0))

# mlp_model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# mlp_model.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)

#lr_sch = mx.lr_scheduler.FactorScheduler(step=100, factor=0.5, stop_factor_lr=1e-08)
#mlp_model.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 1e-6), ('lr_scheduler', lr_sch)))

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

sgd_opt = mx.optimizer.SGD(learning_rate=1e-1, momentum=0.9, wd=0.0001, rescale_grad=(1.0/batch_size))
def lr_callback(epoch, *args):
   if epoch % 30 == 0 and epoch > 0 and sgd_opt.lr >= 1e-5:
     sgd_opt.lr /= 10 # decrease learning rate by a factor of 10 every 10 batches
     logging.debug('change learning rate to '+str(sgd_opt.lr))
   logging.debug('nbatch:%d, learning rate:%f' % (epoch, sgd_opt.lr))

#sym, arg_params, aux_params = mx.model.load_checkpoint('pretrained/mnist/lenet5', 10000)

mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer=sgd_opt,  # use SGD to train
              # arg_params=arg_params,
              # aux_params=aux_params,
              # allow_missing=True,
              #optimizer_params={'learning_rate':1e-6}, 
              eval_metric='acc',  # report accuracy during training
              epoch_end_callback = [mx.callback.do_checkpoint('pretrained/mnist/lenet5', period=100), lr_callback],
              #begin_epoch=10001,
              num_epoch=STEP1)  # train for at most 10 dataset passes
# print mlp_model.get_params()[0]['mw0']
# print mx.ndarray.sum(mlp_model.get_params()[0]['mw0'])
print mlp_model.score(val_iter, mx.metric.Accuracy())

