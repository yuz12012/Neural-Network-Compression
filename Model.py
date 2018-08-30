import mxnet as mx
import numpy as np
import sys
import NN
from Config import * # import hyper parameters

def model_init():
  data = mx.sym.var('data')
  w = [None]*LAYER
  mw = [None]*LAYER
  b = [None]*LAYER
  for i in range(LAYER):
    w[i] = mx.sym.var('w'+str(i), init=mx.initializer.Xavier())
    b[i] = mx.sym.var('b'+str(i), init=mx.initializer.Uniform())
  fc = eval(NN_NAME) 
  loss = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  model = mx.mod.Module(symbol=loss, context=mx.gpu(0))
  return model

def init(train_iter, val_iter):
  model = model_init()
  import logging
  logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
  sgd_opt = mx.optimizer.SGD(learning_rate=LR_INIT, momentum=0.9, wd=0.0001, rescale_grad=(1.0/batch_size))
  def lr_callback(epoch, *args):
    if epoch % 10 == 0 and epoch > 0 and sgd_opt.lr >= 1e-3:
      sgd_opt.lr /= 2 # decrease learning rate by a factor of 10 every 10 batches
      print('change learning rate to '+str(sgd_opt.lr))
    print('nbatch:%d, learning rate:%f' % (epoch, sgd_opt.lr))

  if IF_WARMSTART == True:
    sym, arg_params, aux_params = mx.model.load_checkpoint(warm_prefix, WARM_EP)
    model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer=sgd_opt,  # use SGD to train
              eval_metric='acc',  # report accuracy during training
              epoch_end_callback = [mx.callback.do_checkpoint(prefix, period=STEP_INIT), lr_callback],
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              num_epoch=STEP_INIT,
              begin_epoch=STEP_INIT-STEP_FORWARD,
    )  
  else:
    print 'start'
    model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer=sgd_opt,  # use SGD to train
              eval_metric='acc',  # report accuracy during training
              epoch_end_callback = [mx.callback.do_checkpoint(prefix, period=STEP_INIT), lr_callback],
              num_epoch=STEP_INIT)  
  print '--------- finish initial train --------'
  return model.score(val_iter, mx.metric.Accuracy())

def model_forward(mask_w):
  data = mx.sym.var('data')
  #data = mx.sym.flatten(data=data)
  w = [None]*LAYER
  mw = [None]*LAYER
  b = [None]*LAYER
  for i in range(LAYER):
    b[i]  = mx.sym.var('b'+str(i))
    mw[i] = mx.sym.var('mw'+str(i), shape=mask_w[i].shape, init=mx.initializer.Constant(mask_w[i].asnumpy().tolist()))
    mw[i] = mx.sym.BlockGrad(mw[i])
    w[i]  = mx.sym.var('w'+str(i),  shape=mask_w[i].shape)
    w[i]  = mx.sym.broadcast_mul(w[i], mw[i])
  
  # build network
  fc = eval(NN_NAME)
  loss = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  model = mx.mod.Module(symbol=loss, context=mx.gpu(0))
  return model 

def model_backward(layer, arg_params):
  data = mx.sym.var('X') 
  data = mx.sym.BlockGrad(data)
  w = [None]*LAYER
  b = [None]*LAYER
  mw = [None]*LAYER
  for i in range(LAYER):
    if i == layer:
      w[i] = mx.sym.var('w'+str(i)) # w at target layer is var, otherwise are const
    else:
      mw[i] = mx.sym.var('mw'+str(i)) # mw is mask_w
      mw[i] = mx.sym.BlockGrad(mw[i])
      w[i] = mx.sym.broadcast_mul(mx.sym.var('w'+str(i)), mw[i])
      w[i] = mx.sym.BlockGrad(w[i])
    b[i] = mx.sym.var('b'+str(i))
    b[i] = mx.sym.BlockGrad(b[i])
  
  # build model
  fc = eval(NN_NAME)
  out_pick = mx.sym.var('id')
  out_pick = mx.sym.BlockGrad(out_pick)
  loss = - mx.sym.dot(mx.sym.softmax(fc), out_pick) + SPARSITY_PENALTY*mx.sym.sum(w[layer] != 0)
  #loss = - mx.sym.dot(fc, out_pick) + SPARSITY_PENALTY*mx.sym.sum(w[layer] != 0) #logit
  mlp = mx.sym.MakeLoss(loss)
  
  # build executor
  in_shapes = {'X':(1,)+X_SHAPE, 'id':(OUTPUTN,)}
  for i in range(LAYER):
    in_shapes['w'+str(i)] = arg_params['w'+str(i)].shape
    in_shapes['b'+str(i)] = arg_params['b'+str(i)].shape
    in_shapes['mw'+str(i)] = arg_params['w'+str(i)].shape
  exe = mlp.simple_bind(ctx=mx.gpu(0), grad_req='write', **in_shapes)
  return exe
      

