import mxnet as mx
import numpy as np
import sys
import NN
import Model
from Config import *
import Data
import time
# import logging
# logging.getLogger().setLevel(logging.NOTSET)  # logging to stdout

# get dataset
X_train, y_train, X_val, y_val = Data.get_data(DATA_NAME)
train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(X_val, y_val, batch_size)

t0 = time.time()
# --------------------- init train to get params -----------------------
acc_init = Model.init(train_iter, val_iter)
print 'init train acc is '+str(acc_init[0][1])
sgd_opt = mx.optimizer.SGD(learning_rate=LR_FORWARD, momentum=0.9, wd=0.0001, rescale_grad=(1.0/batch_size))
#adjusted_thres = np.copy(THRES)
sparsity = [1]*LAYER

# -------------------------- init params shapes ------------------------
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_INIT)
mask_w_total = []
mask_w_last = []
for j in range(LAYER):
  mask_w_total.append(mx.ndarray.ones(arg_params['w'+str(j)].shape, ctx=mx.gpu(0)))
  mask_w_last.append([None]*OUTPUTN)
  for i in range(OUTPUTN):
    mask_w_last[j][i] = mx.ndarray.ones(arg_params['w'+str(j)].shape, ctx=mx.gpu(0))
print '------------------ init time : '+str(time.time()-t0)
change_cnt = 0

if AUTO_SELECT == True:
  # --------------------------- auto select hyper params -----------------
  print THRES
  for j in range(LAYER):
    print '----------> start tuning layer '+str(j)
    while True:
      exe = Model.model_backward(j, arg_params)
      exe.copy_params_from(arg_params, aux_params, allow_extra_params=True)
      exe.arg_dict['id'][:] = mx.ndarray.ones((OUTPUTN,)) # all 1 vector
      exe.arg_dict['X'][:] = mx.ndarray.ones((1,)+X_SHAPE)
      for i in range(LAYER):
        if i != j:
          exe.arg_dict['mw'+str(i)][:] = mask_w_total[i]
      # train
      last_loss = 100000
      for i in range(STEP_BACKWARD):
        exe.forward()
        exe.backward()
        exe.arg_dict['w'+str(j)][:] -= STEPSIZE * exe.grad_dict['w'+str(j)]
        exe.arg_dict['w'+str(j)][:] = mx.ndarray.clip(exe.arg_dict['w'+str(j)][:], -1, 1)
        curr_loss = exe.output_dict.values()[0]
        if mx.ndarray.abs(curr_loss - last_loss) < epsilon: break
        last_loss = mx.ndarray.identity(curr_loss) 
      # prune
      thres = THRES[j]
      mask_w = mx.ndarray.abs(exe.arg_dict['w'+str(j)]) > thres
      while mx.ndarray.sum(mask_w) == 0 and thres > epsilon:
        thres = thres*0.5
        print 'change to '+str(thres)
        mask_w = mx.ndarray.abs(exe.arg_dict['w'+str(j)]) >= thres
      if mx.ndarray.sum(mask_w) == 0:
        print '******************** warning mask_w all 0 *********************'
      mask_w_total[j] = mask_w
      # forward
      model = Model.model_forward(mask_w_total) 
      for i in range(LAYER):
        if 'mw'+str(i) in arg_params.keys(): del arg_params['mw'+str(i)]
      # train
      model.fit(train_iter,  # train data
                eval_data=val_iter,  # validation data
                optimizer=sgd_opt, #'adam',  # use SGD to train
                eval_metric='acc',  # report accuracy during training
                arg_params=arg_params,
                allow_missing=True,
                num_epoch=STEP_FORWARD)  # train for at most 10 dataset passes
      acc_forward = model.score(val_iter, mx.metric.Accuracy())
      if acc_forward[0][1] >= acc_init[0][1] - THRES_STEP: THRES[j] += THRES_STEP
      elif acc_forward[0][1] < DROP_TO[j]: THRES[j] -= THRES_STEP
      else: break
      print 'curr acc '+str(acc_forward[0][1])+' and curr thres ' + str(THRES[j])
      mask_w_total = [mx.ndarray.ones(arg_params['w'+str(i)].shape, ctx=mx.gpu(0)) for i in range(LAYER)]
      sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_INIT)
  print THRES

# -------------------------------- big loop ----------------------------------
for K in range(TOTAL_K):
  print '=================> iter '+str(K)+' <==============='
  t_backward = time.time()
  j = LAYER-1
  total_diff = 0
  while j >= 0:
    # ----------------------------- start layerwise backward ------------------------
    exe = Model.model_backward(j, arg_params)
    # init all essential params
    exe.copy_params_from(arg_params, aux_params, allow_extra_params=True)
    #exe.arg_dict['id'][:] = mx.ndarray.array(np.identity(OUTPUTN))[target_out] # vector for picking one
    exe.arg_dict['id'][:] = mx.ndarray.ones((OUTPUTN,)) # all 1 vector
    exe.arg_dict['X'][:] = mx.ndarray.ones((1,)+X_SHAPE)
    for i in range(LAYER):
      if i != j:
        exe.arg_dict['mw'+str(i)][:] = mask_w_total[i]
    
    # train
    last_loss = 100000
    for i in range(STEP_BACKWARD):
      exe.forward()
      exe.backward()
      exe.arg_dict['w'+str(j)][:] -= STEPSIZE * exe.grad_dict['w'+str(j)]
      exe.arg_dict['w'+str(j)][:] = mx.ndarray.clip(exe.arg_dict['w'+str(j)][:], -1, 1)
      curr_loss = exe.output_dict.values()[0]
      if mx.ndarray.abs(curr_loss - last_loss) < epsilon:
        #print ' the loss is ' + str(curr_loss.asnumpy())
        break
      last_loss = mx.ndarray.identity(curr_loss) 
    
    # prune
    thres = THRES[j]
    mask_w = mx.ndarray.abs(exe.arg_dict['w'+str(j)]) > thres
    while mx.ndarray.sum(mask_w) == 0 and thres > epsilon:
      thres = thres*0.1
      mask_w = mx.ndarray.abs(exe.arg_dict['w'+str(j)]) >= thres
    if mx.ndarray.sum(mask_w) == 0:
      print '******************** warning mask_w all 0 *********************'
      mask_w = mx.ndarray.ones(arg_params['w'+str(j)].shape, ctx=mx.gpu(0))

    # compare
    wdiff = mx.ndarray.norm(mask_w - mask_w_last[j][target_out]) / mx.ndarray.norm(mask_w_last[j][target_out])
    total_diff += wdiff
    mask_w_last[j][target_out] = mx.ndarray.identity(mask_w)
    mask_w_total[j] = mask_w
    print ' diff for logit ' + str(target_out) + ' at layer '+ str(j) +' is ' + str(wdiff.asnumpy())
    #print '----------------------- finish a backward ------------------ backward time '+str(time.time() - t_backward)

    if j % GROUP_LAYERS == 0:
      # ------------------------- forward training ------------------------
      t_forward = time.time()
      model = Model.model_forward(mask_w_total) 
      # load and init all essential params
      if K == 0: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_INIT)
      else: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_FORWARD)
      for i in range(LAYER):
        if 'mw'+str(i) in arg_params.keys():
          del arg_params['mw'+str(i)]
      # train
      epochs = STEP_FORWARD 
      model.fit(train_iter,  # train data
                eval_data=val_iter,  # validation data
                optimizer=sgd_opt, #'adam',  # use SGD to train
                eval_metric='acc',  # report accuracy during training
                arg_params=arg_params,
                allow_missing=True,
                epoch_end_callback = mx.callback.do_checkpoint(prefix, period=STEP_FORWARD),
                num_epoch=epochs)  # train for at most 10 dataset passes
      #print '--------- finish forward step ------- forward time ' + str(time.time() - t_forward)
      # -------------------------------------------------------------------
  
      acc_forward = model.score(val_iter, mx.metric.Accuracy())
      #print acc_forward
      #print '########## running time '+str(time.time()-t0)
      sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_FORWARD)

    j = j -1 
      
  print ' --------------- finish a full loop, time ' + str(time.time() - t_backward) + '------- acc ' + str(acc_forward[0][1])
  if total_diff <= STOP_DIFF and K - change_cnt > 5:
    break
  nnz = 0.
  total = 0.
  for j in range(LAYER):
    sparsity[j] = mx.ndarray.sum(mask_w_total[j]).asnumpy() / mask_w_total[j].size
    print 'layer '+str(j)+' size = ' + str(mask_w_total[j].size) + ' sparsity ' + str(1-sparsity[j])
    nnz += mx.ndarray.sum(mask_w_total[j])
    total += mask_w_total[j].size
  print 'full compress ratio = ' + str(1/(nnz/total).asnumpy())
  # -------------------------------------------------------------------
  # prepare for next backward
  print ' ========================================================================'

# ------------------------- post training ------------------------
t_post = time.time()
model = Model.model_forward(mask_w_total) 
# load and init all essential params
for j in range(LAYER):
  if 'mw'+str(j) in arg_params.keys():
    del arg_params['mw'+str(j)]
# train
epochs = STEP_POST
sgd_opt = mx.optimizer.SGD(learning_rate=LR_POST, momentum=0.9, wd=0.0001, rescale_grad=(1.0/batch_size))
def lr_callback(epoch, *args):
   if epoch % 50 == 0 and epoch > 0 and sgd_opt.lr >= 1e-8:
     sgd_opt.lr /= 10 # decrease learning rate by a factor of 10 every 10 batches
     print('change learning rate to '+str(sgd_opt.lr))
   print('nbatch:%d, learning rate:%f' % (epoch, sgd_opt.lr))

model.fit(train_iter,  # train data
          eval_data=val_iter,  # validation data
          optimizer=sgd_opt, #ALG,  # use SGD to train
          #optimizer_params={'learning_rate':lr},  # use fixed learning rate
          eval_metric='acc',  # report accuracy during training
          arg_params=arg_params,
          allow_missing=True,
          epoch_end_callback = [mx.callback.do_checkpoint(prefix, period=STEP_FORWARD), lr_callback],
          num_epoch=epochs)  # train for at most 10 dataset passes
print model.score(val_iter, mx.metric.Accuracy())
print '----------- finish post step ------------ post time: ' + str(time.time() - t_post)
# -------------------------------------------------------------------
print '------------ total time '+str(time.time() - t0)
print THRES

