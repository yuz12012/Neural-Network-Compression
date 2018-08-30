import mxnet as mx

def lenet_300_100(data, w, b, OUTPUTN):
  fc0  = mx.sym.FullyConnected(data=data, weight=w[0], bias=b[0], num_hidden=300)
  act0 = mx.sym.Activation(data=fc0, act_type="relu")
  fc1  = mx.sym.FullyConnected(data=act0, weight=w[1], bias=b[1], num_hidden=100)
  act1 = mx.sym.Activation(data=fc1, act_type="relu")
  fc2  = mx.sym.FullyConnected(data=act1, weight=w[2], bias=b[2], num_hidden=OUTPUTN)
  return fc2

def lenet_5(data, w, b, OUTPUTN):
  conv0 = mx.sym.Convolution(data=data, weight=w[0], bias=b[0], kernel=(5,5), num_filter=20)#, stride=(1,1))
  conv0 = mx.sym.Activation(data=conv0, act_type='relu')
  p0 = mx.sym.Pooling(data=conv0, kernel=(2,2), pool_type='max', stride=(2,2))
  conv1 = mx.sym.Convolution(data=p0, weight=w[1], bias=b[1], kernel=(5,5), num_filter=50)#, stride=(1,1))
  conv1 = mx.sym.Activation(data=conv1, act_type='relu')
  p1 = mx.sym.Pooling(data=conv1, kernel=(2,2), pool_type='max', stride=(2,2))
  f1 = mx.sym.Flatten(data=p1)
  fc0  = mx.sym.FullyConnected(data=f1, weight=w[2], bias=b[2], num_hidden=500)
  act0 = mx.sym.Activation(data=fc0, act_type="relu")
  fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  return fc1

def lenet_5_optfc(data, w, b, OUTPUTN):
  conv0 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20, stride=(1,1))
  p0 = mx.sym.Pooling(data=conv0, kernel=(2,2), pool_type='max', stride=(2,2))
  conv1 = mx.sym.Convolution(data=p0, kernel=(5,5), num_filter=50, stride=(1,1))
  p1 = mx.sym.Pooling(data=conv1, kernel=(2,2), pool_type='max', stride=(2,2))
  fc0  = mx.sym.FullyConnected(data=p1, weight=w[0], bias=b[0], num_hidden=500)
  act0 = mx.sym.Activation(data=fc0, act_type="relu")
  fc1  = mx.sym.FullyConnected(data=act0, weight=w[1], bias=b[1], num_hidden=OUTPUTN)
  return fc1

def lenet_300_100_neuron(data, w, b, OUTPUTN, starting_layer):
  if starting_layer == 1:
    fc1  = mx.sym.FullyConnected(data=data, weight=w[1], bias=b[1], num_hidden=100)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2  = mx.sym.FullyConnected(data=act1, weight=w[2], bias=b[2], num_hidden=OUTPUTN)
  elif starting_layer == 2:
    fc2  = mx.sym.FullyConnected(data=data, weight=w[2], bias=b[2], num_hidden=OUTPUTN)
  else:
    fc0  = mx.sym.FullyConnected(data=data, weight=w[0], bias=b[0], num_hidden=300)
    act0 = mx.sym.Activation(data=fc0, act_type="relu")
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[1], bias=b[1], num_hidden=100)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2  = mx.sym.FullyConnected(data=act1, weight=w[2], bias=b[2], num_hidden=OUTPUTN)
  return fc2

def lenet_5_neuron(data, w, b, OUTPUTN, starting_layer):
  if starting_layer == 1:
    conv1 = mx.sym.Convolution(data=data, weight=w[1], bias=b[1], kernel=(5,5), num_filter=50, stride=(1,1))
    p1 = mx.sym.Pooling(data=conv1, kernel=(2,2), pool_type='max', stride=(2,2))
    fc0  = mx.sym.FullyConnected(data=p1, weight=w[2], bias=b[2], num_hidden=500)
    act0 = mx.sym.Activation(data=fc0, act_type="relu")
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  elif starting_layer == 2:
    fc0  = mx.sym.FullyConnected(data=data, weight=w[2], bias=b[2], num_hidden=500)
    act0 = mx.sym.Activation(data=fc0, act_type="relu")
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  elif starting_layer == 3:
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  else:
    conv0 = mx.sym.Convolution(data=data, weight=w[0], bias=b[0], kernel=(5,5), num_filter=20, stride=(1,1))
    p0 = mx.sym.Pooling(data=conv0, kernel=(2,2), pool_type='max', stride=(2,2))
    conv1 = mx.sym.Convolution(data=p0, weight=w[1], bias=b[1], kernel=(5,5), num_filter=50, stride=(1,1))
    p1 = mx.sym.Pooling(data=conv1, kernel=(2,2), pool_type='max', stride=(2,2))
    fc0  = mx.sym.FullyConnected(data=p1, weight=w[2], bias=b[2], num_hidden=500)
    act0 = mx.sym.Activation(data=fc0, act_type="relu")
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  return fc1

def alexnet(data, w, OUTPUTN): 
  # on imagenet, data dim (10, 3, 227, 227) for imagenet, outputn = 1000
  # w has 8 elements
  # conv1
  conv1 = mx.sym.Convolution(data=data, weight=w[0], kernel=(11,11), num_filter=96, stride=(4,4))
  act1 = mx.sym.Activation(data=conv1, act_type="relu")
  lrn1 = mx.sym.LRN(data=act1, alpha=0.0001, beta=0.75, nsize=5)
  p1 = mx.sym.Pooling(data=lrn1, kernel=(3,3), pool_type='max', stride=(2,2))
  # conv2
  conv2 = mx.sym.Convolution(data=p1, weight=w[1], kernel=(5,5), num_filter=256, pad=(2,2), num_group=2)
  act2 = mx.sym.Activation(data=conv2, act_type="relu")
  lrn2 = mx.sym.LRN(data=act2, alpha=0.0001, beta=0.75, nsize=5)
  p2 = mx.sym.Pooling(data=lrn2, kernel=(3,3), pool_type='max', stride=(2,2))
  # conv3
  conv3 = mx.sym.Convolution(data=p2, weight=w[2], kernel=(3,3), num_filter=384, pad=(1,1))
  act3 = mx.sym.Activation(data=conv3, act_type="relu")
  # conv4
  conv4 = mx.sym.Convolution(data=act3, weight=w[3], kernel=(3,3), num_filter=384, pad=(1,1), num_group=2)
  act4 = mx.sym.Activation(data=conv4, act_type="relu")
  # conv5
  conv5 = mx.sym.Convolution(data=act4, weight=w[4], kernel=(3,3), num_filter=256, pad=(1,1), num_group=2)
  act5 = mx.sym.Activation(data=conv5, act_type="relu")
  p5 = mx.sym.Pooling(data=act5, kernel=(3,3), pool_type='max', stride=(2,2))
  # fc6
  fc6  = mx.sym.FullyConnected(data=p5, weight=w[5], num_hidden=4096)
  act6 = mx.sym.Activation(data=fc6, act_type="relu")
  drop6 = mx.sym.Dropout(data=act6, p=0.5)
  # fc7
  fc7  = mx.sym.FullyConnected(data=drop6, weight=w[6], num_hidden=4096)
  act7 = mx.sym.Activation(data=fc7, act_type="relu")
  drop7 = mx.sym.Dropout(data=act7, p=0.5)
  # fc8
  fc8  = mx.sym.FullyConnected(data=drop7, weight=w[7], num_hidden=OUTPUTN)
  return fc8

def vgg16(data, w, b, OUTPUTN): # w has 16 elements
  # conv1
  conv1_1 = mx.sym.Convolution(data=data, weight=w[0], bias=b[0], kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
  relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu", name="relu1_1")
  conv1_2 = mx.sym.Convolution(data=relu1_1, weight=w[1], bias=b[1], kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
  relu1_2 = mx.sym.Activation(data=conv1_2, act_type="relu", name="relu1_2")
  pool1   = mx.sym.Pooling(data=relu1_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool1")
  # conv2
  conv2_1 = mx.sym.Convolution(data=pool1, weight=w[2], bias=b[2], kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
  relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu", name="relu2_1")
  conv2_2 = mx.sym.Convolution(data=relu2_1, weight=w[3], bias=b[3], kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
  relu2_2 = mx.sym.Activation(data=conv2_2, act_type="relu", name="relu2_2")
  pool2   = mx.sym.Pooling(data=relu2_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool2")
  # conv3
  conv3_1 = mx.sym.Convolution(data=pool2, weight=w[4], bias=b[4], kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
  relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu", name="relu3_1")
  conv3_2 = mx.sym.Convolution(data=relu3_1, weight=w[5], bias=b[5], kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
  relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu", name="relu3_2")
  conv3_3 = mx.sym.Convolution(data=relu3_2, weight=w[6], bias=b[6], kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
  relu3_3 = mx.sym.Activation(data=conv3_3, act_type="relu", name="relu3_3")
  pool3   = mx.sym.Pooling(data=relu3_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool3")
  # conv4
  conv4_1 = mx.sym.Convolution(data=pool3, weight=w[7], bias=b[7], kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
  relu4_1 = mx.sym.Activation(data=conv4_1, act_type="relu", name="relu4_1")
  conv4_2 = mx.sym.Convolution(data=relu4_1, weight=w[8], bias=b[8], kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
  relu4_2 = mx.sym.Activation(data=conv4_2, act_type="relu", name="relu4_2")
  conv4_3 = mx.sym.Convolution(data=relu4_2, weight=w[9], bias=b[9], kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
  relu4_3 = mx.sym.Activation(data=conv4_3, act_type="relu", name="relu4_3")
  pool4   = mx.sym.Pooling(data=relu4_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")
  # conv5
  conv5_1 = mx.sym.Convolution(data=pool4, weight=w[10], bias=b[10], kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
  relu5_1 = mx.sym.Activation(data=conv5_1, act_type="relu", name="relu5_1")
  conv5_2 = mx.sym.Convolution(data=relu5_1, weight=w[11], bias=b[11], kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
  relu5_2 = mx.sym.Activation(data=conv5_2, act_type="relu", name="relu5_2")
  conv5_3 = mx.sym.Convolution(data=relu5_2, weight=w[12], bias=b[12], kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
  relu5_3 = mx.sym.Activation(data=conv5_3, act_type="relu", name="relu5_3")
  pool5   = mx.sym.Pooling(data=relu5_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool5")
  # fc6
  #flat6     = mx.sym.Flatten(data=pool5, name="flat6")
  fc6     = mx.sym.FullyConnected(data=pool5, weight=w[13], bias=b[13], num_hidden=4096, name="fc6")
  relu6   = mx.sym.Activation(data=fc6, act_type="relu", name="relu6")
  #drop6   = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")
  # fc7
  fc7     = mx.sym.FullyConnected(data=relu6, weight=w[14], bias=b[14], num_hidden=4096, name="fc7")
  relu7   = mx.sym.Activation(data=fc7, act_type="relu", name="relu7")
  #drop7   = mx.sym.Dropout(data=relu7, p=0.5, name="drop7")
  # fc8
  fc8     = mx.sym.FullyConnected(data=relu7, weight=w[15], bias=b[15], num_hidden=OUTPUTN, name="fc8")
  return fc8

def vgg_like_K(data, w, b, OUTPUTN, K):
  def conv_bn_relu(data, w, b, n):
    conv = mx.sym.Convolution(data=data, weight=w, bias=b, kernel=(3, 3), pad=(1, 1), num_filter=n)
    bn = mx.sym.BatchNorm(data=conv)
    relu = mx.sym.Activation(data=bn, act_type='relu')
    return relu
  # conv1
  cbr1_1 = conv_bn_relu(data, w[0], b[0], int(64*K))
  cbr1_2 = conv_bn_relu(cbr1_1, w[1], b[1], int(64*K))
  p1 = mx.sym.Pooling(data=cbr1_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool1")
  # conv2
  cbr2_1 = conv_bn_relu(p1, w[2], b[2], int(128*K))
  cbr2_2 = conv_bn_relu(cbr2_1, w[3], b[3], int(128*K))
  p2 = mx.sym.Pooling(data=cbr2_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool2")
  # conv3
  cbr3_1 = conv_bn_relu(p2, w[4], b[4], int(256*K))
  cbr3_2 = conv_bn_relu(cbr3_1, w[5], b[5], int(256*K))
  cbr3_3 = conv_bn_relu(cbr3_2, w[6], b[6], int(256*K))
  p3 = mx.sym.Pooling(data=cbr3_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool3")
  # conv4
  cbr4_1 = conv_bn_relu(p3, w[7], b[7], int(512*K))
  cbr4_2 = conv_bn_relu(cbr4_1, w[8], b[8], int(512*K))
  cbr4_3 = conv_bn_relu(cbr4_2, w[9], b[9], int(512*K))
  p4 = mx.sym.Pooling(data=cbr4_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")
  # conv5
  cbr5_1 = conv_bn_relu(p4, w[10], b[10], int(512*K))
  cbr5_2 = conv_bn_relu(cbr5_1, w[11], b[11], int(512*K))
  cbr5_3 = conv_bn_relu(cbr5_2, w[12], b[12], int(512*K))
  p5 = mx.sym.Pooling(data=cbr5_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")
  # fc6
  fc6 = mx.sym.FullyConnected(data=p5, weight=w[13], bias=b[13], num_hidden=int(512*K), name="fc6")
  bn6 = mx.sym.BatchNorm(data=fc6)
  relu6   = mx.sym.Activation(data=bn6, act_type="relu", name="relu6")
  # fc7
  fc7 = mx.sym.FullyConnected(data=relu6, weight=w[14], bias=b[14], num_hidden=OUTPUTN, name="fc7")
  return fc7

def vgg_like_dropout_K(data, w, b, OUTPUTN, K):
  def conv_bn_relu(data, w, b, n):
    conv = mx.sym.Convolution(data=data, weight=w, bias=b, kernel=(3, 3), pad=(1, 1), num_filter=n)
    bn = mx.sym.BatchNorm(data=conv)
    relu = mx.sym.Activation(data=bn, act_type='relu')
    return relu
  # conv1
  cbr1_1 = conv_bn_relu(data, w[0], b[0], int(64*K))
  drop1 = mx.sym.Dropout(data=cbr1_1, p=0.3)
  cbr1_2 = conv_bn_relu(drop1, w[1], b[1], int(64*K))
  p1 = mx.sym.Pooling(data=cbr1_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool1")
  # conv2
  cbr2_1 = conv_bn_relu(p1, w[2], b[2], int(128*K))
  drop2 = mx.sym.Dropout(data=cbr2_1, p=0.4)
  cbr2_2 = conv_bn_relu(drop2, w[3], b[3], int(128*K))
  p2 = mx.sym.Pooling(data=cbr2_2, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool2")
  # conv3
  cbr3_1 = conv_bn_relu(p2, w[4], b[4], int(256*K))
  drop3_1 = mx.sym.Dropout(data=cbr3_1, p=0.4)
  cbr3_2 = conv_bn_relu(drop3_1, w[5], b[5], int(256*K))
  drop3_2 = mx.sym.Dropout(data=cbr3_2, p=0.4)
  cbr3_3 = conv_bn_relu(drop3_2, w[6], b[6], int(256*K))
  p3 = mx.sym.Pooling(data=cbr3_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool3")
  # conv4
  cbr4_1 = conv_bn_relu(p3, w[7], b[7], int(512*K))
  drop4_1 = mx.sym.Dropout(data=cbr4_1, p=0.4)
  cbr4_2 = conv_bn_relu(drop4_1, w[8], b[8], int(512*K))
  drop4_2 = mx.sym.Dropout(data=cbr4_2, p=0.4)
  cbr4_3 = conv_bn_relu(drop4_2, w[9], b[9], int(512*K))
  p4 = mx.sym.Pooling(data=cbr4_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")
  # conv5
  cbr5_1 = conv_bn_relu(p4, w[10], b[10], int(512*K))
  drop5_1 = mx.sym.Dropout(data=cbr5_1, p=0.4)
  cbr5_2 = conv_bn_relu(drop5_1, w[11], b[11], int(512*K))
  drop5_2 = mx.sym.Dropout(data=cbr5_2, p=0.4)
  cbr5_3 = conv_bn_relu(drop5_2, w[12], b[12], int(512*K))
  p5 = mx.sym.Pooling(data=cbr5_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")
  # fc6
  fc6 = mx.sym.FullyConnected(data=p5, weight=w[13], bias=b[13], num_hidden=int(512*K), name="fc6")
  bn6 = mx.sym.BatchNorm(data=fc6)
  relu6   = mx.sym.Activation(data=bn6, act_type="relu", name="relu6")
  drop6 = mx.sym.Dropout(data=relu6, p=0.5)
  # fc7
  fc7 = mx.sym.FullyConnected(data=drop6, weight=w[14], bias=b[14], num_hidden=OUTPUTN, name="fc7")
  return fc7


