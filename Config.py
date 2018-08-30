# NN and data parameters

DATA_NAME = 'cifar10' # cifar10, mnist
X_SHAPE = (3, 32, 32) # cifar10: (3,32,32); mnist: (1,784)
# DATA_NAME = 'mnist'
# X_SHAPE = (1, 784)
NN_NAME = 'NN.vgg_like_K(data, w, b, OUTPUTN, 1.0)'
#NN_NAME = 'NN.lenet_5(data, w, b, OUTPUTN)'
#NN_NAME = 'NN.lenet_5(data, w, b, OUTPUTN)'
prefix = 'checkpoints/vgg_like_dropout_K1'
#prefix = 'checkpoints/vgg_like_K025'
#prefix = 'checkpoints/lenet5'
ALG = 'sgd'
warm_prefix = 'pretrained/cifar10/vgg_like_dropout_K1' 
#warm_prefix = 'pretrained/mnist/lenet5' 
LAYER = 15

# hyper parameters
OUTPUTN = 10
IF_WARMSTART = False
AUTO_SELECT = False
WARM_EP = 1600
SPARSITY_PENALTY = 1e3
STEP_INIT = 500
STEP_POST = 50
STEP_FORWARD = 1
STEP_BACKWARD = 80000
target_out = 0
epsilon = 1e-4
batch_size = 100
#THRES = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2]
THRES = [0.02]*LAYER
GROUP_LAYERS = 5
DROP_TO = [0.9]*LAYER
THRES_STEP = 0.001
STEPSIZE = 5e-2
LR_INIT = 1e-2
LR_FORWARD = 3e-3
LR_POST = 1e-5
STOP_DIFF = 0.00
TOTAL_K = int(150/ (LAYER/GROUP_LAYERS*STEP_FORWARD))


# some default values
if prefix == 'checkpoints/vgg_like_dropout_K1': 
  DATA_NAME == 'cifar10'
  X_SHAPE = (3, 32, 32)
  #THRES = [0.85, 0.45, 0.3, 0.18, 0.2, 0.19, 0.08, 0.14, 0.076, 0.055, 0.099, 0.098, 0.072, 0.099, 0.32]
  #THRES = [0.8, 0.4, 0.28, 0.17, 0.18, 0.18, 0.078, 0.13, 0.07, 0.05, 0.09, 0.09, 0.07, 0.09, 0.3]
  THRES = [0.018]*LAYER
  GROUP_LAYERS = 5
  DROP_TO = [0.88]*LAYER
  STOP_DIFF = 0.1

if prefix == 'checkpoints/vgg_like_dropout_K025': 
  DATA_NAME == 'cifar10'
  X_SHAPE = (3, 32, 32)
  #THRES = [0.9, 0.4, 0.3, 0.18, 0.2, 0.19, 0.08, 0.14, 0.076, 0.055, 0.099, 0.098, 0.1, 0.17, 0.31]
  THRES = [0.055]*LAYER
  GROUP_LAYERS = 5
  DROP_TO = [0.8]*LAYER
  STOP_DIFF = 0.1

if prefix == 'checkpoints/vgg_like_dropout_K075': 
  DATA_NAME == 'cifar10'
  X_SHAPE = (3, 32, 32)
  #THRES = [0.9, 0.4, 0.3, 0.18, 0.2, 0.19, 0.08, 0.14, 0.076, 0.055, 0.099, 0.098, 0.1, 0.17, 0.31]
  THRES = [0.036]*LAYER
  GROUP_LAYERS = 5
  DROP_TO = [0.8]*LAYER
  STOP_DIFF = 0.1

if prefix == 'checkpoints/vgg_like_K025': 
  DATA_NAME == 'cifar10'
  X_SHAPE = (3, 32, 32)
  THRES = [0.85, 0.45, 0.3, 0.18, 0.2, 0.19, 0.08, 0.14, 0.076, 0.055, 0.099, 0.098, 0.072, 0.099, 0.32]
  GROUP_LAYERS = 5
  DROP_TO = [0.845]*LAYER
  STOP_DIFF = 0.1

if prefix == 'checkpoints/lenet31': 
  SPARSITY_PENALTY = 1
  DATA_NAME = 'mnist'
  X_SHAPE = (1, 784)
  LAYER = 3
  #THRES = [0.09, 0.16, 0.59]#[0.1]*LAYER
  THRES = [0.09, 0.16, 0.59]#0.98#[0.1]*LAYER
  warm_prefix = 'warm_start/lenet31' 
  WARM_EP = 300
  DROP_TO = [0.98]*LAYER
  GROUP_LAYERS = 3
  STEP_FORWARD = 3
  STEP_POST = 10
  LR_FORWARD = 3e-3
  LR_POST = 5e-6
  TOTAL_K = int(1000/ (LAYER/GROUP_LAYERS*STEP_FORWARD))
  STOP_DIFF = 0.01

if prefix == 'checkpoints/lenet5': # and IF_WARMSTART == True: 
  X_SHAPE = (1, 28, 28)
  LAYER = 4
  THRES = [0.42, 0.12, 0.08, 0.18] #0.99
  warm_prefix = 'warm_start/lenet5' 
  WARM_EP = 300
  DROP_TO = [0.991]*LAYER
  GROUP_LAYERS = 4
  STEP_INIT = 100
  STEP_FORWARD = 1
  STEP_POST = 100
  LR_FORWARD = 1e-3
  LR_POST = 5e-6
  TOTAL_K = int(300/ (LAYER/GROUP_LAYERS*STEP_FORWARD))
  STOP_DIFF = 0.001

