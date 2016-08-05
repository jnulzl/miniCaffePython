#encoding:utf-8

#说明
'''
1、实际中使用本例子的前提是对Caffe的命令行运行方式
以及相关配置文件有个大概了解才行。
2、本例子只是用了Caffe自带的配置文件和网络结构
用的数据并不是mnist数据。
'''

#1、生成自己的图像标签,一般用txt保存起来
'''
该步骤，根据个人实际情况以及习惯可手工或者编程实现。
这里，我的图像标签保存在"train.txt"文件夹下面，内容如下：
...
N1806147914_1.png 0
N1806148214_1.png 0
N1806148514_1.png 0
N1806149114_1.png 0
N1806111695_1.png 1
N1806111750_1.png 1
N1806111805_1.png 1
N1806122433_1.png 1
N1806122660_1.png 1
N1806151284_1.png 1
N1806151346_1.png 1
N1806151456_1.png 1
...
'''

#2、将图像数据(train数据和test数据)转化为Caffe可用的格式，一般为lmdb

import caffePython
'''
#我的没有GPU，所以resize的图像不能太大，太大的话训练时间太长了
resizeHeight = 32
resizeWidth = 32

imagePath = "./img/"#原始图像保存在当前文件夹下的"img"文件夹中
imageLabelsList = "./train.txt"#第一步得到的图像标签列表
LMDBDir = 	"./train_lmdb"#保存转换后的lmdb数据文件夹
	
caffePython.ImagesToLMDB(resizeHeight,resizeWidth,imagePath,imageLabelsList,LMDBDir,False,True)
'''

#3、计算所有图像像素数据的均值
'''
#这里只是演示，暂时不需要计算图像均值
LMDBDir = 	"./train_lmdb"
MeanFile = "./coiss384Mean"
	
caffePython.computerImagesMean(LMDBDir,MeanFile)
'''

#4,5、编写配置文件和网络结构文件

'''
这里我们用的是caffe自带的mnist数字分类识别，
Solver文件为:lenet_solver.prototxt
网络结构文件为:lenet_train_test.prototxt
'''

#6、训练网络

'''
因为训练网络的过程中也包含阶段性测试,因此也需要测试数据，
这里为了简便，我们用的test数据与train数据一样，包含在test_lmdb文件夹下面
实际中的test数据处理方式与train数据的处理方式一样
'''
solverFilePath = "./lenet_solver.prototxt"

caffePython.trainNet(solverFilePath)

#训练日志如下：
'''
I0805 02:18:36.326828  7288 caffe.cpp:179] Use CPU.
I0805 02:18:36.326828  7288 solver.cpp:48] Initializing solver from parameters:
test_iter: 10
test_interval: 1
base_lr: 0.01
display: 20
max_iter: 100
lr_policy: "inv"
gamma: 0.0001
power: 0.75
momentum: 0.9
weight_decay: 0.0005
snapshot: 50
snapshot_prefix: "lenet"
solver_mode: CPU
net: "lenet_train_test.prototxt"
I0805 02:18:36.326828  7288 solver.cpp:91] Creating training net from net file: lenet_train_tes             t.prototxt
I0805 02:18:36.326828  7288 net.cpp:313] The NetState phase (0) differed from the phase (1) spe             cified by a rule in layer mnist
I0805 02:18:36.326828  7288 net.cpp:313] The NetState phase (0) differed from the phase (1) spe             cified by a rule in layer accuracy
I0805 02:18:36.326828  7288 net.cpp:49] Initializing net from parameters:
name: "LeNet"
state {
  phase: TRAIN
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "train_lmdb"
    batch_size: 38
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 250
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
I0805 02:18:36.326828  7288 layer_factory.hpp:77] Creating layer mnist
I0805 02:18:36.326828  7288 common.cpp:36] System entropy source not available, using fallback              algorithm to generate seed instead.
I0805 02:18:36.326828  7288 net.cpp:91] Creating Layer mnist
I0805 02:18:36.326828  7288 net.cpp:399] mnist -> data
I0805 02:18:36.326828  7288 net.cpp:399] mnist -> label
I0805 02:18:36.326828  1520 db_lmdb.cpp:40] Opened lmdb train_lmdb
I0805 02:18:36.326828  7288 data_layer.cpp:41] output data size: 38,3,32,32
I0805 02:18:36.326828  7288 net.cpp:141] Setting up mnist
I0805 02:18:36.326828  7288 net.cpp:148] Top shape: 38 3 32 32 (116736)
I0805 02:18:36.326828  7288 net.cpp:148] Top shape: 38 (38)
I0805 02:18:36.326828  7288 net.cpp:156] Memory required for data: 467096
I0805 02:18:36.326828  7288 layer_factory.hpp:77] Creating layer conv1
I0805 02:18:36.326828  7288 net.cpp:91] Creating Layer conv1
I0805 02:18:36.326828  7288 net.cpp:425] conv1 <- data
I0805 02:18:36.326828  7288 net.cpp:399] conv1 -> conv1
I0805 02:18:36.326828  7288 net.cpp:141] Setting up conv1
I0805 02:18:36.326828  7288 net.cpp:148] Top shape: 38 20 28 28 (595840)
I0805 02:18:36.326828  7288 net.cpp:156] Memory required for data: 2850456
I0805 02:18:36.326828  7288 layer_factory.hpp:77] Creating layer pool1
I0805 02:18:36.326828  7288 net.cpp:91] Creating Layer pool1
I0805 02:18:36.326828  7288 net.cpp:425] pool1 <- conv1
I0805 02:18:36.326828  7288 net.cpp:399] pool1 -> pool1
I0805 02:18:36.326828  7288 net.cpp:141] Setting up pool1
I0805 02:18:36.326828  7288 net.cpp:148] Top shape: 38 20 14 14 (148960)
I0805 02:18:36.326828  7288 net.cpp:156] Memory required for data: 3446296
I0805 02:18:36.326828  7288 layer_factory.hpp:77] Creating layer conv2
I0805 02:18:36.326828  7288 net.cpp:91] Creating Layer conv2
I0805 02:18:36.326828  7288 net.cpp:425] conv2 <- pool1
I0805 02:18:36.326828  7288 net.cpp:399] conv2 -> conv2
I0805 02:18:36.326828  7288 net.cpp:141] Setting up conv2
I0805 02:18:36.326828  7288 net.cpp:148] Top shape: 38 50 10 10 (190000)
I0805 02:18:36.326828  7288 net.cpp:156] Memory required for data: 4206296
I0805 02:18:36.326828  7288 layer_factory.hpp:77] Creating layer pool2
I0805 02:18:36.326828  7288 net.cpp:91] Creating Layer pool2
I0805 02:18:36.326828  7288 net.cpp:425] pool2 <- conv2
I0805 02:18:36.326828  7288 net.cpp:399] pool2 -> pool2
I0805 02:18:36.326828  7288 net.cpp:141] Setting up pool2
I0805 02:18:36.326828  7288 net.cpp:148] Top shape: 38 50 5 5 (47500)
I0805 02:18:36.326828  7288 net.cpp:156] Memory required for data: 4396296
I0805 02:18:36.326828  7288 layer_factory.hpp:77] Creating layer ip1
I0805 02:18:36.326828  7288 net.cpp:91] Creating Layer ip1
I0805 02:18:36.326828  7288 net.cpp:425] ip1 <- pool2
I0805 02:18:36.326828  7288 net.cpp:399] ip1 -> ip1
I0805 02:18:36.342453  7288 net.cpp:141] Setting up ip1
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 250 (9500)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4434296
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer relu1
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer relu1
I0805 02:18:36.342453  7288 net.cpp:425] relu1 <- ip1
I0805 02:18:36.342453  7288 net.cpp:386] relu1 -> ip1 (in-place)
I0805 02:18:36.342453  7288 net.cpp:141] Setting up relu1
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 250 (9500)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4472296
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer ip2
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer ip2
I0805 02:18:36.342453  7288 net.cpp:425] ip2 <- ip1
I0805 02:18:36.342453  7288 net.cpp:399] ip2 -> ip2
I0805 02:18:36.342453  7288 net.cpp:141] Setting up ip2
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 2 (76)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4472600
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer loss
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer loss
I0805 02:18:36.342453  7288 net.cpp:425] loss <- ip2
I0805 02:18:36.342453  7288 net.cpp:425] loss <- label
I0805 02:18:36.342453  7288 net.cpp:399] loss -> loss
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer loss
I0805 02:18:36.342453  7288 net.cpp:141] Setting up loss
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: (1)
I0805 02:18:36.342453  7288 net.cpp:151]     with loss weight 1
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4472604
I0805 02:18:36.342453  7288 net.cpp:217] loss needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] ip2 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] relu1 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] ip1 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] pool2 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] conv2 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] pool1 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] conv1 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:219] mnist does not need backward computation.
I0805 02:18:36.342453  7288 net.cpp:261] This network produces output loss
I0805 02:18:36.342453  7288 net.cpp:274] Network initialization done.
I0805 02:18:36.342453  7288 solver.cpp:181] Creating test net (#0) specified by net file: lenet             _train_test.prototxt
I0805 02:18:36.342453  7288 net.cpp:313] The NetState phase (1) differed from the phase (0) spe             cified by a rule in layer mnist
I0805 02:18:36.342453  7288 net.cpp:49] Initializing net from parameters:
name: "LeNet"
state {
  phase: TEST
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "test_lmdb"
    batch_size: 38
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 250
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer mnist
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer mnist
I0805 02:18:36.342453  7288 net.cpp:399] mnist -> data
I0805 02:18:36.342453  7288 net.cpp:399] mnist -> label
I0805 02:18:36.342453  3604 db_lmdb.cpp:40] Opened lmdb test_lmdb
I0805 02:18:36.342453  7288 data_layer.cpp:41] output data size: 38,3,32,32
I0805 02:18:36.342453  7288 net.cpp:141] Setting up mnist
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 3 32 32 (116736)
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 (38)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 467096
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer label_mnist_1_split
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer label_mnist_1_split
I0805 02:18:36.342453  7288 net.cpp:425] label_mnist_1_split <- label
I0805 02:18:36.342453  7288 net.cpp:399] label_mnist_1_split -> label_mnist_1_split_0
I0805 02:18:36.342453  7288 net.cpp:399] label_mnist_1_split -> label_mnist_1_split_1
I0805 02:18:36.342453  7288 net.cpp:141] Setting up label_mnist_1_split
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 (38)
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 (38)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 467400
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer conv1
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer conv1
I0805 02:18:36.342453  7288 net.cpp:425] conv1 <- data
I0805 02:18:36.342453  7288 net.cpp:399] conv1 -> conv1
I0805 02:18:36.342453  7288 net.cpp:141] Setting up conv1
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 20 28 28 (595840)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 2850760
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer pool1
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer pool1
I0805 02:18:36.342453  7288 net.cpp:425] pool1 <- conv1
I0805 02:18:36.342453  7288 net.cpp:399] pool1 -> pool1
I0805 02:18:36.342453  7288 net.cpp:141] Setting up pool1
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 20 14 14 (148960)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 3446600
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer conv2
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer conv2
I0805 02:18:36.342453  7288 net.cpp:425] conv2 <- pool1
I0805 02:18:36.342453  7288 net.cpp:399] conv2 -> conv2
I0805 02:18:36.342453  7288 net.cpp:141] Setting up conv2
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 50 10 10 (190000)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4206600
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer pool2
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer pool2
I0805 02:18:36.342453  7288 net.cpp:425] pool2 <- conv2
I0805 02:18:36.342453  7288 net.cpp:399] pool2 -> pool2
I0805 02:18:36.342453  7288 net.cpp:141] Setting up pool2
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 50 5 5 (47500)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4396600
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer ip1
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer ip1
I0805 02:18:36.342453  7288 net.cpp:425] ip1 <- pool2
I0805 02:18:36.342453  7288 net.cpp:399] ip1 -> ip1
I0805 02:18:36.342453  7288 net.cpp:141] Setting up ip1
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 250 (9500)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4434600
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer relu1
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer relu1
I0805 02:18:36.342453  7288 net.cpp:425] relu1 <- ip1
I0805 02:18:36.342453  7288 net.cpp:386] relu1 -> ip1 (in-place)
I0805 02:18:36.342453  7288 net.cpp:141] Setting up relu1
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 250 (9500)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4472600
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer ip2
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer ip2
I0805 02:18:36.342453  7288 net.cpp:425] ip2 <- ip1
I0805 02:18:36.342453  7288 net.cpp:399] ip2 -> ip2
I0805 02:18:36.342453  7288 net.cpp:141] Setting up ip2
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 2 (76)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4472904
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer ip2_ip2_0_split
I0805 02:18:36.342453  7288 net.cpp:425] ip2_ip2_0_split <- ip2
I0805 02:18:36.342453  7288 net.cpp:399] ip2_ip2_0_split -> ip2_ip2_0_split_0
I0805 02:18:36.342453  7288 net.cpp:399] ip2_ip2_0_split -> ip2_ip2_0_split_1
I0805 02:18:36.342453  7288 net.cpp:141] Setting up ip2_ip2_0_split
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 2 (76)
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: 38 2 (76)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4473512
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer accuracy
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer accuracy
I0805 02:18:36.342453  7288 net.cpp:425] accuracy <- ip2_ip2_0_split_0
I0805 02:18:36.342453  7288 net.cpp:425] accuracy <- label_mnist_1_split_0
I0805 02:18:36.342453  7288 net.cpp:399] accuracy -> accuracy
I0805 02:18:36.342453  7288 net.cpp:141] Setting up accuracy
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: (1)
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4473516
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer loss
I0805 02:18:36.342453  7288 net.cpp:91] Creating Layer loss
I0805 02:18:36.342453  7288 net.cpp:425] loss <- ip2_ip2_0_split_1
I0805 02:18:36.342453  7288 net.cpp:425] loss <- label_mnist_1_split_1
I0805 02:18:36.342453  7288 net.cpp:399] loss -> loss
I0805 02:18:36.342453  7288 layer_factory.hpp:77] Creating layer loss
I0805 02:18:36.342453  7288 net.cpp:141] Setting up loss
I0805 02:18:36.342453  7288 net.cpp:148] Top shape: (1)
I0805 02:18:36.342453  7288 net.cpp:151]     with loss weight 1
I0805 02:18:36.342453  7288 net.cpp:156] Memory required for data: 4473520
I0805 02:18:36.342453  7288 net.cpp:217] loss needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:219] accuracy does not need backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] ip2_ip2_0_split needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] ip2 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] relu1 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] ip1 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] pool2 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] conv2 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] pool1 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:217] conv1 needs backward computation.
I0805 02:18:36.342453  7288 net.cpp:219] label_mnist_1_split does not need backward computation             .
I0805 02:18:36.342453  7288 net.cpp:219] mnist does not need backward computation.
I0805 02:18:36.342453  7288 net.cpp:261] This network produces output accuracy
I0805 02:18:36.342453  7288 net.cpp:261] This network produces output loss
I0805 02:18:36.342453  7288 net.cpp:274] Network initialization done.
I0805 02:18:36.342453  7288 solver.cpp:60] Solver scaffolding done.
I0805 02:18:36.342453  7288 caffe.cpp:220] Starting Optimization
I0805 02:18:36.342453  7288 solver.cpp:279] Solving LeNet
I0805 02:18:36.342453  7288 solver.cpp:280] Learning Rate Policy: inv
I0805 02:18:36.342453  7288 solver.cpp:337] Iteration 0, Testing net (#0)
I0805 02:18:36.711809  7288 solver.cpp:404]     Test net output #0: accuracy = 0.134211
I0805 02:18:36.711809  7288 solver.cpp:404]     Test net output #1: loss = 0.706133 (* 1 = 0.70             6133 loss)
I0805 02:18:36.796450  7288 solver.cpp:228] Iteration 0, loss = 0.71298
I0805 02:18:36.796450  7288 solver.cpp:244]     Train net output #0: loss = 0.71298 (* 1 = 0.71             298 loss)
I0805 02:18:36.796450  7288 sgd_solver.cpp:106] Iteration 0, lr = 0.01
I0805 02:18:36.796450  7288 solver.cpp:337] Iteration 1, Testing net (#0)
I0805 02:18:37.159715  7288 solver.cpp:404]     Test net output #0: accuracy = 0.85
I0805 02:18:37.159715  7288 solver.cpp:404]     Test net output #1: loss = 0.652425 (* 1 = 0.65             2425 loss)
I0805 02:18:37.227823  7288 solver.cpp:337] Iteration 2, Testing net (#0)
I0805 02:18:37.643410  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:37.643410  7288 solver.cpp:404]     Test net output #1: loss = 0.570612 (* 1 = 0.57             0612 loss)
I0805 02:18:37.727704  7288 solver.cpp:337] Iteration 3, Testing net (#0)
I0805 02:18:38.080926  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:38.080926  7288 solver.cpp:404]     Test net output #1: loss = 0.488094 (* 1 = 0.48             8094 loss)
I0805 02:18:38.164567  7288 solver.cpp:337] Iteration 4, Testing net (#0)
I0805 02:18:38.512850  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:38.512850  7288 solver.cpp:404]     Test net output #1: loss = 0.41366 (* 1 = 0.413             66 loss)
I0805 02:18:38.597091  7288 solver.cpp:337] Iteration 5, Testing net (#0)
I0805 02:18:38.944263  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:38.944263  7288 solver.cpp:404]     Test net output #1: loss = 0.355491 (* 1 = 0.35             5491 loss)
I0805 02:18:39.028643  7288 solver.cpp:337] Iteration 6, Testing net (#0)
I0805 02:18:39.397848  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:39.397848  7288 solver.cpp:404]     Test net output #1: loss = 0.311008 (* 1 = 0.31             1008 loss)
I0805 02:18:39.482331  7288 solver.cpp:337] Iteration 7, Testing net (#0)
I0805 02:18:39.829668  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:39.829668  7288 solver.cpp:404]     Test net output #1: loss = 0.280745 (* 1 = 0.28             0745 loss)
I0805 02:18:39.914149  7288 solver.cpp:337] Iteration 8, Testing net (#0)
I0805 02:18:40.267833  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:40.267833  7288 solver.cpp:404]     Test net output #1: loss = 0.261368 (* 1 = 0.26             1368 loss)
I0805 02:18:40.345966  7288 solver.cpp:337] Iteration 9, Testing net (#0)
I0805 02:18:40.699921  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:40.699921  7288 solver.cpp:404]     Test net output #1: loss = 0.24963 (* 1 = 0.249             63 loss)
I0805 02:18:40.784170  7288 solver.cpp:337] Iteration 10, Testing net (#0)
I0805 02:18:41.131491  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:41.131491  7288 solver.cpp:404]     Test net output #1: loss = 0.244273 (* 1 = 0.24             4273 loss)
I0805 02:18:41.200495  7288 solver.cpp:337] Iteration 11, Testing net (#0)
I0805 02:18:41.569164  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:41.569164  7288 solver.cpp:404]     Test net output #1: loss = 0.241819 (* 1 = 0.24             1819 loss)
I0805 02:18:41.647404  7288 solver.cpp:337] Iteration 12, Testing net (#0)
I0805 02:18:42.000494  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:42.000494  7288 solver.cpp:404]     Test net output #1: loss = 0.240104 (* 1 = 0.24             0104 loss)
I0805 02:18:42.085134  7288 solver.cpp:337] Iteration 13, Testing net (#0)
I0805 02:18:42.448127  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:42.448127  7288 solver.cpp:404]     Test net output #1: loss = 0.23463 (* 1 = 0.234             63 loss)
I0805 02:18:42.516705  7288 solver.cpp:337] Iteration 14, Testing net (#0)
I0805 02:18:42.869765  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:42.869765  7288 solver.cpp:404]     Test net output #1: loss = 0.228542 (* 1 = 0.22             8542 loss)
I0805 02:18:42.948124  7288 solver.cpp:337] Iteration 15, Testing net (#0)
I0805 02:18:43.316798  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:43.316798  7288 solver.cpp:404]     Test net output #1: loss = 0.218959 (* 1 = 0.21             8959 loss)
I0805 02:18:43.385740  7288 solver.cpp:337] Iteration 16, Testing net (#0)
I0805 02:18:43.748456  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:43.748456  7288 solver.cpp:404]     Test net output #1: loss = 0.208029 (* 1 = 0.20             8029 loss)
I0805 02:18:43.833097  7288 solver.cpp:337] Iteration 17, Testing net (#0)
I0805 02:18:44.186846  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:44.186846  7288 solver.cpp:404]     Test net output #1: loss = 0.19398 (* 1 = 0.193             98 loss)
I0805 02:18:44.270984  7288 solver.cpp:337] Iteration 18, Testing net (#0)
I0805 02:18:44.618903  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:44.618903  7288 solver.cpp:404]     Test net output #1: loss = 0.180609 (* 1 = 0.18             0609 loss)
I0805 02:18:44.687927  7288 solver.cpp:337] Iteration 19, Testing net (#0)
I0805 02:18:45.050699  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:45.050699  7288 solver.cpp:404]     Test net output #1: loss = 0.169522 (* 1 = 0.16             9522 loss)
I0805 02:18:45.135320  7288 solver.cpp:337] Iteration 20, Testing net (#0)
I0805 02:18:45.488168  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:45.488168  7288 solver.cpp:404]     Test net output #1: loss = 0.162909 (* 1 = 0.16             2909 loss)
I0805 02:18:45.573308  7288 solver.cpp:228] Iteration 20, loss = 0.149419
I0805 02:18:45.573308  7288 solver.cpp:244]     Train net output #0: loss = 0.149419 (* 1 = 0.1             49419 loss)
I0805 02:18:45.573308  7288 sgd_solver.cpp:106] Iteration 20, lr = 0.00998503
I0805 02:18:45.573308  7288 solver.cpp:337] Iteration 21, Testing net (#0)
I0805 02:18:45.935848  7288 solver.cpp:404]     Test net output #0: accuracy = 0.855263
I0805 02:18:45.935848  7288 solver.cpp:404]     Test net output #1: loss = 0.158318 (* 1 = 0.15             8318 loss)
I0805 02:18:46.004732  7288 solver.cpp:337] Iteration 22, Testing net (#0)
I0805 02:18:46.351954  7288 solver.cpp:404]     Test net output #0: accuracy = 0.857895
I0805 02:18:46.351954  7288 solver.cpp:404]     Test net output #1: loss = 0.154819 (* 1 = 0.154819 loss)
I0805 02:18:46.452035  7288 solver.cpp:337] Iteration 23, Testing net (#0)
I0805 02:18:46.806040  7288 solver.cpp:404]     Test net output #0: accuracy = 0.926316
I0805 02:18:46.806040  7288 solver.cpp:404]     Test net output #1: loss = 0.151212 (* 1 = 0.151212 loss)
I0805 02:18:46.874550  7288 solver.cpp:337] Iteration 24, Testing net (#0)
I0805 02:18:47.270892  7288 solver.cpp:404]     Test net output #0: accuracy = 0.98421
I0805 02:18:47.270892  7288 solver.cpp:404]     Test net output #1: loss = 0.147982 (* 1 = 0.147982 loss)
I0805 02:18:47.353529  7288 solver.cpp:337] Iteration 25, Testing net (#0)
I0805 02:18:47.738582  7288 solver.cpp:404]     Test net output #0: accuracy = 0.989474
I0805 02:18:47.738582  7288 solver.cpp:404]     Test net output #1: loss = 0.144446 (* 1 = 0.144446 loss)
I0805 02:18:47.822798  7288 solver.cpp:337] Iteration 26, Testing net (#0)
I0805 02:18:48.175756  7288 solver.cpp:404]     Test net output #0: accuracy = 0.986842
I0805 02:18:48.175756  7288 solver.cpp:404]     Test net output #1: loss = 0.14067 (* 1 = 0.14067 loss)
I0805 02:18:48.239269  7288 solver.cpp:337] Iteration 27, Testing net (#0)
I0805 02:18:48.608799  7288 solver.cpp:404]     Test net output #0: accuracy = 0.978947
I0805 02:18:48.608799  7288 solver.cpp:404]     Test net output #1: loss = 0.136304 (* 1 = 0.136304 loss)
I0805 02:18:48.677310  7288 solver.cpp:337] Iteration 28, Testing net (#0)
I0805 02:18:49.055769  7288 solver.cpp:404]     Test net output #0: accuracy = 0.968421
I0805 02:18:49.055769  7288 solver.cpp:404]     Test net output #1: loss = 0.131599 (* 1 = 0.131599 loss)
I0805 02:18:49.124789  7288 solver.cpp:337] Iteration 29, Testing net (#0)
I0805 02:18:49.477285  7288 solver.cpp:404]     Test net output #0: accuracy = 0.971053
I0805 02:18:49.477285  7288 solver.cpp:404]     Test net output #1: loss = 0.126558 (* 1 = 0.126558 loss)
I0805 02:18:49.572376  7288 solver.cpp:337] Iteration 30, Testing net (#0)
I0805 02:18:49.909014  7288 solver.cpp:404]     Test net output #0: accuracy = 0.971053
I0805 02:18:49.909014  7288 solver.cpp:404]     Test net output #1: loss = 0.121519 (* 1 = 0.121519 loss)
I0805 02:18:49.993499  7288 solver.cpp:337] Iteration 31, Testing net (#0)
I0805 02:18:50.356427  7288 solver.cpp:404]     Test net output #0: accuracy = 0.971053
I0805 02:18:50.356427  7288 solver.cpp:404]     Test net output #1: loss = 0.116793 (* 1 = 0.116793 loss)
I0805 02:18:50.425442  7288 solver.cpp:337] Iteration 32, Testing net (#0)
I0805 02:18:50.795011  7288 solver.cpp:404]     Test net output #0: accuracy = 0.971053
I0805 02:18:50.795011  7288 solver.cpp:404]     Test net output #1: loss = 0.111852 (* 1 = 0.111852 loss)
I0805 02:18:50.879148  7288 solver.cpp:337] Iteration 33, Testing net (#0)
I0805 02:18:51.242904  7288 solver.cpp:404]     Test net output #0: accuracy = 0.971053
I0805 02:18:51.242904  7288 solver.cpp:404]     Test net output #1: loss = 0.106738 (* 1 = 0.106738 loss)
I0805 02:18:51.311918  7288 solver.cpp:337] Iteration 34, Testing net (#0)
I0805 02:18:51.680330  7288 solver.cpp:404]     Test net output #0: accuracy = 0.976316
I0805 02:18:51.680330  7288 solver.cpp:404]     Test net output #1: loss = 0.102015 (* 1 = 0.102015 loss)
I0805 02:18:51.758543  7288 solver.cpp:337] Iteration 35, Testing net (#0)
I0805 02:18:52.127637  7288 solver.cpp:404]     Test net output #0: accuracy = 0.976316
I0805 02:18:52.127637  7288 solver.cpp:404]     Test net output #1: loss = 0.0973812 (* 1 = 0.0973812 loss)
I0805 02:18:52.212280  7288 solver.cpp:337] Iteration 36, Testing net (#0)
I0805 02:18:52.581396  7288 solver.cpp:404]     Test net output #0: accuracy = 0.976316
I0805 02:18:52.581396  7288 solver.cpp:404]     Test net output #1: loss = 0.0931733 (* 1 = 0.0931733 loss)
I0805 02:18:52.659525  7288 solver.cpp:337] Iteration 37, Testing net (#0)
I0805 02:18:53.013715  7288 solver.cpp:404]     Test net output #0: accuracy = 0.976316
I0805 02:18:53.013715  7288 solver.cpp:404]     Test net output #1: loss = 0.0891916 (* 1 = 0.0891916 loss)
I0805 02:18:53.097965  7288 solver.cpp:337] Iteration 38, Testing net (#0)
I0805 02:18:53.460435  7288 solver.cpp:404]     Test net output #0: accuracy = 0.976316
I0805 02:18:53.460435  7288 solver.cpp:404]     Test net output #1: loss = 0.0855421 (* 1 = 0.0855421 loss)
I0805 02:18:53.529400  7288 solver.cpp:337] Iteration 39, Testing net (#0)
I0805 02:18:53.914209  7288 solver.cpp:404]     Test net output #0: accuracy = 0.976316
I0805 02:18:53.914209  7288 solver.cpp:404]     Test net output #1: loss = 0.0820232 (* 1 = 0.0820232 loss)
I0805 02:18:53.982722  7288 solver.cpp:337] Iteration 40, Testing net (#0)
I0805 02:18:54.345510  7288 solver.cpp:404]     Test net output #0: accuracy = 0.978947
I0805 02:18:54.345510  7288 solver.cpp:404]     Test net output #1: loss = 0.0790738 (* 1 = 0.0790738 loss)
I0805 02:18:54.414526  7288 solver.cpp:228] Iteration 40, loss = 0.0707816
I0805 02:18:54.414526  7288 solver.cpp:244]     Train net output #0: loss = 0.0707816 (* 1 = 0.0707816 loss)
I0805 02:18:54.414526  7288 sgd_solver.cpp:106] Iteration 40, lr = 0.00997011
I0805 02:18:54.414526  7288 solver.cpp:337] Iteration 41, Testing net (#0)
I0805 02:18:54.783579  7288 solver.cpp:404]     Test net output #0: accuracy = 0.984211
I0805 02:18:54.783579  7288 solver.cpp:404]     Test net output #1: loss = 0.07625 (* 1 = 0.07625 loss)
I0805 02:18:54.861712  7288 solver.cpp:337] Iteration 42, Testing net (#0)
I0805 02:18:55.215157  7288 solver.cpp:404]     Test net output #0: accuracy = 0.986842
I0805 02:18:55.215157  7288 solver.cpp:404]     Test net output #1: loss = 0.0738107 (* 1 = 0.0738107 loss)
I0805 02:18:55.284169  7288 solver.cpp:337] Iteration 43, Testing net (#0)
I0805 02:18:55.646924  7288 solver.cpp:404]     Test net output #0: accuracy = 0.986842
I0805 02:18:55.646924  7288 solver.cpp:404]     Test net output #1: loss = 0.0708317 (* 1 = 0.0708317 loss)
I0805 02:18:55.746736  7288 solver.cpp:337] Iteration 44, Testing net (#0)
I0805 02:18:56.099936  7288 solver.cpp:404]     Test net output #0: accuracy = 0.986842
I0805 02:18:56.099936  7288 solver.cpp:404]     Test net output #1: loss = 0.0677046 (* 1 = 0.0677046 loss)
I0805 02:18:56.184074  7288 solver.cpp:337] Iteration 45, Testing net (#0)
I0805 02:18:56.531754  7288 solver.cpp:404]     Test net output #0: accuracy = 0.984211
I0805 02:18:56.531754  7288 solver.cpp:404]     Test net output #1: loss = 0.0645572 (* 1 = 0.0645572 loss)
I0805 02:18:56.616396  7288 solver.cpp:337] Iteration 46, Testing net (#0)
I0805 02:18:56.985436  7288 solver.cpp:404]     Test net output #0: accuracy = 0.981579
I0805 02:18:56.985436  7288 solver.cpp:404]     Test net output #1: loss = 0.0617554 (* 1 = 0.0617554 loss)
I0805 02:18:57.063761  7288 solver.cpp:337] Iteration 47, Testing net (#0)
I0805 02:18:57.416677  7288 solver.cpp:404]     Test net output #0: accuracy = 0.981579
I0805 02:18:57.416677  7288 solver.cpp:404]     Test net output #1: loss = 0.0590396 (* 1 = 0.0590396 loss)
I0805 02:18:57.501821  7288 solver.cpp:337] Iteration 48, Testing net (#0)
I0805 02:18:57.849179  7288 solver.cpp:404]     Test net output #0: accuracy = 0.976316
I0805 02:18:57.849179  7288 solver.cpp:404]     Test net output #1: loss = 0.0566682 (* 1 = 0.0566682 loss)
I0805 02:18:57.917774  7288 solver.cpp:337] Iteration 49, Testing net (#0)
I0805 02:18:58.286100  7288 solver.cpp:404]     Test net output #0: accuracy = 0.978947
I0805 02:18:58.286100  7288 solver.cpp:404]     Test net output #1: loss = 0.0545731 (* 1 = 0.0545731 loss)
I0805 02:18:58.348724  7288 solver.cpp:454] Snapshotting to binary proto file lenet_iter_50.caffemodel
I0805 02:18:58.402117  7288 sgd_solver.cpp:273] Snapshotting solver state to binary proto file lenet_iter_50.solverstate
I0805 02:18:58.417747  7288 solver.cpp:337] Iteration 50, Testing net (#0)
I0805 02:18:58.765154  7288 solver.cpp:404]     Test net output #0: accuracy = 0.984211
I0805 02:18:58.765154  7288 solver.cpp:404]     Test net output #1: loss = 0.0523614 (* 1 = 0.0523614 loss)
I0805 02:18:58.849794  7288 solver.cpp:337] Iteration 51, Testing net (#0)
I0805 02:18:59.203824  7288 solver.cpp:404]     Test net output #0: accuracy = 0.984211
I0805 02:18:59.203824  7288 solver.cpp:404]     Test net output #1: loss = 0.0505201 (* 1 = 0.0505201 loss)
I0805 02:18:59.286962  7288 solver.cpp:337] Iteration 52, Testing net (#0)
I0805 02:18:59.688545  7288 solver.cpp:404]     Test net output #0: accuracy = 0.986842
I0805 02:18:59.688545  7288 solver.cpp:404]     Test net output #1: loss = 0.0486458 (* 1 = 0.0486458 loss)
I0805 02:18:59.767055  7288 solver.cpp:337] Iteration 53, Testing net (#0)
I0805 02:19:00.120668  7288 solver.cpp:404]     Test net output #0: accuracy = 0.989474
I0805 02:19:00.120668  7288 solver.cpp:404]     Test net output #1: loss = 0.0469459 (* 1 = 0.0469459 loss)
I0805 02:19:00.220939  7288 solver.cpp:337] Iteration 54, Testing net (#0)
I0805 02:19:00.567886  7288 solver.cpp:404]     Test net output #0: accuracy = 0.989474
I0805 02:19:00.567886  7288 solver.cpp:404]     Test net output #1: loss = 0.0454346 (* 1 = 0.0454346 loss)
I0805 02:19:00.652102  7288 solver.cpp:337] Iteration 55, Testing net (#0)
I0805 02:19:01.005743  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:01.005743  7288 solver.cpp:404]     Test net output #1: loss = 0.0439673 (* 1 = 0.0439673 loss)
I0805 02:19:01.090380  7288 solver.cpp:337] Iteration 56, Testing net (#0)
I0805 02:19:01.437019  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:01.437019  7288 solver.cpp:404]     Test net output #1: loss = 0.0424628 (* 1 = 0.0424628 loss)
I0805 02:19:01.505641  7288 solver.cpp:337] Iteration 57, Testing net (#0)
I0805 02:19:01.868614  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:01.868614  7288 solver.cpp:404]     Test net output #1: loss = 0.0409566 (* 1 = 0.0409566 loss)
I0805 02:19:01.952947  7288 solver.cpp:337] Iteration 58, Testing net (#0)
I0805 02:19:02.306574  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:02.306574  7288 solver.cpp:404]     Test net output #1: loss = 0.0396168 (* 1 = 0.0396168 loss)
I0805 02:19:02.388217  7288 solver.cpp:337] Iteration 59, Testing net (#0)
I0805 02:19:02.737639  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:02.737639  7288 solver.cpp:404]     Test net output #1: loss = 0.0381724 (* 1 = 0.0381724 loss)
I0805 02:19:02.806531  7288 solver.cpp:337] Iteration 60, Testing net (#0)
I0805 02:19:03.153704  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:03.153704  7288 solver.cpp:404]     Test net output #1: loss = 0.0367321 (* 1 = 0.0367321 loss)
I0805 02:19:03.237939  7288 solver.cpp:228] Iteration 60, loss = 0.0354384
I0805 02:19:03.237939  7288 solver.cpp:244]     Train net output #0: loss = 0.0354384 (* 1 = 0.0354384 loss)
I0805 02:19:03.237939  7288 sgd_solver.cpp:106] Iteration 60, lr = 0.00995523
I0805 02:19:03.237939  7288 solver.cpp:337] Iteration 61, Testing net (#0)
I0805 02:19:03.591223  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:03.591223  7288 solver.cpp:404]     Test net output #1: loss = 0.0355842 (* 1 = 0.0355842 loss)
I0805 02:19:03.669739  7288 solver.cpp:337] Iteration 62, Testing net (#0)
I0805 02:19:04.023164  7288 solver.cpp:404]     Test net output #0: accuracy = 0.994737
I0805 02:19:04.023164  7288 solver.cpp:404]     Test net output #1: loss = 0.0347181 (* 1 = 0.0347181 loss)
I0805 02:19:04.091675  7288 solver.cpp:337] Iteration 63, Testing net (#0)
I0805 02:19:04.454399  7288 solver.cpp:404]     Test net output #0: accuracy = 0.994737
I0805 02:19:04.454399  7288 solver.cpp:404]     Test net output #1: loss = 0.0337878 (* 1 = 0.0337878 loss)
I0805 02:19:04.523416  7288 solver.cpp:337] Iteration 64, Testing net (#0)
I0805 02:19:04.890579  7288 solver.cpp:404]     Test net output #0: accuracy = 0.994737
I0805 02:19:04.890579  7288 solver.cpp:404]     Test net output #1: loss = 0.0325952 (* 1 = 0.0325952 loss)
I0805 02:19:04.954913  7288 solver.cpp:337] Iteration 65, Testing net (#0)
I0805 02:19:05.323719  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:05.323719  7288 solver.cpp:404]     Test net output #1: loss = 0.0314321 (* 1 = 0.0314321 loss)
I0805 02:19:05.392731  7288 solver.cpp:337] Iteration 66, Testing net (#0)
I0805 02:19:05.755445  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:05.755445  7288 solver.cpp:404]     Test net output #1: loss = 0.0304322 (* 1 = 0.0304322 loss)
I0805 02:19:05.824460  7288 solver.cpp:337] Iteration 67, Testing net (#0)
I0805 02:19:06.189793  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:06.189793  7288 solver.cpp:404]     Test net output #1: loss = 0.0294896 (* 1 = 0.0294896 loss)
I0805 02:19:06.256080  7288 solver.cpp:337] Iteration 68, Testing net (#0)
I0805 02:19:06.609148  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:06.609148  7288 solver.cpp:404]     Test net output #1: loss = 0.0286699 (* 1 = 0.0286699 loss)
I0805 02:19:06.693789  7288 solver.cpp:337] Iteration 69, Testing net (#0)
I0805 02:19:07.040943  7288 solver.cpp:404]     Test net output #0: accuracy = 0.992105
I0805 02:19:07.040943  7288 solver.cpp:404]     Test net output #1: loss = 0.0278288 (* 1 = 0.0278288 loss)
I0805 02:19:07.125205  7288 solver.cpp:337] Iteration 70, Testing net (#0)
I0805 02:19:07.472616  7288 solver.cpp:404]     Test net output #0: accuracy = 0.994737
I0805 02:19:07.472616  7288 solver.cpp:404]     Test net output #1: loss = 0.0270197 (* 1 = 0.0270197 loss)
I0805 02:19:07.557262  7288 solver.cpp:337] Iteration 71, Testing net (#0)
I0805 02:19:07.894731  7288 solver.cpp:404]     Test net output #0: accuracy = 0.994737
I0805 02:19:07.894731  7288 solver.cpp:404]     Test net output #1: loss = 0.026271 (* 1 = 0.026271 loss)
I0805 02:19:07.973101  7288 solver.cpp:337] Iteration 72, Testing net (#0)
I0805 02:19:08.326593  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:08.326593  7288 solver.cpp:404]     Test net output #1: loss = 0.0256099 (* 1 = 0.0256099 loss)
I0805 02:19:08.411201  7288 solver.cpp:337] Iteration 73, Testing net (#0)
I0805 02:19:08.758379  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:08.758379  7288 solver.cpp:404]     Test net output #1: loss = 0.0249613 (* 1 = 0.0249613 loss)
I0805 02:19:08.842595  7288 solver.cpp:337] Iteration 74, Testing net (#0)
I0805 02:19:09.196341  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:09.196341  7288 solver.cpp:404]     Test net output #1: loss = 0.0242344 (* 1 = 0.0242344 loss)
I0805 02:19:09.258853  7288 solver.cpp:337] Iteration 75, Testing net (#0)
I0805 02:19:09.613123  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:09.613123  7288 solver.cpp:404]     Test net output #1: loss = 0.023615 (* 1 = 0.023615 loss)
I0805 02:19:09.697762  7288 solver.cpp:337] Iteration 76, Testing net (#0)
I0805 02:19:10.045094  7288 solver.cpp:404]     Test net output #0: accuracy = 0.994737
I0805 02:19:10.045094  7288 solver.cpp:404]     Test net output #1: loss = 0.0231051 (* 1 = 0.0231051 loss)
I0805 02:19:10.113721  7288 solver.cpp:337] Iteration 77, Testing net (#0)
I0805 02:19:10.476856  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:10.476856  7288 solver.cpp:404]     Test net output #1: loss = 0.0226364 (* 1 = 0.0226364 loss)
I0805 02:19:10.545466  7288 solver.cpp:337] Iteration 78, Testing net (#0)
I0805 02:19:10.899094  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:10.899094  7288 solver.cpp:404]     Test net output #1: loss = 0.0222204 (* 1 = 0.0222204 loss)
I0805 02:19:10.977223  7288 solver.cpp:337] Iteration 79, Testing net (#0)
I0805 02:19:11.330703  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:11.330703  7288 solver.cpp:404]     Test net output #1: loss = 0.0215548 (* 1 = 0.0215548 loss)
I0805 02:19:11.415205  7288 solver.cpp:337] Iteration 80, Testing net (#0)
I0805 02:19:11.762137  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:11.762137  7288 solver.cpp:404]     Test net output #1: loss = 0.0209249 (* 1 = 0.0209249 loss)
I0805 02:19:11.846778  7288 solver.cpp:228] Iteration 80, loss = 0.0241723
I0805 02:19:11.846778  7288 solver.cpp:244]     Train net output #0: loss = 0.0241723 (* 1 = 0.0241723 loss)
I0805 02:19:11.846778  7288 sgd_solver.cpp:106] Iteration 80, lr = 0.00994042
I0805 02:19:11.846778  7288 solver.cpp:337] Iteration 81, Testing net (#0)
I0805 02:19:12.199707  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:12.199707  7288 solver.cpp:404]     Test net output #1: loss = 0.0204544 (* 1 = 0.0204544 loss)
I0805 02:19:12.278031  7288 solver.cpp:337] Iteration 82, Testing net (#0)
I0805 02:19:12.631784  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:12.631784  7288 solver.cpp:404]     Test net output #1: loss = 0.0200803 (* 1 = 0.0200803 loss)
I0805 02:19:12.700296  7288 solver.cpp:337] Iteration 83, Testing net (#0)
I0805 02:19:13.062958  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:13.062958  7288 solver.cpp:404]     Test net output #1: loss = 0.0197648 (* 1 = 0.0197648 loss)
I0805 02:19:13.147222  7288 solver.cpp:337] Iteration 84, Testing net (#0)
I0805 02:19:13.479094  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:13.479094  7288 solver.cpp:404]     Test net output #1: loss = 0.0192485 (* 1 = 0.0192485 loss)
I0805 02:19:13.563527  7288 solver.cpp:337] Iteration 85, Testing net (#0)
I0805 02:19:13.916857  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:13.916857  7288 solver.cpp:404]     Test net output #1: loss = 0.0187474 (* 1 = 0.0187474 loss)
I0805 02:19:14.001497  7288 solver.cpp:337] Iteration 86, Testing net (#0)
I0805 02:19:14.333204  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:14.333204  7288 solver.cpp:404]     Test net output #1: loss = 0.0183323 (* 1 = 0.0183323 loss)
I0805 02:19:14.417397  7288 solver.cpp:337] Iteration 87, Testing net (#0)
I0805 02:19:14.764318  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:14.764318  7288 solver.cpp:404]     Test net output #1: loss = 0.0179662 (* 1 = 0.0179662 loss)
I0805 02:19:14.848714  7288 solver.cpp:337] Iteration 88, Testing net (#0)
I0805 02:19:15.202013  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:15.202013  7288 solver.cpp:404]     Test net output #1: loss = 0.01766 (* 1 = 0.01766 loss)
I0805 02:19:15.280393  7288 solver.cpp:337] Iteration 89, Testing net (#0)
I0805 02:19:15.649106  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:15.649106  7288 solver.cpp:404]     Test net output #1: loss = 0.0172775 (* 1 = 0.0172775 loss)
I0805 02:19:15.717664  7288 solver.cpp:337] Iteration 90, Testing net (#0)
I0805 02:19:16.080340  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:16.080340  7288 solver.cpp:404]     Test net output #1: loss = 0.016939 (* 1 = 0.016939 loss)
I0805 02:19:16.164538  7288 solver.cpp:337] Iteration 91, Testing net (#0)
I0805 02:19:16.502742  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:16.502742  7288 solver.cpp:404]     Test net output #1: loss = 0.0166098 (* 1 = 0.0166098 loss)
I0805 02:19:16.581220  7288 solver.cpp:337] Iteration 92, Testing net (#0)
I0805 02:19:16.934193  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:16.934193  7288 solver.cpp:404]     Test net output #1: loss = 0.0163446 (* 1 = 0.0163446 loss)
I0805 02:19:17.018379  7288 solver.cpp:337] Iteration 93, Testing net (#0)
I0805 02:19:17.365928  7288 solver.cpp:404]     Test net output #0: accuracy = 0.997368
I0805 02:19:17.365928  7288 solver.cpp:404]     Test net output #1: loss = 0.016099 (* 1 = 0.016099 loss)
I0805 02:19:17.434541  7288 solver.cpp:337] Iteration 94, Testing net (#0)
I0805 02:19:17.781354  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:17.781354  7288 solver.cpp:404]     Test net output #1: loss = 0.0157068 (* 1 = 0.0157068 loss)
I0805 02:19:17.881271  7288 solver.cpp:337] Iteration 95, Testing net (#0)
I0805 02:19:18.234609  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:18.234609  7288 solver.cpp:404]     Test net output #1: loss = 0.0153735 (* 1 = 0.0153735 loss)
I0805 02:19:18.303120  7288 solver.cpp:337] Iteration 96, Testing net (#0)
I0805 02:19:18.650723  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:18.650723  7288 solver.cpp:404]     Test net output #1: loss = 0.0151255 (* 1 = 0.0151255 loss)
I0805 02:19:18.735190  7288 solver.cpp:337] Iteration 97, Testing net (#0)
I0805 02:19:19.082682  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:19.082682  7288 solver.cpp:404]     Test net output #1: loss = 0.0149323 (* 1 = 0.0149323 loss)
I0805 02:19:19.166878  7288 solver.cpp:337] Iteration 98, Testing net (#0)
I0805 02:19:19.520220  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:19.520220  7288 solver.cpp:404]     Test net output #1: loss = 0.0147758 (* 1 = 0.0147758 loss)
I0805 02:19:19.582723  7288 solver.cpp:337] Iteration 99, Testing net (#0)
I0805 02:19:19.936482  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:19.936482  7288 solver.cpp:404]     Test net output #1: loss = 0.0144556 (* 1 = 0.0144556 loss)
I0805 02:19:20.020927  7288 solver.cpp:454] Snapshotting to binary proto file lenet_iter_100.caffemodel
I0805 02:19:20.083432  7288 sgd_solver.cpp:273] Snapshotting solver state to binary proto file lenet_iter_100.solverstate
I0805 02:19:20.121194  7288 solver.cpp:317] Iteration 100, loss = 0.0177263
I0805 02:19:20.121194  7288 solver.cpp:337] Iteration 100, Testing net (#0)
I0805 02:19:20.468515  7288 solver.cpp:404]     Test net output #0: accuracy = 1
I0805 02:19:20.468515  7288 solver.cpp:404]     Test net output #1: loss = 0.014137 (* 1 = 0.014137 loss)
I0805 02:19:20.468515  7288 solver.cpp:322] Optimization Done.
I0805 02:19:20.468515  7288 caffe.cpp:223] Optimization Done.
'''

#从训练结果也可以看出(Test net output #0: accuracy = 1)，
#我们肯定不能讲train数据与test数据设置为一样的



























