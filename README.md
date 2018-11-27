# Tiny_DNN是一个轻型CNN框架,基于C++和C++矩阵运算库Eigen。  
1、 Tiny_DNN实现了Convolution、Pooling、FullConnect、Softmax、Cross Entropy 等Layers 及其FP、BP 和Weight Update，能快速构建基于自定义卷积核数目、自定义Layers 类型和层数的计算模型

2、 基于该框架快速计算模型计算精度如下：  
Mnist :  99%  （Conv+Pool+Conv+Pool+FullConn*2+Softmax_Cross_Entropy）  
Cifar :  68%  （Conv+Pool+Conv+Pool+FullConn*2+Softmax_Cross_Entropy）