# Tiny_CNN是一个轻型CNN框架，基于C++和C++矩阵运算库Eigen实现，目前仅支持CPU运算。  

1、 Tiny_CNN自定义实现了Convolution、Pooling、FullConnect、Softmax、CrossEntropy 等Layers 及其FP、BP 和Weight Update，实现了Sigmoid、Tanh、Relu、LRelu等激活函数。通过运算符重载实现了快捷构建基于自定义卷积核数目、自定义Layers 类型和层数的计算模型，可以快速进行模型搭建，快速运算。

2、 基于该框架快速计算模型精度如下：  
Mnist Accuracy  :  99%    
Net 结构        :   2 * ( Conv + LRelu + Pool ) + 2 * ( FullConn + LRelu ) + Softmax_Cross_Entropy  
Cifar Accuracy  :  70%  
Net 结构        :   2 * ( Conv + LRelu + Pool ) + 2 * ( FullConn + LRelu ) + Softmax_Cross_Entropy  

3、 水平有限，代码有不完善的地方请在ISSUE批评指正！