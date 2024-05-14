# Simple Transformer
  24年高级软件工程课程作业：利用Transformer模型实现中译英。

  使用KAN而不是MLP作为Transformer中的前馈神经网络
  
  缺点是训练时间长。优点是比起MLP的效果更好
  
  通过更改position_wise_feedforward.py中的gridsize可以改变KAN的大小
  
  出于本人的pc机能限制，默认为5，更大的gridsize可能会提高模型的性能，但同时也会增加计算的复杂性和内存需求（100时需70G内存）
``
