# Simple Transformer
24年高级软件工程课程作业：利用Transformer模型实现中译英。

使用KAN而不是MLP作为Transformer中的前馈神经网络

缺点是训练时间长。优点是比起MLP的效果更好

通过更改position_wise_feedforward.py中的gridsize可以改变KAN的大小

出于本人的pc机能限制，默认为5，更大的gridsize可能会提高模型的性能，但同时也会增加计算的复杂性和内存需求（100时需70G内存）
# 参考：
https://arxiv.org/abs/1706.03762   （Attension is all you need）

https://arxiv.org/html/2404.19756v1   (KAN: Kolmogorov–Arnold Networks)

# 运行

在根目录创建一个save文件夹用来保存训练好的模型

使用"python3 run.py"训练模型

如果已存在训练好的模型，则"python3 run.py --type evaluate"
