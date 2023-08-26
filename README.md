## 1 简介(Belif introduction) 
本项目基于`tf.net`框架复现了`bert`模型，风格为`huggingface style`，主要复现IMDB数据集的结果。

Reproducing `bert` using `tf.net(huggingface style)`, reproduce the result on IMDB dataset.

**说明(Introduction)**

运行项目前请先下载bert的预训练权重与IMDB数据集。

Please download bert's pretraining weights and IMDB dataset before running the project.

**下载地址：**
- [预训练权重(Pretraining weights)，密码(Password):z9fw](https://pan.baidu.com/s/15cpYWQ4PVKtDbQOleisslw)

- [IMDB数据集(IMDB dataset)](http://ai.stanford.edu/~amaas/data/sentiment/)


## 2环境依赖(Environment)
运行以下命令即可配置环境

Run the following command to configure the environment
```bash
dotnet add package TensorFlow.NET
```
```bash
dotnet add package TensorFlow.Keras
```
```bash
dotnet add package SciSharp.TensorFlow.Redist
```
或者也可以依赖tf.net的官方源码

Alternatively, you can rely on the official source code of tf.net：
-[tf.net source code](https://github.com/SciSharp/TensorFlow.NET)
```bash
<ItemGroup>
    <ProjectReference Include="TensorFlow.NET-master\src\TensorFlowNET.Core\Tensorflow.Binding.csproj" />
    <ProjectReference Include="TensorFlow.NET-master\src\TensorFlowNET.Keras\Tensorflow.Keras.csproj" />
</ItemGroup>
```

## 3超参说明（Description of hyperparameters）
在`main.cs`文件里面(in `main.cs`)
```bash
batch_size = 4; 
learning_rate = (float)2e-5;
num_classes = 2;
max_seq_len = 180;
epoch = 10;
config = new BertConfig(); 
pretrained_weight_path = "D:\\bert_model.h5";
dataset_path = "D:\\datasets\\";
vocab_file = "D:\\vocab.txt";
```
如果您有更好的计算资源，推荐将`batch_size`设置为32，`max_seq_len`设置为512

If you have better computing resources, it is recommended to set `batch_size` to 32 and `max_seq_len` to 512
## 4运行说明（Code operation instructions）
运行如下代码即可运行项目：

Run the following code to run the project:
```bash
dotnet run bert-tf.net.csproj
```



