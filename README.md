

> pytorch -＞ onnx -＞ caffe -＞ nnie
>
> https://github.com/ultralytics/yolov5/tree/v4.0
>

## 1.pytorch

1. 将 focus 层换成一个卷积，这个 op 海思上面不用想了(修正，自己试了下focus可以用caffe实现，海思也支持op， [yolo v5 focus 层海思部署的可能性](https://blog.csdn.net/tangshopping/article/details/111150493) )；
2. 将 leaky relu、SiLU换成 relu或者relu6，海思是支持 prelu 的，所以也支持它，不过群友们反映这个 op 很慢，输出还不稳定（这个我没有做实验，真伪性存疑），所以就干脆给它替换了；
3. 上采样层，海思支持的上采样层是 unpooling 方式，而 yolo v5里的上采样方式是最近邻插值（nearest），鉴于各种因素考虑，还是把它换成了分组转置卷积（分组这里要注意，yolo v5 网络其实大部分卷积都是深度卷积 + 逐点卷积，所以转置卷积也分组吧）；
4. spp 层的 maxpool ceil mode 都是默认的 false 状态，而海思里的 caffe 只支持 ceil mode 方式，所以要改成 ceil mode = True 。一开始忘记打开使得输出bbox明显偏大（不敢确认是这个的原因），后来特意停止训练修改该参数，再继续训练，后来发现 bbox 正常了。

### 1.Activation 修改

> 两种思路：1.直接relu；2.海思支持rrelu,可参考文档进行prototxt改写
>
> 经试验对比，relu6比relu效果更好，可在量化前使用relu6训练
>
> silu = x*sigmoid(x)

`yolov5/models/common.py`：

```python
#class Conv(nn.Module):
# self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity()) # ! 源码
self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
```

### 2.SPP修改

`yolov5/models/common.py`

```python
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2,ceil_mode=True) for x in k]) #ceil mode = True
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1)
```

### 3.unpooling转换

> yolov5s：修改head部分的两处upsample即可

```python
# YOLOv5 head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    # [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [-1, 1, nn.ConvTranspose2d, [256, 256, 2,2]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, C3, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    # [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [-1, 1, nn.ConvTranspose2d, [128, 128, 2,2]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, C3, [256, False]], # 17 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]
```

#### nn.ConvTranspose2d与nn.Upsample

+ `nn.ConTranspose2d`有参数可以训练

  ```python
  nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0,  output_padding=0, groups=1, bias=True, dilation=1)
  
  in_channels(int) – 输入信号的通道数
  out_channels(int) – 卷积产生的通道数
  kerner_size(int or tuple) - 卷积核的大小
  stride(int or tuple,optional) - 卷积步长，即要将输入扩大的倍数。
  padding(int or tuple, optional) - 输入的每一条边补充0的层数，高宽都增加2*padding
  output_padding(int or tuple, optional) - 输出边补充0的层数，高宽都增加padding
  groups(int, optional) – 从输入通道到输出通道的阻塞连接数
  bias(bool, optional) - 如果bias=True，添加偏置
  dilation(int or tuple, optional) – 卷积核元素之间的间距
  ```

+ `nn.Upsample`没有参数、速度快，基于给定策略上采样

  ```python
  torch.nn.Upsample(size=None, scale_factor=None, mode=‘nearest’, align_corners=None)
  
  size (tuple ，optional)– 是输出矩阵大小（[optional D_out], [optional H_out], W_out）的元组。
  scale_factor (int / tuple of python:ints, optional) – 图像宽/高/深度的倍数
  mode – (string, optional) – 上采样方法: 有nearest, linear, bilinear, bicubic and trilinear. 默认是: nearest
  align_corners (bool, optional) – 如果为true，则输入和输入tensor的角点像素对齐，从而保留这些像素的像素值。 默认值: False
  ```

+ 对于反卷积当stride=2，kernal为奇数时候会出现棋盘格问题

+ 不同的s/m/l/x的kernal尺寸的确定可以参考

  ```python
  #models/yolo.py
  
  from torchsummaryX import summary
  summary(model, torch.zeros((1, 3, 640,640)))
  ```

### 4.focus层修改

#### 1.passthrough

> yolo v2的 passthrough 层（也叫做Reorg层）与 v5 的 focus 层很像，海思是支持 passthrough 层的
>
> PassThrough 层，参考设计为 YOLO v2 网络，开源工程地址为 https://pjreddie.com/darknet/yolo/。v2里面不叫 passthrough layer，叫 reorg laye

##### 1.区别

###### 1.passthrough层

+ 海思文档

  PassThrough层为Yolo v2中的一个自定义层，由于Yolo v2并不是使用Caffe框架实现，因此对于该层没有标准的定义。该层实现的功能为将feature map在spatial维度上的数据展 开到channel维度上，原始在channel维度上连续的元素在展开后的feature map中依然是 连续的。如将26×26×512的feature变成13×13×2048的feature，做法为将相邻的像素展开 到channel维度，示意图如图3-9。

  <img src="./imgs/passthrough.png" style="zoom:60%;" />

  该层操作在实现过程中需要给定的参数包括在spatial维度上进行展开的窗口大小，以下 称为block，需要定义该block窗口的高度height和宽度width，两者均为正整数，分别即为block_height和block_width，且需要保证block_height和block_width能够被输入feature map的height和width整除。若输入feature map的channel维度为num_in，则输出feature map的channel维度$num\_output=num\_in*block\_height*block\_width$。如上述的示例中， block_height=2，block_width=2，num_output=2048。

  在定义该层的参数时，需要block_height、block_width和输出feature map的channel维度 数num_output。虽然num_output可以通过num_in、block_height、block_width计算得出，但是在参数定义是还是需要进行设定，用于参数合法性的检查。

  PassThrough层在prototxt中进行定义的方式示例如下，layer的type定义为PassThrough：

  ```protobuf
  layer {
   name: "pass_through"
   type: "PassThrough"
   bottom: "some_input"
   top: "some_output"
   pass_through_param{
   num_output: xxx
   block_height: xxx
   block_width: xxx
   }
  }
  ```

+ 源码

```c++
// 它的源码是 c++ 的，不是 python 格式
int reorg_cpu(THFloatTensor *x_tensor, int w, int h, int c, int batch, int stride, int forward, THFloatTensor *out_tensor)
 3 {
 4     // Grab the tensor
 5     float * x = THFloatTensor_data(x_tensor);
 6     float * out = THFloatTensor_data(out_tensor);
 7
 8     // https://github.com/pjreddie/darknet/blob/master/src/blas.c
 9     int b,i,j,k;
10     int out_c = c/(stride*stride);
11
12     for(b = 0; b < batch; ++b){  //batch_size
13
14         for(k = 0; k < c; ++k){  //channel
15
16             for(j = 0; j < h; ++j){  //height
17
18                 for(i = 0; i < w; ++i){  //width，可以看见passthrough 是行优先 ！
19
20                     int in_index  = i + w*(j + h*(k + c*b));
21                     int c2 = k % out_c;
22                     int offset = k / out_c;
23                     int w2 = i*stride + offset % stride;
24                     int h2 = j*stride + offset / stride;
25                     int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
26                     if(forward) out[out_index] = x[in_index]; // 压缩channel
27                     else out[in_index] = x[out_index];        // 扩展channel
28                 }
29             }
30         }
31     }
32
33     return 1;
34 }
```

再结合海思的文档的图示来看，更加清晰，上图可以看见数据重新排布的顺序是 红色 -> 天蓝色 -> 淡绿（左下） -> 深绿（右下），即行优先。再结合上文第18行代码可得出，**passthrough 层确实是行优先**，这个先记住。

###### 2.focus层

```python
# yolov5/models/common.py  line81
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
```

测试输出顺序：

```python
a = np.array(range(8)).reshape(1, 2, 2, 2)
print(a.shape)
print(a)

d = np.concatenate([a[..., ::2, ::2], a[..., 1::2, ::2], a[..., ::2, 1::2], a[..., 1::2, 1::2]], 1)
print(d.shape)
print(d)

# 结果为

a.shape = (1, 2, 2, 2)

a =  [[[[0 1]
	   [2 3]]

	  [[4 5]
	   [6 7]]]]
------------------------
d.shape = (1, 8, 1, 1)

d =	[[[[0]]  # 0 、4 是每个通道左上角位置处的元素值

	  [[4]]
	----------
	  [[2]]  # 2 、6 是每个通道左下角位置处的元素值

	  [[6]]  # 这说明 focus 层是列优先
	----------
	  [[1]]

	  [[5]]
	----------
	  [[3]]

	  [[7]]]]
```

从上面打印结果可看出，**focus 层是列优先**。

###### 3.总结

<img src="./imgs/channel.png" style="zoom:60%;" />

##### 2.植入

三个思路：

+ 源码修改
+ caffe permute op
+ 修改caffe模型权值*(选用该条思路)*

###### 1.修改源码

> 修改python源码重新进行训练，因为要修改的地方较多，精度不至于因为模型转换而掉太多

```python
# 将上面官方源码改为下面样子，再训练
def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2]， x[..., ::2, 1::2], x[..., 1::2, ::2],  x[..., 1::2, 1::2]], 1))
```

###### 2.caffe permute op

permute即numpy中的转置，详细解释如下：

```python
pytorch : input_data -> focus（列优先） -> 卷积
caffe : input_data -> passthrough（行优先） -> 卷积
caffe_add_permute : input_data -> permute(0, 1, 3, 2) -> passthrough（行优先） -> 卷积
```

因为转换模型时，如果后面那个卷积层权值不做变动，那 pytorch 模型与 caffe 模型的输出肯定不一样。我上图也讲述过，它们数据排布不一样，而权值排布不变，导致输出肯定不一样！就此，如果我们在数据输入时，加个 permute 操作，将行、列的数据对换一下，这样就可以对应的上了。我做了个实验，结果如下：

```python
a = np.array(range(8)).reshape(1, 2, 2, 2)
print(a.shape)
print(a)
# focus 代码，列优先                         ↓↓↓↓↓
d = np.concatenate([a[..., ::2, ::2], a[..., 1::2, ::2], a[..., ::2, 1::2], a[..., 1::2, 1::2]], 1)
print(d)

a1 = a.transpose(0, 1, 3, 2)  # 做转置，caffe 里是 permute(0, 1, 3, 2)
# 注意，此处代码有变动，它不是 focus 代码，它是行优先   ↓↓↓↓↓
d1 = np.concatenate([a1[..., ::2, ::2], a1[..., ::2, 1::2], a1[..., 1::2, ::2], a1[..., 1::2, 1::2]], 1)
print(d1)

# 输出结果如下：

# 输入数据 shape
(1, 2, 2, 2)
# 输入数据打印结果
[[[[0 1]
   [2 3]]

  [[4 5]
   [6 7]]]]
------------
# 模拟 focus 层输出结果
[[[[0]]

  [[4]]

  [[2]]

  [[6]]

  [[1]]

  [[5]]

  [[3]]

  [[7]]]]
--------------
# 先转置，再行优先采样，输出结果
[[[[0]]

  [[4]]

  [[2]]

  [[6]]

  [[1]]

  [[5]]

  [[3]]

  [[7]]]]
```

可以发现，二者输出是一致的。但是还有个问题，海思中的 permute 操作受限很大，所以此方案较难实现，参考如下：

<img src="./imgs/permute.png" style="zoom:80%;" />

###### 3.对caffe模型的权值做修改

上述第2个方案是对 x 做变换，由于一些 op 受限，使得想法没能实现。能否对 w 进行变换？因为转换模型是在 pc 端进行的，而模型的转换对于模型参数来说，几乎就是简单的复制粘贴，板端它的 shape 是不变的，既然它是在 pc 端做的变换，限制可以说基本没有，所以实现可能性大大增加。

先看看 focus 输出及其后面的 op 参数 shape，考虑到yolo v5不方便打印中间层参数，自建了个网络，结构如下：

<img src="./imgs/focus.png" style="zoom:67%;" />

以上是pytorch模型，要把红框里的结构干掉，换成 passthrough 层，此外图中的卷积层也要做变换。

##### 3.实现对caffe模型得权值做修改

###### 1.注意

1. 由于onnx的切片操作时是先对 height 方向做切片，再对 width 方向做切片，所以会导致中间的 feature shape 宽高不一致，在`yolov5_nnie2caffe/convertCaffe.py`中的`getGrapt`中直接load进来就好。

   ```python
   #  该函数在 convertCaffe.py 下
   def getGraph(onnx_path):
       model = onnx.load(onnx_path)
       #model = shape_inference.infer_shapes(model)  # 注释它，不然会报错
       model_graph = model.graph
       graph = Graph.from_onnx(model_graph)
       graph = graph.transformed(transformers)
       graph.channel_dims = {}

       return graph
   ```

2. 总共要修改两处：

   1. passthrough层
   2. passthrough层输出后第一个卷积层

###### 2.添加caffe op

`yolov5_nnie2caffe/onnx2caffe/_operators.py`

```python
# onnx2caffe\_operators.py 最下面
_ONNX_NODE_REGISTRY = {
    "Conv": _convert_conv,
    "Relu": _convert_relu,
    "PRelu": _convert_prelu,
    "BatchNormalization": _convert_BatchNorm,
    "Add": _convert_Add,
    "Mul": _convert_Mul,
    "Reshape": _convert_Reshape,
    "MaxPool": _convert_pool,
    "AveragePool": _convert_pool,
    "Dropout": _convert_dropout,
    "Gemm": _convert_gemm,
    "MatMul": _convert_matmul,
    "Upsample": _convert_upsample,
    "Concat": _convert_concat,
    "ConvTranspose": _convert_conv_transpose,
    "Sigmoid": _convert_sigmoid,
    "Flatten": _convert_Flatten,
    "Transpose": _convert_Permute,
    "Softmax": _convert_Softmax,
    "PassThrough": _convert_PassThrough  # 在最后一行加上我们需要的 passthrough 层键值
}
```

之后再在 op 字典上面选个看顺眼的位置，加上下面的代码：

```python
# onnx2caffe\_operators.py 中
def _convert_PassThrough(node_name, input_name, output_name, input_channel, block_height, block_width):

    layer = myf('PassThrough', node_name, [input_name], [output_name],
                pass_through_param=dict(
                    num_output=input_channel * block_height * block_width,
                    block_height=block_height,
                    block_width=block_width,
    ))

    return layer
```

`yolov5_nnie2caffe/onnx2caffe/_weightloader.py`

```python
# onnx2caffe\_weightloader.py 最下面
_ONNX_NODE_REGISTRY = {
    "Conv": _convert_conv,
    "Relu": _convert_relu,
    "PRelu": _convert_prelu,
    "BatchNormalization": _convert_BatchNorm,
    "Add": _convert_Add,
    "Mul": _convert_Mul,
    "Reshape": _convert_Reshape,
    "MaxPool": _convert_pool,
    "AveragePool": _convert_pool,
    "Dropout": _convert_dropout,
    "Gemm": _convert_gemm,
    "MatMul": _convert_matmul,
    "Upsample": _convert_upsample,
    "Concat": _convert_concat,
    "ConvTranspose": _convert_conv_transpose,
    "Sigmoid": _convert_sigmoid,
    "Flatten": _convert_Flatten,
    "Transpose": _convert_Permute,
    "Softmax": _convert_Softmax,
    "PassThrough": _convert_PassThrough  # 添加我们需要的 passthrough 层
}
```

同样，在其上面加入下面的代码：

```python
# onnx2caffe\_weightloader.py 中，因为这个 op 没有权值，所以不需要复制权值，写个 pass 就好
def _convert_PassThrough(node, graph, err):
    pass
```

注： _operators.py 是生成 prototxt 文件用的，而 _weightloader.py 是生成 caffemodel 文件用的，故二者很相似。
这就可以了吗？还没有。

###### 3.其他修改

回到 convertCaffe.py 文件，修改下面这个函数，建议开两个网页，对着我的网络图来加强理解：

```python
# convertCaffe.py 文件下
def convertToCaffe(graph, prototxt_save_path, caffe_model_save_path, exis_focus=True, focus_concat_name="Concat_40", focus_conv_name="Conv_41"):  # 如果有 focus 层，自己添加参数
    exist_edges = []
    layers = []
    exist_nodes = []
    err = ErrorHandling()
    gap_kernel_shape = [4, 4]  # 定制化操作参数，不会通用, gap 的池化卷积层
    for i in graph.inputs:  # input 就是可视化中，第一个灰色东西，显示输入名 和 输入 shape，不是 op.
        edge_name = i[0]  # 一般是 images, data, input 这种名字

        input_layer = cvt.make_input(i)  # 生成 prototxt 风格的input

        layers.append(input_layer)
        exist_edges.append(i[0])
        graph.channel_dims[edge_name] = graph.shape_dict[edge_name][1]  # shape_dict[edge_name] 如 (1, 3, 112, 112) 这种

    for id, node in enumerate(graph.nodes):

        node_name = node.name  # node name 参数，就是节点在当前模型中的名字

        op_type = node.op_type  # op 类型，卷积， relu 这种

        if exis_focus:
            if op_type == "Slice":
                continue
            if node_name == focus_concat_name:
                converter_fn = cvt._ONNX_NODE_REGISTRY["PassThrough"]
                output_name = str(node.outputs[0])
                layer = converter_fn("focus", "images", output_name, 3, 2, 2)  # 3是输入通道，2 是 pytorch 中的步长
                if type(layer) == tuple:
                    for l in layer:  # 一般是 bn 层， caffe 中的 bn 是分为两部分， BN 和 Scale 层
                        #  print("layer.name = ", l.layer_name)
                        layers.append(l)
                else:
                    layers.append(layer)
                outs = node.outputs  # 节点输出名
                for out in outs:
                    exist_edges.append(out)
                continue
        if op_type == "Clip":  # relu6 在 onnx 里是 clip
            op_type = "Relu6"

        #print(node_name)
        inputs = node.inputs  # 列表，由可视化中 input 一栏中 name 字段组成，顺序同可视化界面一致。如果某个键有参数数组，则也会在 input_tensors 存在

        inputs_tensor = node.input_tensors  # 字典，可视化界面中，如果有参数数组就是这里面的值，键也在input 中， 有多少参数数组就有多少键值

        input_non_exist_flag = False

        for inp in inputs:  # input 组成元素有两种，一是上层节点 name，二是本层参数 name
            if inp not in exist_edges and inp not in inputs_tensor:  # 筛除，正常节点判断条件是不会成立的
                input_non_exist_flag = True
                break
        if input_non_exist_flag:
            continue

        if op_type not in cvt._ONNX_NODE_REGISTRY:  # 如果没在 op 字典中，报错
            err.unsupported_op(node)
            continue
        converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]  # 相应转换函数
        if op_type == "GlobalAveragePool":
            layer = converter_fn(node, graph, err, gap_kernel_shape)
        else:
            #print("GlobalAveragePool  GlobalAveragePool")
            #print(op_type)
            layer = converter_fn(node, graph, err)
        if type(layer) == tuple:
            for l in layer:  # 一般是 bn 层， caffe 中的 bn 是分为两部分， BN 和 Scale 层
                #  print("layer.name = ", l.layer_name)
                layers.append(l)
        else:
            layers.append(layer)
        outs = node.outputs  # 节点输出名
        for out in outs:
            exist_edges.append(out)  # 储存输出节点，方便下面使用

    net = caffe_pb2.NetParameter()  # caffe 模型结构
    for id, layer in enumerate(layers):

        layers[id] = layer._to_proto()  # 转为 proto 风格？
        print(layers[id])
    net.layer.extend(layers)  # 将层名加入网络模型

    with open(prototxt_save_path, 'w') as f:  # 形成 prototxt 文件
        print(net, file=f)  # 写入 prototxt 文件
    # ------ 到此 prototxt 文件转换结束 ------
    # ------ 下面转换 caffemodel 文件 ------
    caffe.set_mode_cpu()
    deploy = prototxt_save_path
    net = caffe.Net(deploy,
                    caffe.TEST)

    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type

        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False
        if exis_focus:
            if op_type == "Slice":
                continue
        if op_type == "Clip":
            op_type = "Relu6"
        if op_type not in wlr._ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        #print(node_name)
        converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
        if node_name == focus_conv_name:
            converter_fn(net, node, graph, err, pass_through=1)
        else:
            converter_fn(net, node, graph, err)  # 复制模型参数

    net.save(caffe_model_save_path)  # 保存模型
    return net
# 该函数在 convertCaffe.py 下
def getGraph(onnx_path):
    model = onnx.load(onnx_path)
    #model = shape_inference.infer_shapes(model)  # 注释它，不然会报上面的错
    model_graph = model.graph
    graph = Graph.from_onnx(model_graph)
    graph = graph.transformed(transformers)
    graph.channel_dims = {}

    return graph
```

这里讲下为什么切片后的第一个卷积要稍微变换一下：[focus与passthrough区别图](#3.总结)

由于做切片操作时，passthrough 与 focus 层的方式不同，前者是行优先，后者是列优先，这就使得二者输出的 feature map 虽然 shape 一样，但是里面的数据排布有些不同，我用红框做了提示，上面也说过，对 feature map 在海思里没法实现，只能对权值做变换，具体就是调换一下权值的参数顺序，看我代码吧，有些抽象：

```python
# onnx2caffe\_weightloader.py 中，找到这个函数（这个函数已经存在了，它是卷积的转换函数），做些修改
def _convert_conv(net, node, graph, err, pass_through=1):
    weight_name = node.inputs[1]
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    # net.params[node_name][0].data = W
    # if bias_flag:
    #     net.params[node_name][1].data = bias
    if pass_through:  # 如果涉及到 passthrough
        pass_through_group = W.shape[1] // 4  # 分成四组，这是 passthrough 的性质, e.g W.shape = (3, 12, 1, 1)
        w1 = W[:, 0: pass_through_group, :, :]  # shape = (3, 3, 1, 1)
        w2 = W[:, pass_through_group: pass_through_group * 2, :, :]  # shape = (3, 3, 1, 1)
        w3 = W[:, pass_through_group * 2:pass_through_group * 3, :, :]  # shape = (3, 3, 1, 1)
        w4 = W[:, pass_through_group * 3:pass_through_group * 4, :, :]  # shape = (3, 3, 1, 1)
        W = np.concatenate((w1, w3, w2, w4), 1)  # 调换一下 w2、w3 的位置
        np.copyto(net.params[node_name][0].data, W, casting='same_kind')
    else:
        np.copyto(net.params[node_name][0].data, W, casting='same_kind')
    if bias_flag:  # b 不用做修改，因为是卷积核内部的通道变换，并不是卷积核之间的变化
        np.copyto(net.params[node_name][1].data, bias, casting='same_kind')
```

提醒一下，由于我水平有限，是直接加个条件执行，例如 `if node_name == "Conv_41"`这种，并不是每个网络的相应节点名都叫`Conv_41`，你们根据实际情况来改，不需要时最好注释掉，免得莫名其妙的错误。源码在`convertCaffe.py `的line135

转换后的结构：

![](./imgs/res_focus.png)

#### 2.conv

直接将focus层替换为conv层

```yaml
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 3,2]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, C3, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 9, C3, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, C3, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 1, SPP, [1024, [5, 9, 13]]],
    [-1, 3, C3, [1024, False]], # 9
  ]
```

## 2.onnx
   onnx-simplifier 的作用是通过等价替换简化模型结构提升推理性能，但简化前后提取的特征向量应该是一致的。如果遇到了两者提取特征向量不一致的问题，请先切换到 CPU 后端查看结果。顺便建议在导出模型时将模型转换到 CPU 后端导出。

   ```PYTHON
pip install onnx==1.8.1
pip install onnx-simplifier

model.model[-1].export = True
opset_version = 10
   ```
  ```bash
python models/export.py
python -m onnxsim onnx模型名称 yolov5s-simple.onnx 得到最终简化后的onnx模型
  ```

## 3.caffemodel

> yolov5_Caffe：https://github.com/Hiwyl/yolov5_caffe

+ ubuntu18.04
+ cuda10.0
+ cudnn7

### 1.环境

#### 1.依赖包

```shell
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install git cmake build-essential
```

#### 2.opencv

**opencv-3.4.8 **

+   下载安装包

```python
https://opencv.org/releases/
```

+   编译安装

```python
cd opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_TIFF=ON ..
############################################################################
#安装
make -j8
make install
#安装完成之后，添加路径。
        sudo vim /etc/ld.so.conf.d/opencv.conf
#在末尾添加如下内容：
          /usr/local/lib
#保存之后执行：
          sudo ldconfig
          sudo vim /etc/bash.bashrc
#在末尾添加如下内容：
        PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
        export PKG_CONFIG_PATH
#source一下，使修改立即生效。
        source /etc/bash.bashrc
#查看opencv版本   
    pkg-config --modversion opencv
```

#### 3.anaconda

```bash
chmod +x Anaconda3-5.2.0-Linux-x86_64.sh
./Anaconda3-5.2.0-Linux-x86_64.sh

按ENTER，然后按q调至结尾
接受协议　yes
安装路径　使用默认路径
执行安装
vi .bashrc
export PATH=/root/anaconda3/bin:$PATH
```

#### 4.caffe

```bash
git clone https://github.com/Wulingtian/yolov5_caffe
#修改Makefile.config  ananconda和cuda的安装路径即可
cd yolov5_caffe
make clean
export CPLUS_INCLUDE_PATH=/root/anaconda3/include/python3.6m
```

1. 修改Makefile文件

   ```shell
   LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5

   修改为：

   LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
   ```

   ```shell
   NVCCFLAGS +=-ccbin=$(CXX) -Xcompiler-fPIC $(COMMON_FLAGS)

   修改为：

   NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
   ```

   ```shell
   LIBRARIES += boost_thread stdc++后加boost_regex

   修改为:

   LIBRARIES += boost_thread stdc++ boost_regex
   ```

2. 编译Caffe-SSD

   ```shell
   make all -j8
   make pycaffe -j8
   vim ~/.bashrc
   export PYTHONPATH=/root/yolov5_caffe/python:$PYTHONPATH
   source ~/.bashrc
   ```

3. 测试

   ```shell
   XXX@XXX:~$ python
   Python 2.7.15rc1 (default, Apr 15 2018, 21:51:34)
   [GCC 7.3.0] on linux2
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import caffe
   >>>
   ```

#### bug

+ fatal error: numpy/arrayobject.h没有那个文件或目录

  ```bash
  apt-get install python-numpy
  ```

+ no module name google

  ```bash
  pip install --upgrade google-cloud-translate
  ```

+ numpy

  ```bash
  pip install numpy==1.15.1
  ```

### 2.convert

根据focus层修改方式分为：

#### 1.conv

```shell
#convertToCaffe.py
convertToCaffe(graph, prototxt_path, caffemodel_path)
```

#### 2.passthrough

1. caffe.proto

   passthrough在NNIE中属于扩展层，即NNIE支持的公开但非Caffe标准层。为了使mapper能支持这些网络，需要对原始的 Caffe进行扩展。在caffe.proto文件的 LayerParameter中加入名称为PassThroughParameter的定义（下图中的100004为一个任意 的当前caffe.proto的LayerParameter中没有被占用的数值）；并在文件中定义一个名称为 PassThroughParameter的message定义

   NNIE对该层的支持规格为在保证参数合法有效的情形下，block_height可以为一个1到 255之间的正整数，block_width可以为一个1到255之间的正整数，且block_height和 block_width可以不相等。

2. Add caffe layer

   + `reorg -> passthrough`

   ```bash
   cp passthrough_layer.cpp src/caffe/layers/
   cp passthrough_layer.cu src/caffe/layers/
   cp passthrough_layer.hpp include/caffe/layers/
   ```

   + `src/caffe/proto/caffe.proto`

   ```protobuf
   // LayerParameter next available layer-specific ID: 147 (last added: recurrent_param)
   message LayerParameter {
     optional TileParameter tile_param = 138;
     optional VideoDataParameter video_data_param = 207;
     optional WindowDataParameter window_data_param = 129;
     optional PassThroughParameter pass_through_param = 150; 
   }
   
   // added by lance for yolov5
   message PassThroughParameter{
     optional uint32 num_output = 1 [default = 0];
     optional uint32 block_height = 2 [default = 0];
     optional uint32 block_width = 3 [default = 0];
   }
   ```

3. recompile caffe

   ```bash
   make clean
   export CPLUS_INCLUDE_PATH=/root/anaconda3/include/python3.6m
   make -j16
   make pycaffe -j16
   ```

4. convert

```shell
#convertToCaffe.py
convertToCaffe(graph, prototxt_path, caffemodel_path,  exis_focus=True, focus_concat_name="Concat_40", focus_conv_name="Conv_41")
#_weightloader.py
def _convert_conv(net, node, graph, err, pass_through=0)#改为pass_through=1
```

## 4.wk

### 环境依赖

+ 3559A software

+ aarch64-himix100-linux.tgz

+ Install required softwares:

  ```shell
  sudo apt-get install make libc6:i386 lib32z1 lib32stdc++6 
  sudo apt-get install zlib1g-dev libncurses5-dev ncurses-term 
  sudo apt-get install libncursesw5-dev g++ u-boot-tools:i386
  sudo apt-get install texinfo texlive gawk libssl-dev openssl bc
  ```

  install err:

  + `lib32stdc++6`

    ```shell
    apt update
    apt install lib32stdc++6 
    ```

  + `u-boot-tools:i386`

    ```shell
    apt-get install u-boot-tools
    ```

+ Install the cross compiler and choose the default install path `/opt/hisi-linux`:

  ```shell
  tar -xzf aarch64-himix100-linux.tgz
  cd aarch64-himix100-linux
  chmod +x aarch64-himix100-linux.install
  sudo ./aarch64-himix100-linux.install
  ```

+ Testing installation

  ```shell
  export PATH=/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin:$PATH
  > which aarch64-himix100-linux-gcc
  /opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-gcc
  ```

### 1.base files

+ xxx.prototxt

+ xxx.caffemodel

+ images

  100张resize为模型输入大小的图片用于量化

  + image_list.txt
  + mean.txt (可不用)

```shell
#base files
cp xxx.prototxt yolov5_caffe2wk/models/meta/
cp xxx.caffemodel yolov5_caffe2wk/models/meta/
cp images yolov5_caffe2wk/models/
cp image_list.txt yolov5_caffe2wk/models/meta/
#make cfg
cd  hycrate-caffe-example-public/convert_model/
##############################################################
[prototxt_file] ./meta/model.prototxt
[caffemodel_file] ./meta/model.caffemodel
[net_type] 0
[log_level] 0
[batch_num] 1
[compile_mode] 1 #0：int8 1:int16
[internal_stride] 16
[sparse_rate] 0
[is_simulation] 0
[instruction_name] model_608x608x3_int16
[data_scale] 0.0039215686
[image_list] ./meta/image_list.txt
[image_type] 1
[mean_file] null
[norm_type] 3
##############################################################
```

### 2.model converter

#### 1.Permute层修改

```shell
ERROR: file: parsePermutePara  line: 6149
layer name: [Transpose_174]. error:Permute can only support (0,1,2,3) -> (0,2,3,1) order!
```

 Permute层实测速度和精度差异不是很大，看到博客也有人推荐 prototxt文件里删去它（**总共三处**）

#### 2.Reshape层修改

总共三处

```powershell
#原始格式
layer {
  name: "Reshape_154"
  type: "Reshape"
  bottom: "287"
  top: "305"
  reshape_param {   #此处要改为4维，且第一维为0
    shape {
      dim: 1
      dim: 3
      dim: 9
      dim: 80
      dim: 80
    }
  }
}
```

```shell
#修改后
layer {
  name: "Reshape_154"
  type: "Reshape"
  bottom: "287"
  top: "305"
  reshape_param {   #此处要改为4维，且第一维为0
    shape {
      dim: 0
      dim: 3
      dim: 9
      dim: 6400
    }
  }
}
```

+ 也可以删除Reshape层

## 5.重点问题

### output

yolo v5 与 yolo v3 不同的是，其网络输出不同于v3的那种方式，可以看源码：`yolo.py`

```python
y = x[i].sigmoid()
y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
x[:, 5:] *= x[:, 4:5]  # score = obj_conf * cls_conf
```

对于网络输出结果，先统一做 sigmoid 映射，然后x y w h 再做各自的变化，confidence 与 class_confidence 也是

### class

因为yolo v5里面对于类别是用了多标签loss训练（不互斥多分类），即每个类别都是经 sigmoid 函数映射后输出，故不用比较大小，最后返回一个最大值对应索引。可看看源码的代码：`utils.py`

```python
# Detections matrix nx6 (xyxy, conf, cls)
if multi_label:
    # 注意，yolo v5 不是 softmax分类 ，所以不用遍历出概率最高的那个类别
    i, j = (x[:, 5:] > conf_thres).nonzero().t()  # 只做一个阈值筛选就好
    x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
```

作者只做了个阈值筛选就返回所需要的类别索引，如果你说这样会不会有两个类别及以上的情况出现，对于这种，我只能说你要么模型没训练好，要么取数据出现失误，一个训练好的模型是不会的。

###  nms

`utils.py`

```python
# 先拿类别索引乘以一个较大值 max_wh（4096）
c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes，5 是类别对应的索引
boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
```

作者针对每个类别都有一个 bbox偏移，在做 nms 之前，先给类别索引乘以一个较大值，再让bbox坐标值加上偏移值，这样意义是什么呢。试想每个类别都加上一个较大值，等同于每个类别都在专属于本类的坐标系里做 nms，这样做nms的时候，不用和以前一样按类别进行nms操作，而是直接做计算就行了。避免了那些 IOU>阈值 但不属于同一类的bbox被删除。