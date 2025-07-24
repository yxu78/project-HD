# project-HD
to prove the applicability of the open source code in https://github.com/lamm-mit/HierarchicalDesign/tree/main
## 1.跑通原代码
取原数据集中的10张微结构图及其应力数据为训练集，训练原代码的AItool，并利用训练后的AItool进行预测。具体流程如下  
### 1.1进入工作环境
在本地终端运行
```
ssh yxu78@xionglab3.mae.ncsu.edu
```
```
cd /data8
conda activate HierarchicalDesign
jupyter lab
```
再开一个本地终端窗口，运行
```
ssh -L 8888:localhost:8888 yxu78@xionglab3.mae.ncsu.edu (其中8888是上一个窗口给出的port,以具体情况为准)
```
用本地浏览器打开第一个窗口给出的链接进入jupyter lab，进入./test2文件夹，开一个名为test2.ipynb的kernel开始写代码块  
设置数据目录路径和csv文件路径(后续工作视情况而定)
```
data_dir = '/data8/test2/'
csvfile = '/data8/test2/test.csv'
```
### 1.2 VQ-VAE Model
[VQ-VAE具体操作](./VQ-VAE.md)  

### 1.3 Generative attention-diffusion model
```
import os,sys
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    #加载第0块CPU
```
常规工具
```
import shutil    #文件操作
from torchvision.utils import save_image, make_grid    #保存或拼接生成图片
import torch    #深度学习库
from sklearn.model_selection import train_test_split
```
导入前面已经写好的VQ-VAE模块、工具函数等
```
from HierarchicalDesign import VectorQuantize, VQVAEModel, Encoder_Attn ,Decoder_Attn,count_parameters
from HierarchicalDesign import get_fmap_from_codebook
```
device设为0号GPU，检查实际能看到多少块GPU并打印
```
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

device='cuda:0'

num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
available_gpus
```
```
device
```
```
print("Torch version:", torch.__version__)
```
PyTorch核心库
```
import torch    #主库，包含Tensor定义和计算
import torch.nn as nn    #包含常用神经网络模块
import torch.nn.functional as F    #定义无状态神经网络函数接口
```
```
import numpy as np    #科学计算

#from tqdm import tqdm    #进度条
from tqdm.autonotebook import tqdm
```
图像处理工具
```
from torchvision.utils import save_image, make_grid    #Tensor保存图片，拼接，可视化
import torch.nn.functional as F
from torchvision import datasets, transforms, models    #数据集接口，图像预处理，预训练模型
```
```
from sklearn.metrics import r2_score    #回归拟合评估指标
```
可视化与数据处理
```
import matplotlib.pyplot as plt    #绘图

import ast
import pandas as pd    #数据分析库
import numpy as np  
from einops import rearrange    #张量重排
```
数据加载，时间，图像对象
```
from torch.utils.data import DataLoader,Dataset
from torchvision.io import read_image
import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image
import time
to_pil = transforms.ToPILImage()

from torchvision.utils import save_image, make_grid
```
