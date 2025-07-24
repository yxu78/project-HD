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
