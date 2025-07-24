# project-HD
to prove the applicability of the open source code in https://github.com/lamm-mit/HierarchicalDesign/tree/main
## 我所做的事
### 1.跑通原代码
取原数据集中的10张微结构图及其应力数据为训练集，训练原代码的AItool，并利用训练后的AItool进行预测。具体流程如下  
#### 1.1进入工作环境
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
设置数据目录路径和csv文件路径
```
data_dir = '/data8/test2/'
csvfile = '/data8/test2/test.csv'
```
#### 1.2 Model 1: VQ-VAE的训练
导入os库，选择使用第0张GPU
```
import os

#Which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
引入tqdm进度条工具，自动适应jupyter notebook
```
#from tqdm import tqdm
from tqdm.autonotebook import tqdm
```
PyTorch和常用工具包的引入，后两行是导入自己的模型模块
```
import torch

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid
from PIL import Image

from HierarchicalDesign import VectorQuantize, VQVAEModel, Encoder_Attn ,Decoder_Attn,count_parameters
from HierarchicalDesign import get_fmap_from_codebook
```
设备选择，判断使用GPU还是CPU
```
CPUonly=False
DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

if CPUonly == True:
     print ("CPU!")
     device = torch.device("cpu")
```
打印PyTorch版本信息
```
print("Torch version:", torch.__version__)
```
设置图像分辨率512*512，设备选择，训练参数设置（每次传入模型的样本数量12，学习率0.0002），设置模型保存路径，设置图像文件路径
```
im_res_VAE=512 
 
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

batch_size = 12
img_size = (im_res_VAE, im_res_VAE) # (width, height)

lr = 2e-4
prefix = './VQ_VAE_results/'
    
#relative directory of images - added to whatever is stored in CSV file
data_dir = '/data8/test2/'

if not os.path.exists(prefix):
        os.mkdir (prefix)
```
scale把图像像素值从[0,1]映射到[-1,1]，用于神经网络训练。unscale恢复回[0,1]，用于可视化
```
def scale_image(image2):
    return image2 * 2. - 1.0

def unscale_image(image2):
    image2=(image2 +1. )/ 2. 
    image2=image2.clamp(0, 1)
    return image2
```
读取图像路径，打开图像并标准化为RGB，数据增强（transform），返回归一化图像Tensor
```
class ImageDataset_ImagePairs(Dataset):
    def __init__(self,paths2,transform):
 
        self.paths2=list(paths2)
        self.transform=transform

    def __len__(self):
        return len(self.paths2)

    def __getitem__(self,index):
       
        fname_image=data_dir+self.paths2[index]
         
        im2_pil = Image.open(fname_image).convert('RGB')#Image.fromarray(image2)

        image2=self.transform(im2_pil)
        image2=scale_image(image2)    
    
        return (image2)
```
定义数据增强（TRansform），HorizontalFlip/VerticalFlip数据增强，Resize把图像调整到同一分辨率512*512，ToTensor
```
def ImagePairs_load_split_train_test(csvfile, valid_size = .2, im_res=256, batch_size_=16, ):
    
    train_transforms = transforms.Compose([#transforms.RandomRotation(30),  # data augmentations are great
                                       #transforms.RandomResizedCrop(224),  # but not in this case of map tiles
                                       #transforms.RandomHorizontalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.Resize((im_res,im_res)),
                                           #transforms.Normalize([0.485, 0.456, 0.406],  
                                           #                  [0.229, 0.224, 0.225]),
                                       transforms.ToTensor(),
                                           #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                       ])
    test_transforms = transforms.Compose([#transforms.RandomRotation(30),  # data augmentations are great
                                       #transforms.RandomResizedCrop(224),  # but not in this case of map tiles
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.Resize((im_res,im_res)),
                                           #transforms.Normalize([0.485, 0.456, 0.406],  
                                           #                  [0.229, 0.224, 0.225]),
                                       #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       transforms.ToTensor(),
                                          # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                       ])
```
读取csv中的图像路径列
```
df = pd.read_csv(csvfile)
imgpair = df["microstructure"]
```
将图像路径列表划分为训练集和测试集，确定训练集和测试集的比例valid_size
```
X_train, X_test = train_test_split(imgpair, test_size=valid_size, random_state=451)
```
使用定义的Dataset加载图像并应用transform
```
train_data=ImageDataset_ImagePairs(X_train,train_transforms)
test_data=ImageDataset_ImagePairs(X_test,test_transforms)
```
每次从训练/测试集中成批(batch)读取图像数据，返回训练与测试数据加载器，确定csv文件路径和测试集占比
```
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_)
return trainloader, testloader
csvfile = '/data8/test2/test.csv'
train_loader, test_loader  = ImagePairs_load_split_train_test(csvfile, .1, im_res=im_res_VAE,
                                                              batch_size_=batch_size)
```
打印训练集/测试集统计和总图像数
```
print ("Number of training batches: ", len(train_loader), "batch size= ", batch_size, "total: ",len(train_loader)*batch_size )
print ("Number of test batches: ", len(test_loader), "batch size= ", batch_size, "total: ",len(test_loader)*batch_size)

print("TOTAL images (account for full batches): ", len(train_loader)*batch_size+len(test_loader)*batch_size )
```
```
im_res_VAE
```

```
hidden_dim = 16   #this is how much depth the sequence will have

num_codebook_vectors=128    #码本中词元个数
embedding_dim = 256    #每个codebook向量维数

output_dim = 3    #RGB图像通道为3
input_dim = 3

chann_enc=  [64, 64, 64, 128, 128, 128, 256, 256]    #编辑器/解码器各层通道数
chann_dec= [256, 256, 256, 128, 64, 64, 64] 

codebook = VectorQuantize(          #码本模块
            dim = hidden_dim,
            codebook_dim = embedding_dim,
            codebook_size = num_codebook_vectors,
            decay = 0.9,
            commitment_weight = 1.,
            kmeans_init = True,
            accept_image_fmap = True,
            use_cosine_sim = True,
            orthogonal_reg_weight = 2., #0.
        )

###########################################################################
encoder = Encoder_Attn( image_channels=input_dim, latent_dim=hidden_dim,      #编码器
                      channels = chann_enc, 
                      start_resolution_VAE_encoder=512,
                 attn_resolutions_VAE_encoder = [16],
                 num_res_blocks_encoder=2,
                      )
decoder = Decoder_Attn( image_channels=input_dim, latent_dim=hidden_dim,       #从VQcode的latent表达重建图像
                       channels = chann_dec,
                       attn_resolutions_VAE_decoder=[16],
                 start_resolution_VAE_decoder= 16, 
                 num_res_blocks_VAE_decoder=3,
                      )
###########################################################################
model = VQVAEModel(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(DEVICE)    #组建完整模型，放入GPU设备
count_parameters(model)    #统计并输出模型总参数数和可训练参数数
```
