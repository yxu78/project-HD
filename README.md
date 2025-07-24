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
定义把Tensor拼图显示保存的函数(draw)，反解码出图像的函数(generate)
```
#########################################################
# CODE BASE: Helpfunctions for sampling, drawing, etc.
#########################################################    
    
to_pil = transforms.ToPILImage()    #准备tensor到PIL的转换器

def draw_sample_image(x, postfix, fname=None,  dpi=600, padding=0,):
  
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=padding, normalize=True), (1, 2, 0)))
    if fname != None:
        plt.savefig(fname, dpi=dpi)
    plt.show()
    
def generate_from_indices (model, indices=None, flag=0, show_codes=False, num_codebook_vectors=128):
    if indices==None:
        print ("Indices not provided, generate random ones...")
        indices = torch.randint(0, num_codebook_vectors, (1, fmap_lin,fmap_lin))
    else:
        indices=torch.reshape (indices, (1, fmap_lin,fmap_lin))
    fmap=get_fmap_from_codebook (model, indices)
    print ("Fmap shape: ", fmap.shape)
    x_hat   = model.decoder(fmap) #non-snapped
    x_hat=unscale_image(x_hat)
    
    image_sample = to_pil(  x_hat[0,:].cpu()  )
    fname2= prefix+ f"generate_samples_{flag}.png"
        
    image_sample.save(f'{fname2}', format="PNG",  subsampling=0  ) 
    
    plt.imshow (image_sample)
    plt.axis('off')
    plt.show()
    
    if show_codes:

        iu=0
        
        print (indices.shape)
        print (indices[iu,:].flatten())
        plt.plot (indices[iu,:].cpu().detach().flatten().numpy(),label='Codebook vector')
        plt.legend()
        plt.xlabel ('Codebook index')
        plt.ylabel ('Codebook vector ID')
        plt.show()

        plt.figure(figsize=(8, 3))
        plt.imshow (indices[iu,:].cpu().detach().flatten().unsqueeze (0).numpy(), cmap='plasma',
                   aspect=6.)
        ax = plt.gca()

        ax.get_yaxis().set_visible(False) 
        plt.clim (0, num_codebook_vectors)
        plt.colorbar()
        plt.show()   

        plt.imshow (indices[iu,:].cpu().detach().numpy(), cmap='plasma',)
        plt.clim (0, num_codebook_vectors)
        plt.colorbar()
        plt.show() 
```
定义sample函数
```
def sample (model, test_loader, samples = 16, fname1 = None, fname2 = None, 
            batches=1, save_ind_files=False,flag=0,indices_hist=True,
           show_codes=False):
    model.eval()
    e=flag
    indices_list=[]
    batches=min (len (test_loader), batches)
    print ("Number of batches produced: ", batches)
    with torch.no_grad():

        for batch_idx, (x) in enumerate(tqdm(test_loader)):

            x = x.to(DEVICE)
            
            x_hat, indices, commitment_loss,    = model(x)

            with torch.no_grad():
                x_enc =model.encode(x)
             
            with torch.no_grad():
                z, indices=model.encode_z(x)
            
            with torch.no_grad():
                x_hat_snapped, z_quant =  model.decode_snapped(z,  )
    
            samples= min([samples, x.shape[0]])
             
            x_hat=unscale_image(x_hat)
           
            x=unscale_image(x)
            
            indices_list.append (indices.cpu().detach().flatten().numpy())

            if save_ind_files:
               
                for iu in range (samples):
                    fname2= prefix+ f"recon_samples_{e}_{batch_idx}_{iu}.png"
                    print ("Save individual samples ", fname2)

                    image_sample = to_pil(  x[iu,:].cpu()  )

                    image_sample.save(f'{fname2}', format="PNG",  subsampling=0  )
                    
                    fname2= prefix+ f"recon_samples_{e}_{batch_idx}_{iu}.png"
                    print ("Save individual samples ", fname2)

                    image_sample = to_pil( x_hat[iu,:].cpu()  )

                    image_sample.save(f'{fname2}', format="PNG",  subsampling=0  )
                    
                    plt.imshow (image_sample)
                    plt.axis('off')
                    plt.show()
                    if show_codes:
                        
                        plt.plot (indices[iu,:].cpu().detach().flatten().numpy(),label='Codebook vectors')
                        plt.legend()
                        plt.xlabel ('Codebook index')
                        plt.ylabel ('Codebook vector ID')
                        plt.show()
                        
                        indices_list.append (indices[iu,:].cpu().detach().flatten().numpy())
                        
                        plt.figure(figsize=(8, 3))
                        plt.imshow (indices[iu,:].cpu().detach().flatten().unsqueeze (0).numpy(), cmap='plasma',
                                   aspect=6.)
                        ax = plt.gca()
 
                        ax.get_yaxis().set_visible(False) 
                        plt.clim (0, num_codebook_vectors)
                        plt.colorbar()
                        plt.show()   
                        
                        plt.imshow (indices[iu,:].cpu().detach().numpy(), cmap='plasma',)
                        plt.clim (0, num_codebook_vectors)
                        plt.colorbar()
                        plt.show() 

            draw_sample_image(x[:samples], "Ground-truth images", fname1)
            draw_sample_image(x_hat[:samples], "Reconstructed images", fname2)

            if batch_idx>=(batches-1):
                
                if indices_hist:
                    indices_list=np.array (indices_list).flatten()
                    
                    n_bins =num_codebook_vectors 

                    # Creating histogram
                    fig, axs = plt.subplots(1, 1,
                                            figsize =(4, 3),
                                            tight_layout = True)

                    axs.hist(indices_list, bins = n_bins, density=True)
                    plt.xlabel("Codebook indices")
                    plt.ylabel("Frequency")
                   
                    plt.show()
                
                break

    return
```
```
load_model=False    #第一次，没有已经训练的模型可以载入
```
```
from torch.optim import Adam
from torch import nn

mse_loss = nn.MSELoss()    
optimizer = Adam(model.parameters(), lr=lr)   
isample=1 #sample frequency

loss_list=[]
step=0

epochs = 128

print_step = 1
icheckpt = 5

min_loss = 999999999
```
执行训练的函数
```
def train_steps (model, train_loader, test_loader, startep=0, epochs=500,min_loss = 999):
    print("Start training VQ-VAE...")    #epochs训练多少轮

    startstep=len (train_loader)*startep    #已经走过多少步
    step=startep
    for epoch in range(epochs):
        overall_loss = 0
        losscoll=0
        for batch_idx, (x) in enumerate(tqdm(train_loader)):
                x = x.to(DEVICE)    #把图片搬到CPU/GPU
                model.train()    #训练模式
                optimizer.zero_grad()

                x_hat, indices, commitment_loss = model(x)
                recon_loss = mse_loss(x_hat, x)    重建误差（均方差）

                loss =  recon_loss + commitment_loss #+ codebook_loss    #总损失

                loss.backward()    #反向优化
                optimizer.step()

                losscoll=losscoll+loss.item ()

        if step % print_step ==0:     #打印训练日志
            print("epoch:", epoch + 1 + startep, "  step:", batch_idx + 1 +startstep, "  recon_loss:", recon_loss.item(), #"  perplexity: ", perplexity.item(), 
                  "\n\t\tcommit_loss: ", commitment_loss.item(), 
                  " total_loss: ", loss.item())

        loss_list.append (losscoll/len (train_loader))    #记录loss曲线

        if step % isample == 0:    #取样，可视化

            fname1 = f'{prefix}/or_sample-epoch_{step+startstep}.png'
            fname2 = f'{prefix}/rec_sample-epoch_{step+startstep}.png'

            sample (model, test_loader, fname1=fname1, fname2=fname2, samples=16)
            plt.plot (loss_list)
            plt.xlabel ('Iterations')
            plt.ylabel ('Total loss')
            plt.show()

        if losscoll/len (train_loader)<min_loss:    #判断是否刷新最小loss
            min_loss=losscoll/len (train_loader)
            print ("Smaller loss found")
            save=True

        if save or (step % icheckpt ==0):    #保存checkpoint

            fname = f'{prefix}/model-epoch_{step}.pt'
            torch.save(model.state_dict(), fname)
            print ("Saved checkpoint: ", fname)

            save=False

        step=step+1    #迭代计数
```
开始训练,运行这一步的时候会生成很多东西。tqdm会为每个epoch画进度条。每隔print_step打印loss信息。每隔isample从测试集取图片，走encoder到decoder，绘制loss曲线。保存checkpoint。
```
if not  load_model:
    train_steps (model, train_loader, test_loader, startep=0, epochs=epochs)
```
拿几批数据走一遍VQ-VAE，看看重建效果
```
sample (model, test_loader, batches=4)
```
从测试集取一些(batch)测试图像，通过模型重建这些图像，保存和显示原始图像(or_test.png)与重建图像(rec_test.png)  
这一步可以用来查看模型训练效果
```
sample (model, test_loader, fname1 =f'{prefix}/or_test.png', fname2=f'{prefix}/rec_test.png',samples = 32,
       batches=4)
```
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
