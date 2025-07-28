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
from torch.utils.data import DataLoader,Dataset    #将Dataset封装为加载器
from torchvision.io import read_image    #读取图片为Tensor
import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image    #打开处理图片
import time    #记录耗时
to_pil = transforms.ToPILImage()    #将Tensor转为PIL图片对象

from torchvision.utils import save_image, make_grid
```
```
#Define hyperpamaters
batch_size_=16    #每次处理16张图片

im_res_VAE=512     #输入与输出图像的分辨率
im_res_final=im_res_VAE

###########
hidden_dim = 16   #this is how much depth the sequence will have

num_codebook_vectors=128     
embedding_dim = 256 #for codebook

output_dim = 3    
input_dim = 3

chann_enc=  [64, 64, 64, 128, 128, 128, 256, 256]      #卷积通道数
chann_dec= [256, 256, 256, 128, 64, 64, 64]
```
自注意力放在16*16那层
```
codebook = VectorQuantize(    #词典
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
encoder = Encoder_Attn( image_channels=input_dim, latent_dim=hidden_dim,        #用词典描述世界
                      channels = chann_enc, 
                      start_resolution_VAE_encoder=512,
                 attn_resolutions_VAE_encoder = [16],
                 num_res_blocks_encoder=2,
                      )
decoder = Decoder_Attn( image_channels=input_dim, latent_dim=hidden_dim,       #读词典重建世界
                       channels = chann_dec,
                       attn_resolutions_VAE_decoder=[16],
                 start_resolution_VAE_decoder= 16, 
                 num_res_blocks_VAE_decoder=3,
                      )
###########################################################################
VQVAEmodel = VQVAEModel(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(device)    #把模型加载到device
count_parameters(VQVAEmodel)    #打印总参数量
```
```
#Test model and shapes
print ("Image resolution: ", im_res_VAE,  "x", im_res_VAE)
images = torch.randn(4, 3, im_res_VAE, im_res_VAE).cuda()

with torch.no_grad():
    z, indices  =  VQVAEmodel.encode_z(images, )
print ("z ", z.shape )
print ("infices ", indices.shape)

with torch.no_grad():
    images_hat =  VQVAEmodel.decoder(z,  )
print ("image  snapped shape ", images_hat.shape)

with torch.no_grad():
    images_hat_snapped, z_quant =  VQVAEmodel.decode_snapped(z,  )

    print ("image_hat snapped shape ", images_hat_snapped.shape)
print ("z quant ", z_quant.shape)

size_VAE=z.shape [1]
z_reshaped=torch.reshape(z, (z.shape[0],z.shape[1],z.shape[2]**2))

fmap_lin = z.shape[2]
print (f"VAE will produce sequences of shape (b,  {z_reshaped.shape})")
```
把磁盘中第125轮训练保存的VQ-VAE权重载入到内存中的VQVAEmodel
```
VQVAEmodel.load_state_dict(torch.load(f'./VQ_VAE_results/model-epoch_125.pt'))    #具体文件名视情况而定
```
从测试集中取样评估VQ-VAE的重建质量
```
def sample_VAE (model, test_loader, samples = 16, fname1 = None, fname2 = None):
    model.eval()

    with torch.no_grad():

        for batch_idx, (_, x) in enumerate(tqdm(test_loader)):    #读取第一批图像

            x = x.to(device)
        
            x_hat, indices, commitment_loss,    = model(x)

            with torch.no_grad():
                x_enc =model.encode(x)
            
            with torch.no_grad():
                z, indices=model.encode_z(x)
            
            with torch.no_grad():
                x_hat_snapped, z_quant =  model.decode_snapped(z,  )

            break
    samples= min([samples, x.shape[0]])
    
    x_hat=unscale_image(x_hat)
    x_hat_snapped=unscale_image(x_hat)
    x=unscale_image(x)

    draw_sample_image(x[:samples], "Ground-truth images", fname1)
    draw_sample_image(x_hat[:samples], "Reconstructed images", fname2)
    draw_sample_image(x_hat_snapped[:samples], "Reconstructed images, snapped", fname2)
        
    return
```
```
def draw_sample_image(x, postfix, fname=None, dpi=600,padding=0):
  
    plt.figure(figsize=(8,8))    #新建8*8画布
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=padding, normalize=True), (1, 2, 0)))
    if fname != 0:
        plt.savefig(fname, dpi=dpi)
    plt.show()    #在notebook渲染图片
```
为后续数据加载和训练做环境准备
```
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import seaborn as sns
 
import torchvision
 
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from functools import partial, wraps
print("Torch version:", torch.__version__)
```
```
from sklearn.preprocessing import RobustScaler
from PIL import Image
import ast
import pandas as pd
import numpy as np
```
像素值缩放/反缩放
```
def scale_image(image2):
    return image2 * 2. - 1.0
 
def unscale_image(image2):
    image2=(image2 +1. )/ 2. 
    image2=image2.clamp(0, 1)
    return image2
```
设置路径
```
data_dir = '/data8/test2/file/'
csvfile = '/data8/test2/file/test_fixed.csv'
```
自定义数据集
```
class ImageDataset_ImagePairs(Dataset):
    def __init__(self,X_data,paths2,transform):
       
        self.X_data = X_data    #已在内存中的图像
        
        self.paths2=list(paths2)    #另一组图像的磁盘路径
        self.transform=transform    #统一的图像预处理

    def __len__(self):
        return len(self.paths2)

    def __getitem__(self,index):
        
        image1= self.X_data[index]
       
        im2_pil = Image.open(data_dir+self.paths2[index]).convert('RGB')#Image.fromarray(image2)

        image2=self.transform(im2_pil)
        image2=scale_image(image2)    
      
        return (image1, image2)
```
数据加载与拆分函数，打印训练/测试集一共有多少batch
```
def ImagePairs_load_split_train_test(csvfile, valid_size = .2, im_res=256, batch_size_=16, max_l=32,
                                    X_min=None, X_max=None):
    train_transforms = transforms.Compose([#transforms.RandomRotation(30),  # data augmentations are great
                                       #transforms.RandomResizedCrop(224),  # but not in this case of map tiles
                                       #transforms.RandomHorizontalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        #transforms.RandomVerticalFlip(),
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
                                       # transforms.RandomVerticalFlip(),
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.Resize((im_res,im_res)),
                                           #transforms.Normalize([0.485, 0.456, 0.406],  
                                           #                  [0.229, 0.224, 0.225]),
                                       #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       transforms.ToTensor(),
                                          # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                       ])

    df=pd.read_csv(csvfile)      #读取图片和应力
    imgpair=df["microstructure"]
    x=df['stresses']

    max_l=32      #只保留前32个应力点
    X_min=None
    X_max=None

    xs=[]
    for xi in x:
        xentry=ast.literal_eval(xi)
        xs.append(xentry[:max_l])
    X=np.asarray(xs)

    if X_min==None:
        X_min=np.min(X)
    else:
        print ("use provided X_min", X_min)
    if X_max==None:
        X_max=np.max(X)
    else:
        print ("use provided X_max", X_max)
    
    X=(X-X_min)/(X_max-X_min)*2-1 #Normalize range -1 to 1
    
    print ("Check X after norm  ", np.min(X), np.max(X))
    
    X=torch.from_numpy(X).float()
    if valid_size>0:
       #划分训练/测试集
        X_train, X_test, y_train, y_test =train_test_split(X,imgpair,test_size=valid_size, random_state=451)
    else:
        train_dataset = ImageDataset_ImagePairs(X, imgpair,test_transforms)
        trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=False)
      
        return trainloader, None, X_max, X_min
    
    train_data=ImageDataset_ImagePairs(X_train,y_train,train_transforms)
    test_data=ImageDataset_ImagePairs(X_test,y_test,test_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_)
    return trainloader, testloader, X_max, X_min

csvfile = '/data8/test2/file/test_fixed.csv'

train_loader, test_loader, X_max, X_min = ImagePairs_load_split_train_test(csvfile, .1, 
                                                                           im_res=im_res_final,
                                                                           batch_size_=batch_size_, 
                                                                           max_l=32)

print ("Number of training batches: ", len(train_loader), "batch size= ", batch_size_, "total: ",len(train_loader)*batch_size_ )
print ("Number of test batches: ", len(test_loader), "batch size= ", batch_size_, "total: ",len(test_loader)*batch_size_)

print("TOTAL images (account for full batches): ", len(train_loader)*batch_size_+len(test_loader)*batch_size_ )

print ("X_max", X_max, "X_min ", X_min)
```
```
sample_VAE (VQVAEmodel, test_loader, fname1 =f'or_test.png', fname2=f'rec_test.png',samples = 3)
```
Training loop
```
#from pytorch_lightning.utilities.seed import seed_everything
#from pytorch_lightning.utilities.seed import reset_seed, isolate_rng
```
```
def sample_sequence (model,
                X=[[10, 10, 8, 3, 13, 13, 14, 7, 3, 10 ]],
                 flag=0,
                     cond_scales=1.,foldproteins=False,
                     seed=None,plot_original_data=False,GT =None,
               ):
    steps=0
    e=flag

    if seed !=None:
        print ("Set seed to: ", seed)
        seed_everything(seed)
    
    print (f"Producing {len(X)} samples...")
    
    X=torch.Tensor(X)
    X=(X-X_min)/(X_max-X_min)*2-1 #Normalize range -1 to 1
    for iisample in range (len (X)):
        
        
        
        X_cond= (X[iisample] ).to(device).unsqueeze (0)
        
        print (X_cond.shape)
        result=model.sample ( X_cond,stop_at_unet_number=train_unet_number ,
                                 cond_scale=cond_scales )
            
      
        if plot_original_data:
            
            cond_unscaled=(X_cond[0,:].cpu().detach().numpy()+1)/2.*(X_max-X_min)+X_min
            
            plt.plot (cond_unscaled,label= f'Required stress-strain data')
            plt.ylim ((0, 1.1))
            #plt.plot (GT[samples,0,:],label= f'GT {0}')
            plt.legend()
            
            fname4= prefix+ f"input_to_samples_{e}_{steps}.jpg"
            plt.savefig(fname4, dpi=300)   
            plt.show()
    
       
        result=torch.reshape(result, (result.shape[0],result.shape[1], sqer_z,sqer_z))
        
        model.imagen.VAE.eval()
        with torch.no_grad():
            
            result,_ =  model.imagen.VAE.decode_snapped(result,  )
            result=unscale_image(result)
        

        fname2=prefix+ f"sampld_from_X_{flag}_condscale-{str (cond_scales)}_{e}_{steps}.jpg"
        

        draw_sample_image(result[0,:], "Predicted microstructure", fname2)  

 
        steps=steps+1
       
    reset_seed()
```
```
def draw_sample_image(x, postfix, fname=None):
  
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=0, normalize=True), (1, 2, 0)))
    if fname != None:
        plt.savefig(fname, dpi=300)
    plt.show()
```
```
def sample_loop (model,
                train_loader,
                cond_scales=[7.5], #list of cond scales - each sampled...
                num_samples=2, #how many samples produced every time tested.....
                 
                 flag=0,foldproteins=False,
                  seed=None,save_ind_files=False,
                 plot_structure=False,plot_original_data=False,do_forward=False,number_batches=1,strain=None,
                 csv_for_data=None,
                 mix_bernouilli=False, bernouilli_prob=0.5, #if True sample two solutions and mix accordig to Bernouilli
                 #init_images=None,
                 repeat_sampling=1,
               ):
    steps=0
    e=flag
    
    if seed !=None:
        print ("Set seed to: ", seed)
        seed_everything(seed)
    if csv_for_data!=None:
        print ("Open CSV file to save...", csv_for_data)
        stresses_list =[]
        image_list =[]
        
    for item  in train_loader:

            X_train_batch= item[0].to(device)
            y_train_batch=item[1].to(device)

            GT=y_train_batch.cpu().detach() 
            
            print (X_train_batch.shape, y_train_batch.shape)
            num_samples = min (num_samples,y_train_batch.shape[0] )
            print (f"Producing {num_samples} samples..., batch size ={y_train_batch.shape[0]}")
 
            for iisample in range (len (cond_scales)):
        
                cond_unscaled=(X_train_batch.cpu().detach().numpy()+1)/2.*(X_max-X_min)+X_min
            
                
                result=model.sample ( X_train_batch,stop_at_unet_number=train_unet_number ,
                                         cond_scale=cond_scales[iisample])
                
                if repeat_sampling>1:
                    print (f"Repeat sampling {repeat_sampling} times")
                    for irep in range (repeat_sampling-1):
                        print (f"Repeat {irep}")
                        result=model.sample ( X_train_batch,stop_at_unet_number=train_unet_number ,
                                             cond_scale=cond_scales[iisample],
                                            init_images=result)
                print (result.shape)
                
                if mix_bernouilli:
                    result_2=model.sample ( X_train_batch,stop_at_unet_number=train_unet_number ,
                                         cond_scale=cond_scales[iisample])
                    print (f"Repeat sampling {repeat_sampling} times")
                    if repeat_sampling>1:
                        for irep in range (repeat_sampling-1):
                            print (f"Repeat {irep}")
                            result2=model.sample ( X_train_batch,stop_at_unet_number=train_unet_number ,
                                                 cond_scale=cond_scales[iisample],
                                                init_images=result2)

                    mask = torch.bernoulli(torch.full(result.shape, bernouilli_prob)).int().to(device)
                    reverse_mask = torch.ones(result.shape).int().to(device) - mask

                    result = result * mask + result_2 * reverse_mask

                
                result=torch.reshape(result, (result.shape[0],result.shape[1],sqer_z,sqer_z))
                model.imagen.vit_vae.eval()
                
                with torch.no_grad():
                    result,_ =  model.imagen.vit_vae.decode_snapped(result, ) 
                
                result=unscale_image(result)
                GT=unscale_image(GT)
                 
                
                fig, axs = plt.subplots(1, num_samples, figsize=(4.5 * num_samples, 3))

                # 如果 num_samples==1，plt.subplots 会直接返回一个 Axes 而不是列表，
                # 这里把它包装成长度为1的列表，后面统一用 axs[i] 就不会出错了

                if num_samples == 1:
                    axs = [axs]

                for iu in range (num_samples):
                    
                    if strain==None:
                        axs[ iu].plot (cond_unscaled[iu,:] ,
                                        label= f'Input, spl {iu}')
                      
                    else:
                        lstrai=min (len (strain),cond_unscaled[iu,:].shape[0] )
                        axs[ iu].plot (strain[:lstrai], cond_unscaled[iu,:lstrai] ,
                            label= f'Input, spl {iu}')
                        axs[ iu].set_xlabel('Strain')
                        axs[ iu].set_ylabel('Stress $\sigma$ (MPa)')
                    axs[ iu].set_aspect(0.5, adjustable='box')
                    
                    axs[ iu].set_ylim([0,1.2])
                    axs[ iu].legend()
                    
                    if save_ind_files:    
                        fname2= prefix+ f"recon_samples_{e}_{steps}_{iu}.png"
                        print ("Save individual samples ", fname2)
                      
                        image_sample = to_pil( result[iu,:].cpu()  )

                          
                        image_sample.save(f'{fname2}', format="PNG",  subsampling=0  )
                        fname2= prefix+ f"recon_samples_{e}_{steps}_{iu}.tiff"
                        
                        image_sample.save(f'{fname2}', format="TIFF",  subsampling=0  )
                        
                        if csv_for_data!=None:
                            stresses_list.append(np.array2string(cond_unscaled[iu,:lstrai], separator=', ')  )
                            image_list.append(fname2)
                        
                        
                       
                plt.savefig(f'{prefix}/stress_strain_input__{e}_{steps}.svg', dpi=300)#, quality = 95)  
                plt.savefig(f'{prefix}/stress_strain_input__{e}_{steps}.png', dpi=300)#, quality = 95)  
                plt.show()
 
                fname1=  prefix+ f"OR_samples_{e}_{steps}.png"
                fname2= prefix+ f"recon_samples_{e}_{steps}.png"
                draw_sample_image(GT[:num_samples,:], "Ground-truth images", fname1)
    
                draw_sample_image(result[:num_samples,:], "Reconstructed images", fname2)   
         
                     
            steps=steps+1
            if steps>=number_batches:
                
                if csv_for_data!=None:
                    
                    print("Now save CSV file:" , csv_for_data)
                    df = pd.DataFrame()
                    df['stresses'] =  stresses_list
                    df['microstructure'] =  image_list
                    df.to_csv(csv_for_data)
                
                
                break                        
    if csv_for_data!=None:

        print("Now save CSV file:" , csv_for_data)
        df = pd.DataFrame()
        df['stresses'] =  stresses_list
        df['microstructure'] =  image_list
        df.to_csv(csv_for_data)
```
```
def train_loop (model,
                train_loader,
                test_loader,
                optimizer=None,
                print_every=10,
                epochs= 300,
                start_ep=0,
                start_step=0,
                train_unet_number=1,
                print_loss=1000,
                trainer=None,
                plot_unscaled=False,
                max_batch_size=4,
                save_model=False,
                cond_scales=[7.5], #list of cond scales - each sampled...
                num_samples=4, #how many samples produced every time tested.....
                foldproteins=False,
                seed=None,strain=None,
                val_loader = None,save_ind_files=False,
                 
               ):
    

    if not exists (trainer):
        if not exists (optimizer):
            print ("ERROR: If trainer not used, need to provide optimizer.")
    if exists (trainer):
        print ("Trainer provided... will be used")
    steps=start_step
    
    loss_total=0
    for e in range(1, epochs+1):
            start = time.time()

            torch.cuda.empty_cache()
            print ("######################################################################################")
            start = time.time()
            print ("NOW: Training epoch: ", e+start_ep)

            # TRAINING
            train_epoch_loss = 0
            model.train()
            
            print ("Loop over ", len(train_loader), " batches (print . every ", print_every, " steps)")


            for item  in train_loader:


                X_train_batch= item[0].to(device)

                y_train_batch=item[1].to(device)
            
                
                model.imagen.vit_vae.eval()
                
                with torch.no_grad():
                    y_train_batch  =  model.imagen.vit_vae.encode(y_train_batch, )
                  
                    
                    
                y_train_batch=torch.reshape(y_train_batch, (y_train_batch.shape[0],
                                                            y_train_batch.shape[1],
                                                            y_train_batch.shape[2]**2))
                
                
                if exists (trainer):
                    loss = trainer(
                            X_train_batch, y_train_batch,#.unsqueeze(1) ,
                            unet_number=train_unet_number,
                            max_batch_size = max_batch_size,     
                        )
                    trainer.update(unet_number = train_unet_number)

                else:
                    optimizer.zero_grad()
                    loss=model ( X_train_batch, y_train_batch.unsqueeze(1) ,unet_number=train_unet_number)
                    loss.backward( )
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                  
                    optimizer.step()

               
                loss_total=loss_total+loss.item()


                if steps % print_every == 0:
                    print(".", end="")

                if steps>0:
                    if steps % print_loss == 0:

                        norm_loss=loss_total/print_loss
                        print (f"\nTOTAL LOSS at epoch={e+start_ep}, step={steps}: {norm_loss}")

                        loss_list.append (norm_loss)
                        loss_total=0

                        plt.plot (loss_list, label='Loss')
                        plt.legend()

                        outname = prefix+ f"loss_{e}_{steps}.jpg"
                        plt.savefig(outname, dpi=200)
                        plt.show()
                        
                        ####
                        #num_samples = min (num_samples,y_train_batch.shape[0] )
                        #print (f"Producing {num_samples} samples...")
                        
                        sample_loop (model,
                            test_loader,
                            cond_scales=cond_scales,#[7.5], #list of cond scales - each sampled...
                            num_samples=num_samples, #how many samples produced every time tested.....
                          
                                    flag=steps,foldproteins=foldproteins,seed=seed,
                                     strain=strain,save_ind_files=save_ind_files,
                                    )   
                        if val_loader !=None:
                            sample_loop (model,
                                val_loader ,
                                cond_scales=[1],#[7.5], #list of cond scales - each sampled...
                                num_samples=4, #how many samples produced every time tested.....
                               
                                        flag=steps, plot_structure=True, plot_original_data=True,do_forward=False,
             number_batches=4, strain=strain, save_ind_files=save_ind_files,
                                    )
 
                if steps>0:
                    if save_model and steps % print_loss==0: #steps % print_loss == 0:
                        fname=f"{prefix}trainer_save-model-epoch_{e+start_ep}.pt"
                        trainer.save(fname)
                        fname=f"{prefix}statedict_save-model-epoch_{e+start_ep}.pt"
                        torch.save(model.state_dict(), fname)
                        print (f"Model saved: ")
                    
                steps=steps+1
                                         
            print (f"\n\n-------------------\nTime for epoch {e+start_ep}={(time.time()-start)/60}\n-------------------")
```
define, train/load model
```
from HierarchicalDesign import HierarchicalDesignDiffusion, HiearchicalDesignTrainer
```
```
from HierarchicalDesign import HierarchicalDesignDiffusion, HiearchicalDesignTrainer
```
```
loss_list=[]
strain=[]

for i in range (32):
    strain.append (i*0.02 )
    
load_model = True
```
```
prefix='./Diffusion_results/'
if not os.path.exists(prefix):
        os.mkdir (prefix)
```
```
z_reshaped.shape[2]
```
输出64
```
sqer_z=int (z_reshaped.shape[2]**0.5)
sqer_z
```
输出8
```
z_reshaped.shape[1]
```
输出16
```
latent_dim=z_reshaped.shape[2]
max_length=latent_dim

pred_dim=z_reshaped.shape[1]
cond_dim=16
embed_dim_position=16 #cond_dim
model =HierarchicalDesignDiffusion (timesteps=(96), dim=128, pred_dim=pred_dim, 
                     loss_type=0, 
                  padding_idx=0,
                cond_dim = embed_dim_position+cond_dim,
                text_embed_dim =embed_dim_position,#+cond_dim,
                    # input_tokens=input_tokens,
                 embed_dim_position=embed_dim_position,#+cond_dim,
                 max_text_len=32,
                      pos_emb=False,
                     pos_emb_fourier=True,
                     pos_emb_fourier_add=True, #embed_dim_position must be equal to text_embed_dim
             device=device,
    add_z_loss = False, # True,
        loss_z_factor = 1.,
                     VAE=VQVAEmodel,
                                    max_length=max_length,
                )  .to(device)  

train_unet_number=1 #there is only one U-net 

count_parameters (model)
```
输出Total parameters:  154080820  trainable parameters:  154080820
```
if load_model!=True:
    trainer = HiearchicalDesignTrainer(model)
    train_loop (model,
                train_loader,
               test_loader,
                optimizer=None,
                print_every=100,
                epochs= 2400,
                start_ep=0,
            start_step=0,
                train_unet_number=1,
            print_loss =    5*len (train_loader)-1,
            trainer=trainer,
            plot_unscaled=False,#if unscaled data is plotted
            max_batch_size =16,#if trainer....
            save_model=True,
            cond_scales=[  1],#[1, 2.5, 3.5, 5., 7.5, 10., 15., 20.],
            num_samples=16,foldproteins=True,strain=strain,
               )
```
如果有训练好的模型
```
if load_model:
    fname=f"{prefix}statedict_save-model-epoch_4000_FINAL.pt"
    model.load_state_dict(torch.load(fname))
```
没有就直接
```
sample_loop    (model,
                test_loader ,
                cond_scales=[1],#list of cond scales - each sampled...
                num_samples=4, #how many samples produced every time tested.....
                flag=1000, plot_structure=True, plot_original_data=True,do_forward=False,
                strain=strain,
                number_batches=1, #How many batches are sampled
             )
```
```
sample_loop (model,
                            test_loader ,
                            cond_scales=[1],# 
                            num_samples=4, #how many samples produced every time tested.....
                                    flag=8000, plot_structure=True, plot_original_data=True,do_forward=False,
             strain=strain,number_batches=1,save_ind_files=True,csv_for_data='301_test_2.csv' #stored results in CSV file
                                    )
```
        
