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
[具体操作](./VQ-VAE.md)  

### 1.3 Generative attention-diffusion model
[具体操作](.Generative_attention-diffusion_model.md) 
