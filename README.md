# project-HD
to prove the applicability of the open source code in https://github.com/lamm-mit/HierarchicalDesign/tree/main
## 0.准备工作
从本地把训练集拷贝到集群
```
scp -r E:/documents/GEARS/file yxu78@xionglab3.mae.ncsu.edu:/data8/test2
```
在file内开一个ipynb，修改表格
```
import pandas as pd

csv_path = r'test.csv'   # 注意 r'' 原始字符串
out_path = r'test_fixed.csv'
col = 'microstructure'

df = pd.read_csv(csv_path)
df[col] = df[col].str.replace(r'^\.\/gene_output_thick\/', '', regex=True)
df.to_csv(out_path, index=False)
print('Done:', out_path)
```
## 1.跑通原代码
使用链接提供的数据集，训练AI tool，并利用训练后的AI tool进行预测。具体流程如下  
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
ssh -L 8888:localhost:8888 yxu78@xionglab3.mae.ncsu.edu #其中8888是上一个窗口给出的port,以具体情况为准
```
用本地浏览器打开第一个窗口给出的链接进入jupyter lab，进入./test2/file文件夹，开一个名为test2.ipynb的kernel开始写代码块  
开启日志监听
```
tmux new -s jl
```
```
mkdir -p ~/logs
jupyter lab --no-browser --ServerApp.log_level=DEBUG 2>&1 | tee -a ~/logs/jupyter_$(date +%F).log
```
若掉线要检查日志
```
tmux attach -t jl        # 回到会话
# 或者单独看日志文件
tail -f ~/logs/jupyter_$(date +%F).log
```
设置数据目录路径和csv文件路径(后续工作视情况而定)
```
data_dir = '/data8/test2/file/'
csvfile = '/data8/test2/file/test.csv'
```
### 1.2 VQ-VAE Model
[具体操作](./VQ-VAE.md)  

### 1.3 Generative attention-diffusion model
[具体操作](./Generative_attention-diffusion_model.md) 

## 2.在xlab3上测试自动化
批量生成data文件的MATLAB code  
一键提交的.sh指令集
