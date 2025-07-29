在xionglab3上安装Matlab  
先把安装包从本地上传到xionglab3的/data8/目录
```
cd E:\documents\GEARS
scp matlab_R2025a_Linux.zip yxu78@xionglab3.mae.ncsu.edu:/data8/
```
在服务器上解压并启动安装程序
```
ssh -X yxu78@xionglab3.mae.ncsu.edu
```
```
cd /data8
```
```
unzip matlab_R2025a_Linux.zip -d ./matlab_R2025a_Linux
```
```
cd ./matlab_R2025a_Linux
```
```
sudo ./install
```
