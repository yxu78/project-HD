参考
```
https://oitncsu-my.sharepoint.com/:f:/g/personal/ychunxu_ncsu_edu/Em0mHaCBRo5CscXa6SH4gqUBY2Ybfubm706RDzJzjAR46A?e=S6KrHs
```
从官网下载tarball到本地，然后通过scp命令把tarball传到服务器。  
然后解压
```
tar -xzvf lammps*.tar.gz
```
进入src目录
```
make yes-manybody
```
```
make mpi
```
然后这里应该会有一个报错
```
mpicxx: No such file or directory
```
看一下
```
which mpicxx
```
然后应该会显示no mpicxx in (...)  
把mpi加入系统PATH
```
module load mpi
```
可以通过
```
echo $PATH
```
来查看系统PATH  
然后重新
```
make mpi
```
make mpi version of lammps  
结束后可以通过
```
ls lmp_mpi
```
确认目录里有lmp_mpi  
lmp_mpi的路径为
```
/data8/lammps/lammps-22Jul2025/src/lmp_mpi
```
