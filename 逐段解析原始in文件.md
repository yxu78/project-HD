```
##---------------INITIALIZATION-------------------------------

units          metal
dimension 	    3 
boundary       p p f
atom_style 	atomic
newton 		on 
```
初始化。metal单位制。3维MD。x,y方向周期性边界条件。atomic原子样式。牛三优化。
```
##---------------ATOM DEFINITION------------------------------

read_data 	grap-data.data  
```
读取data文件。
```
##---------------FIXING EDGES OF GRAPHENE--------------------------------------

region far_stress block -1.698000 19.302000 -1.000000 152.121000 -10.000000 10.000000 units box 
group  far_stress region far_stress
```
定义远场固定区。6个数值对应xlo, xhi, ylo, yhi, zlo, zhi。  
新建一个原子组，包含所有落在上述区域内的原子。
```
##---------------FORCE FIELDS---------------------------------

pair_style 	airebo 3.0
pair_coeff     * * CH.airebo C
```
定义原子间相互作用的力场（势函数）
```
##---------------SETTINGS-------------------------------------

timestep 	0.0005
thermo 	100
```
积分时间步长设置为0.0005ps。日志每100步打印一次。
```
##---------------COMPUTES-------------------------------------

compute 	1 all stress/atom NULL
compute 	11 far_stress stress/atom NULL
compute    2 all reduce sum c_1[1] c_1[2] c_1[3] c_11[1] c_11[2] c_11[3] 
```
整体原子应力张量。ID=1。作用对象=all。  
远场固定区原子应力张量。ID=11。作用对象=原子组far_stress。  
全局应力和远场应力分量求和。ID=2。作用对象=all。
