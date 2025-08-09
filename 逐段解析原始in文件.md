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
```
##---------------RELAXATION--------------------------------------

thermo_style   custom step temp vol lx ly pe ke c_2[3] 
fix            1 all npt temp 300.0 300.0 0.05 x 0 0 0.5
```
日志打印出的量：时间步编号、温度、体积、模拟和在x,y方向上的长度、势能与动能、compute2计算的全体系应力总和。  
对all施加NPT积分器。恒温300K。压力耦合（仅x方向），让体系沿x方向自由伸缩以达到零压平衡，消除初始应力。
```
##---------------OUTPUTS--------------------------------------
variable   Vol equal 78084.937284
variable   ConvoFac equal 1/1.0e4
variable   sigmaxx equal c_2[1]/v_Vol*v_ConvoFac
variable   sigmayy equal c_2[2]/v_Vol*v_ConvoFac
variable   SigmaxxFar equal c_2[4]/v_Vol*v_ConvoFac/0.138889
variable   SigmayyFar equal c_2[5]/v_Vol*v_ConvoFac/0.138889
thermo         1000
run            50000
```
定义体积、换算系数。  
计算体系应力。c_2[1], c_2[2]除以体积将其转换为应力，再乘以换算系数得到约化到Gpa的数值，结果分别存入sigmaxx, sigmayy。  
计算远场固定区应力。c_2[4], c_2[5]分别对远场固定区σxx, σyy求和。除以0.138889得到该区域平均应力，存入SigmaxxFar, SigmayyFar。  
将打印信息改为每1000步一次。  
进行50000步模拟。
```
##---------------DEFORMATION--------------------------------------
unfix              1
fix                1 all npt temp 300.0 300.0 0.05 x 0 0 0.5
variable           srate equal 1.0e9
variable           srate1 equal "v_srate / 1.0e12"
fix                2 all deform 1 y erate ${srate1} units box remap x
variable           StrainPerTs equal 5.0/1.0e7
variable           strain equal v_StrainPerTs*(step)
thermo_style       custom step temp lx ly pe v_strain v_sigmaxx v_sigmayy v_SigmaxxFar v_SigmayyFar 
reset_timestep     0
fix                write all print 1000 " ${strain} ${sigmaxx} ${sigmayy} ${SigmaxxFar} ${SigmayyFar}" file stress_strain screen no
dump           1 all custom 10 dump.all.p1 id type x y z c_1[1] c_1[2] c_1[3] c_1[4] c_1[5] c_1[6]
run                40000
```
unfix移除前一阶段的松弛。然后以相同的参数施加NPT，保持300K恒温，允许模拟盒在x方向上以0bar压力自由伸缩。  
定义目标应变率。  
沿y方向施加匀速拉伸，每1步按速率srate1改变模拟盒y方向长度，随盒子更新同步更新原子坐标。  
定义累计应变变量。StrainPerTs是每步对应的应变增量。strain是当前总应变。  
定制热力学输出格式。  
重置步数计数。  
定期将数据写入文件sress_strain。  
每10步输出所有原子编号、类型、坐标及应力张量6分量到dump.all.p1.  
进行40000步模拟。
```
##---------------RELAX--------------------------------------
unfix          2
fix            3 all ave/atom 1 5000 5000 c_1[1] c_1[2] 
variable       strain equal 0.02
run            20000
```
取消fix 2。  
fix 3对每个原子的应力分量做时间平均：每步取样一次，累计5000次后输出一次平均值（在步数5000、10000、15000...时更新）。  
演化20000步，让体系在该应变下充分松弛，同时得到平滑的（时间平均的）原子应力。
```
##---------------DEFORMATION--------------------------------------
unfix          3
fix            2 all deform 1 y erate ${srate1} units box remap x
dump           2 all custom 10 dump.all.p2 id type x y z c_1[1] c_1[2] c_1[3] c_1[4] c_1[5] c_1[6]
variable       strain equal v_StrainPerTs*(step-20000)
```
取消fix 3。  
重新加载fix 2，在y方向拉伸模拟盒。
dump 2每10步输出原子id、type、坐标以及应力张量6分量到dump.all.p2。
