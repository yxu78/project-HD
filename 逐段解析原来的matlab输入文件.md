```
CC_bond_length = 1.396;
```
定义碳碳键长
```
[fid] = fopen('zz_15nm');
 
junk=fscanf(fid,'%s ',27); 
[coord,count] = fscanf(fid, '%d %d %f %f %f ',[5,inf]);

fclose(fid);

coord = coord';

original_sheet = coord;
```
打开数据文件zz_15nm，跳过文件头部信息。读取坐标及类型信息（第1行：原子编号；第2行：原子类型；第3-5行：原子的x,y,z坐标）。整理并存储坐标矩阵。
```
%% Dimensions of the sheet
x_max = max(coord(:,3));
x_min = min(coord(:,3));
y_max = max(coord(:,4));
y_min = min(coord(:,4));
z_max = max(coord(:,5));
z_min = min(coord(:,5));

L_x = x_max - x_min;
L_y = y_max - y_min;
```
读取坐标最大值与最小值。计算片层在x,y方向上的长度。
```
%% Define fixed atoms

far_stress_box(1,1) = x_min-1;                        far_stress_box(1,2) = y_min -1; 
far_stress_box(2,1) = x_min-1;                        far_stress_box(2,2) = y_max +1 ; 
far_stress_box(3,1) = x_min+20;                       far_stress_box(3,2) = y_max +1; 
far_stress_box(4,1) = x_min+20;                       far_stress_box(4,2) = y_min -1; 
far_stress_box(5,1) = x_min-1;                        far_stress_box(5,2) = y_min -1; 
```
定义了一个远场固定区far_stress_box，用来标记那些在后续模拟中保持固定的原子。
```
%% Getting Crack & hole dimensions
```
```
y_mid = (y_max - y_min)/2 ;
x_mid = (x_max - x_min)/2 ;
crack_width = 16;
```
片层中心线的位置和裂缝宽度。
```
crack_dim(1,1) = x_mid - crack_width/2-1;                             crack_dim(1,2) = y_mid - CC_bond_length/2; 
crack_dim(2,1) = x_mid + crack_width/2;                               crack_dim(2,2) = y_mid - CC_bond_length/2; 
crack_dim(3,1) = x_mid + crack_width/2;                               crack_dim(3,2) = y_mid + CC_bond_length/0.7; 
crack_dim(4,1) = x_mid - crack_width/2-1;                             crack_dim(4,2) = y_mid + CC_bond_length/0.7; 
crack_dim(5,1) = x_mid - crack_width/2-1;                             crack_dim(5,2) = y_mid - CC_bond_length/2; 
```
定义裂纹多边形顶点crack_dim。  
生成一个闭合四边形表示裂纹区域。  
水平位置以中心线为基准，左右各偏移crack_width/2，在左右边界外各保留1单位余量。  
垂直方向上下分别偏移±与C-C键长相关的量，控制裂纹高度。
```
crack_tip_left_x = 83.06;
crack_tip_right_x = 64.91;
crack_tip_y_top = 76.17;
crack_tip_y_bot = 76.17;
```
手动指定裂纹尖端的坐标。
```
hexagonal_width = 2*cos(pi/6)*CC_bond_length;
hexagonal_height = 2*CC_bond_length;
hole_dia = CC_bond_length* 6.25;
```
定义正六边形宽高。定义孔径。
```
%% Relative positions of a hole
```
计算4个孔的中心位置。
```
dx = 1.396*(1.5 + cos(pi/3)) + (1.396*(2 + 2*cos(pi/3)))*1; % should change 0,1,2,3,4,...
dy = (1.396*cos(pi/6))*10;  % should change 6,8,10.... [dy can also be zero]
```
计算孔中心的相对偏移量。1.396为C-C键长。
```
hole_1_center_r_x = crack_tip_left_x + dx ;
hole_1_center_r_y = crack_tip_y_top  + dy;

hole_2_center_r_x = crack_tip_right_x -  dx; 
hole_2_center_r_y = crack_tip_y_top  + dy;

hole_3_center_r_x = crack_tip_left_x + dx ; 
hole_3_center_r_y = crack_tip_y_bot  - dy;

hole_4_center_r_x = crack_tip_right_x - dx;
hole_4_center_r_y = crack_tip_y_bot  - dy;
```
四个象限孔的位置。  
从裂纹尖端的基础上偏移。
```
%% Positioning the crack & holes
```
移除裂纹区和孔洞内的原子。
```
for i=size(coord(:,1)):-1:1
    
    if (coord(i,3) > crack_dim(1,1) && coord(i,3)<crack_dim(2,1)) && (coord(i,4) > crack_dim(1,2) && coord(i,4)<crack_dim(3,2)) 
    coord(i,:) = [];
```
反向遍历coord中的每一个原子（行）。当原子在crack_dim四边形内时，便将其删除。
```

    elseif sqrt((coord(i,3) - hole_1_center_r_x)^2 + (coord(i,4) - hole_1_center_r_y)^2) < hole_dia/2 %% First hole
    coord(i,:) = [];
    
    elseif sqrt((coord(i,3) - hole_2_center_r_x)^2 + (coord(i,4) - hole_2_center_r_y)^2) < hole_dia/2 %% Second hole
    coord(i,:) = [];
    
    elseif sqrt((coord(i,3) - hole_3_center_r_x)^2 + (coord(i,4) - hole_3_center_r_y)^2) < hole_dia/2 %% First hole
    coord(i,:) = [];
    
    elseif sqrt((coord(i,3) - hole_4_center_r_x)^2 + (coord(i,4) - hole_4_center_r_y)^2) < hole_dia/2 %% Second hole
    coord(i,:) = [];
    
    end
end
```
四个孔洞。
```
figure
plot(coord(:,3),coord(:,4),'ok','MarkerSize',1) % change this in an appropriate way for zig-zag and arm chair
axis equal
axis off
```
绘制剩余原子。  
```
% hold on
% plot(original_sheet(:,3), original_sheet(:,4),'or','MarkerSize',2)
hold on
plot(far_stress_box(:,1), far_stress_box(:,2),'-r')
```
保持当前图像，叠加绘制远场固定区边界。  
'-r'表示用红色实线绘制。
```
%% Printing
paperWidth = 3.15; 
paperHeight = paperWidth;
set(gcf, 'paperunits', 'inches');
set(gcf, 'papersize', [paperWidth paperHeight]);
set(gcf, 'PaperPosition', [0    0   paperWidth paperHeight]);
print(gcf, '-dpng', 'graphene-sheet'); % Colour
```
将当前图形导出为一个正方形的png图像文件。
'-dpng'指定输出格式为png（默认真彩色），最终输出graphene-sheet.png。
```
%% DATA.FILE_initial part
```
写grap-data.data。
```
fid=fopen('grap-data.data','w');
```
以写模式新建名为grap-data.data的文件。
```
fprintf(fid,'Graphene sheet in zig-zag direction\n');
fprintf(fid,'\n');
fprintf(fid,'%d atoms \n',size(coord,1));
fprintf(fid,'\n');
fprintf(fid,'%d atom types \n',1);
fprintf(fid,'\n');
```
写入文件头描述，原子总数与类型。
```
fprintf(fid,'#simulation box \n');
fprintf(fid,'%f %f xlo xhi\n',min(coord(:,3)) - cos(pi/6)*CC_bond_length/2, max(coord(:,3)) + cos(pi/6)*CC_bond_length/2); 
fprintf(fid,'%f %f ylo yhi\n',min(coord(:,4)) - CC_bond_length/2, max(coord(:,4)) + CC_bond_length/2);
fprintf(fid,'%f %f zlo zhi\n',min(coord(:,5)) - 100, max(coord(:,5)) + 100);
fprintf(fid,'\n');
```
定义模拟盒边界。在x,y,z方向分别设置最小值和最大值。
```
fprintf(fid,'Masses\n');
fprintf(fid,'\n');
fprintf(fid,'%d %f \n',1, 12.00000);
fprintf(fid,'\n');
fprintf(fid,'Atoms\n');
fprintf(fid,'\n');
```
写入质量。之后逐行写入每个原子的编号、类型和坐标。
```
%% define atoms - Interchange x,y coordinates to get a zigzag sheet.

for i=1:size(coord,1)

       fprintf(fid,'%d %d %f %f %f \n',i,coord(i,2),coord(i,3) + rand/100,  coord(i,4) + rand/100,  coord(i,5) + rand/100);

end

fclose(fid);

fid=fopen('grap-in.in','w');

fprintf(fid,'##---------------INITIALIZATION-------------------------------\n');
fprintf(fid,'\n');
fprintf(fid,'units          metal\n');
fprintf(fid,'dimension 	    3 \n');

fprintf(fid,'boundary       p p f\n');

fprintf(fid,'atom_style 	atomic\n');
fprintf(fid,'newton 		on \n');
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------ATOM DEFINITION------------------------------\n');
fprintf(fid,'\n');
fprintf(fid,'read_data 	grap-data.data  \n');
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------FIXING EDGES OF GRAPHENE--------------------------------------\n');
fprintf(fid,'\n');
fprintf(fid,'region far_stress block %f %f %f %f %f %f units box \n', x_min-1, x_min+20, y_min-1, y_max+1, z_min-10, z_max+10); %xlo xhi ylo yhi zlo zhi
fprintf(fid,'group  far_stress region far_stress\n');
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------FORCE FIELDS---------------------------------\n');
fprintf(fid,'\n');
fprintf(fid,'pair_style 	airebo 3.0\n');
fprintf(fid,'pair_coeff     * * CH.airebo C\n');
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------SETTINGS-------------------------------------\n');
fprintf(fid,'\n');
fprintf(fid,'timestep 	0.0005\n');
fprintf(fid,'thermo 	100\n'); 
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------COMPUTES-------------------------------------\n');
fprintf(fid,'\n');
fprintf(fid,'compute 	1 all stress/atom NULL\n'); %Define a computation that computes the per-atom potential energy for each atom in a group. 
fprintf(fid,'compute 	11 far_stress stress/atom NULL\n');
fprintf(fid,'compute    2 all reduce sum c_1[1] c_1[2] c_1[3] c_11[1] c_11[2] c_11[3] \n'); %The sum option adds the values in the vector into a global total. http://lammps.sandia.gov/doc/compute_reduce.html
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------RELAXATION--------------------------------------\n');
fprintf(fid,'\n');
fprintf(fid,'thermo_style   custom step temp vol lx ly pe ke c_2[3] \n');
fprintf(fid,'fix            1 all npt temp 300.0 300.0 0.05 x 0 0 0.5 \n');
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------OUTPUTS--------------------------------------\n');

fprintf(fid,'variable   Vol equal %f\n', (L_x + cos(pi/6)*CC_bond_length)*(L_y + CC_bond_length)*3.4);
fprintf(fid,'variable   ConvoFac equal 1/1.0e4\n'); % Lammps gives in stress in bar;  multiply by (10^5*10^-9) to convert it to GPa
fprintf(fid,'variable   sigmaxx equal c_2[1]/v_Vol*v_ConvoFac\n'); %sigma-xx ; LAMMPS gives stress*vol
fprintf(fid,'variable   sigmayy equal c_2[2]/v_Vol*v_ConvoFac\n'); %sigma-xx ; LAMMPS gives stress*vol

fprintf(fid,'variable   SigmaxxFar equal c_2[4]/v_Vol*v_ConvoFac/%f\n',1260/9072); %sigma-xx ; LAMMPS gives stress*vol
fprintf(fid,'variable   SigmayyFar equal c_2[5]/v_Vol*v_ConvoFac/%f\n',1260/9072); %sigma-xx ; LAMMPS gives stress*vol

fprintf(fid,'thermo         1000\n');
fprintf(fid,'run            50000\n');
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------DEFORMATION--------------------------------------\n');
fprintf(fid,'unfix              1\n');
%fprintf(fid,'undump            1\n');
fprintf(fid,'fix                1 all npt temp 300.0 300.0 0.05 x 0 0 0.5\n');
fprintf(fid,'variable           srate equal 1.0e9\n');
fprintf(fid,'variable           srate1 equal "v_srate / 1.0e12"\n');
fprintf(fid,'fix                2 all deform 1 y erate ${srate1} units box remap x\n');
fprintf(fid,'variable           StrainPerTs equal 5.0/1.0e7\n'); %strain per time step [strain rate * size of time step] it is 5.0/1.0e7 for TS of 0.0005 and SR of 10^9
fprintf(fid,'variable           strain equal v_StrainPerTs*(step)\n'); %strain per time step
fprintf(fid,'thermo_style       custom step temp lx ly pe v_strain v_sigmaxx v_sigmayy v_SigmaxxFar v_SigmayyFar \n');
fprintf(fid,'dump               2 all atom 5000 grap-tensile.lammpstrj\n');
fprintf(fid,'reset_timestep     0\n');
fprintf(fid,'fix                write all print 1000 " ${strain} ${sigmaxx} ${sigmayy} ${SigmaxxFar} ${SigmayyFar}" file stress_strain screen no\n');
fprintf(fid,'run                40000\n'); % 20000 steps will give an strain of 1%
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------RELAX--------------------------------------\n');

fprintf(fid,'unfix          2\n');
fprintf(fid,'fix            3 all ave/atom 1 5000 5000 c_1[1] c_1[2] \n');%Use one or more per-atom vectors as inputs every few timesteps, and average them atom by atom over longer timescales. http://lammps.sandia.gov/doc/fix_ave_atom.html
fprintf(fid,'dump           3 all custom 5000 dump.stress.* id type x y z f_3[1] f_3[2]\n');
fprintf(fid,'variable       strain equal 0.02\n'); %strain per time step
fprintf(fid,'run            20000\n'); % relaxing for 5 ps
fprintf(fid,'\n');
fprintf(fid,'\n');

fprintf(fid,'##---------------DEFORMATION--------------------------------------\n');

fprintf(fid,'unfix          3\n');
fprintf(fid,'undump         3\n');
fprintf(fid,'fix            2 all deform 1 y erate ${srate1} units box remap x\n');
fprintf(fid,'variable       strain equal v_StrainPerTs*(step-20000)\n'); %strain at each time step
fprintf(fid,'run            100000\n');  % final strain is 5%

fclose(fid);
```
详情见in文件的解释文档。
