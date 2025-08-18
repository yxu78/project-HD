%% 批量生成：中心裂纹 + 4 对称移动孔 的 LAMMPS data（统一目录+编号命名）
close all; clear; clc;

%% 常量与控制
CC_bond_length = 1.396;            % Å
hole_dia       = CC_bond_length*6.25;
Npos           = 5;                % 所需生成的datafile数量
corner_choice  = 'TR';             % TR/TL/BR/BL
enable_crack   = true;
crack_width    = 16;               % Å（沿 x）
crack_thick    = CC_bond_length*1.2;

%% 输出目录（统一放到 out_all）
out_root = 'out_all';
if ~exist(out_root,'dir'); mkdir(out_root); end

%% 读入基底
fid = fopen('zz_15nm','r'); 
if fid < 0, error('未找到文件 zz_15nm'); end
junk  = fscanf(fid,'%s ',27);                  % 跳过头部 token
coord = fscanf(fid, '%d %d %f %f %f ', [5, inf])';
fclose(fid);

coord0 = coord;
x_max = max(coord(:,3)); x_min = min(coord(:,3));
y_max = max(coord(:,4)); y_min = min(coord(:,4));
z_max = max(coord(:,5)); z_min = min(coord(:,5));
center_xy = [ (x_min+x_max)/2, (y_min+y_max)/2 ];

% 目标角
switch upper(corner_choice)
    case 'TR', target_xy = [x_max, y_max];
    case 'TL', target_xy = [x_min, y_max];
    case 'BR', target_xy = [x_max, y_min];
    case 'BL', target_xy = [x_min, y_min];
    otherwise, error('corner_choice 只能为 TR/TL/BR/BL');
end

% 从中心到角的插值
t_list  = linspace(0,1,Npos);      % 如需避免端点，可改为 linspace(0.1,0.9,Npos)
delta   = target_xy - center_xy;
dx_list = t_list * delta(1);
dy_list = t_list * delta(2);

% 记录表
manifest = table('Size',[0 10], ...
    'VariableTypes',{'double','double','double','double','double','double','double','double','string','string'}, ...
    'VariableNames',{'idx','dx','dy','c1x','c1y','c2x','c2y','hole_dia','outfile','outpath'});

%% 主循环
for i = 1:Npos
    dx = dx_list(i);  dy = dy_list(i);

    % 四孔中心（以 sheet 中心对称）
    c1 = center_xy + [+dx, +dy];
    c2 = center_xy + [-dx, +dy];
    c3 = center_xy + [+dx, -dy];
    c4 = center_xy + [-dx, -dy];
    centers = [c1; c2; c3; c4];

    % 打孔
    coord_def = apply_four_holes(coord0, centers, hole_dia);

    % 裂纹（居中矩形）
    if enable_crack
        x_mid = center_xy(1); y_mid = center_xy(2);
        crack_dim = [ ...
            x_mid - crack_width/2,  y_mid - crack_thick/2; ...
            x_mid + crack_width/2,  y_mid - crack_thick/2; ...
            x_mid + crack_width/2,  y_mid + crack_thick/2; ...
            x_mid - crack_width/2,  y_mid + crack_thick/2; ...
            x_mid - crack_width/2,  y_mid - crack_thick/2 ];
        coord_def = apply_crack_rect(coord_def, crack_dim);
    end

    % 输出文件：统一目录 + 编号命名
    outfile = sprintf('data%05d.data', i);                 % data00001.data
    write_lammps_data(coord_def, out_root, CC_bond_length, 'outfile', outfile);

    % 记录
    outpath = fullfile(out_root, outfile);
    manifest = [manifest; {i, dx, dy, c1(1), c1(2), c2(1), c2(2), hole_dia, string(outfile), string(outpath)}];

    fprintf('进度 %d/%d -> %s  剩余原子数 %d\n', i, Npos, outpath, size(coord_def,1));
end

writetable(manifest, 'batch_4holes_manifest.csv');
disp('完成：已在 out_all/ 生成编号 data 文件，并输出清单 batch_4holes_manifest.csv。');

%% ======== 局部函数 ========
function coord_out = apply_four_holes(coord_in, centers, hole_dia)
    r2 = (hole_dia/2)^2;
    in_any = false(size(coord_in,1),1);
    for k = 1:size(centers,1)
        cx = centers(k,1); cy = centers(k,2);
        in_any = in_any | ((coord_in(:,3)-cx).^2 + (coord_in(:,4)-cy).^2 < r2);
    end
    coord_out = coord_in(~in_any,:);
end

function coord_out = apply_crack_rect(coord_in, crack_dim)
    x1 = min(crack_dim(:,1)); x2 = max(crack_dim(:,1));
    y1 = min(crack_dim(:,2)); y2 = max(crack_dim(:,2));
    in_rect = (coord_in(:,3) >= x1) & (coord_in(:,3) <= x2) & ...
              (coord_in(:,4) >= y1) & (coord_in(:,4) <= y2);
    coord_out = coord_in(~in_rect,:);
end

function write_lammps_data(coord, out_dir, CC_bond_length, varargin)
    % 可选参数：'outfile'，默认 'grap-data.data'
    p = inputParser;
    addParameter(p, 'outfile', 'grap-data.data', @(s)ischar(s) || isstring(s));
    parse(p, varargin{:});
    outfile = char(p.Results.outfile);

    fdat = fopen(fullfile(out_dir, outfile), 'w');
    if fdat < 0, error('无法写入文件：%s', fullfile(out_dir, outfile)); end

    fprintf(fdat,'Graphene sheet with four moving holes\n\n');
    fprintf(fdat,'%d atoms \n\n', size(coord,1));
    fprintf(fdat,'%d atom types \n\n', 1);
    fprintf(fdat,'#simulation box \n');
    fprintf(fdat,'%f %f xlo xhi\n', min(coord(:,3)) - cos(pi/6)*CC_bond_length/2, ...
                                    max(coord(:,3)) + cos(pi/6)*CC_bond_length/2);
    fprintf(fdat,'%f %f ylo yhi\n', min(coord(:,4)) - CC_bond_length/2, ...
                                    max(coord(:,4)) + CC_bond_length/2);
    fprintf(fdat,'%f %f zlo zhi\n\n', min(coord(:,5)) - 100, max(coord(:,5)) + 100);
    fprintf(fdat,'Masses\n\n');
    fprintf(fdat,'%d %f\n\n', 1, 12.00000);
    fprintf(fdat,'Atoms\n\n');
    for k = 1:size(coord,1)
        fprintf(fdat,'%d %d %f %f %f\n', k, coord(k,2), coord(k,3), coord(k,4), coord(k,5));
    end
    fclose(fdat);
end
