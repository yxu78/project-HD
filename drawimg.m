function batch_render_lammps_png()
% 批量渲染 LAMMPS data（atom_style atomic: id type x y z）为 1200x1200 PNG @ 200 dpi
% 统一使用 header 的 xlo/xhi & ylo/yhi 作为取景范围，便于对比

%% ===== 用户参数 =====
inDir   = '/data/home/hanshen/xyb/HD_test/out_all';       % .data 文件所在目录
outDir  = fullfile(inDir, 'png_out');  % 输出目录
pattern = '*.data';                    % 文件匹配
dotSize = 4;                           % 点大小（像素）
byTypeColoring = false;                % false=单色（黑点）；true=按 type 上色
bgColor = 'w';                         % 'w' 白底；如需 32bit RGBA 透明底：'none'

% 输出规格：1200x1200 @ 200 dpi -> 6x6 英寸
dpi = 200;
figSizeInch = [6 6];

%% ===== 准备 =====
if ~exist(outDir,'dir'); mkdir(outDir); end
files = dir(fullfile(inDir, pattern));
if isempty(files)
    error('目录中未找到 %s ：%s', pattern, inDir);
end
fprintf('发现 %d 个 data 文件，输出目录：%s\n', numel(files), outDir);

%% ===== 主循环（可改为 parfor） =====
t0 = tic;
for k = 1:numel(files)
    fpath = fullfile(files(k).folder, files(k).name);

    % 读取：盒子范围 + Atoms 段
    header = read_header_bounds(fpath);            % 结构体：xlo,xhi,ylo,yhi,zlo,zhi
    [id, typ, xyz] = read_atoms_atomic(fpath);     % id,type,x,y,z

    % === 绘图（x-y 投影） ===
    fig = figure('Visible','off','Units','inches',...
        'Position',[1 1 figSizeInch], 'Color', bgColor);
    ax  = axes('Parent',fig); hold(ax,'on');

    if byTypeColoring && numel(unique(typ))>1
        ut = unique(typ);
        cmap = lines(max(numel(ut),7));
        for ii = 1:numel(ut)
            sel = (typ==ut(ii));
            plot(ax, xyz(sel,1), xyz(sel,2), '.', ...
                 'MarkerSize', dotSize, ...
                 'Color', cmap(mod(ii-1,size(cmap,1))+1,:));
        end
    else
        plot(ax, xyz(:,1), xyz(:,2), 'k.', 'MarkerSize', dotSize);
    end

    axis(ax,'equal'); axis(ax,'off');
    % 统一按 header 取景（略加2%边距防止贴边）
    xpad = 0.02*(header.xhi - header.xlo);
    ypad = 0.02*(header.yhi - header.ylo);
    xlim(ax,[header.xlo - xpad, header.xhi + xpad]);
    ylim(ax,[header.ylo - ypad, header.yhi + ypad]);

    % 导出：1200x1200 @ 200 dpi
    [~, base, ~] = fileparts(files(k).name);
    outPng = fullfile(outDir, [base '.png']);
    % 若需要严格 32-bit RGBA，可把 BackgroundColor 改为 'none'
    exportgraphics(fig, outPng, 'Resolution', dpi, 'BackgroundColor', bgColor);
    close(fig);

    fprintf('[%4d/%4d] %-20s -> %s (N=%d)\n', ...
        k, numel(files), files(k).name, outPng, size(xyz,1));
end
fprintf('完成：%d 张，耗时 %.1f s\n', numel(files), toc(t0));
end

%% ====== 辅助：读取 header 中的盒子范围 ======
function H = read_header_bounds(fname)
fid = fopen(fname,'r'); assert(fid>0, '无法打开文件：%s', fname);
c = onCleanup(@() fclose(fid));
H = struct('xlo',[],'xhi',[],'ylo',[],'yhi',[],'zlo',[],'zhi',[]);
while true
    t = fgetl(fid);
    if ~ischar(t), break; end
    if contains(t,'xlo') && contains(t,'xhi')
        v = sscanf(t, '%f %f'); H.xlo = v(1); H.xhi = v(2);
    elseif contains(t,'ylo') && contains(t,'yhi')
        v = sscanf(t, '%f %f'); H.ylo = v(1); H.yhi = v(2);
    elseif contains(t,'zlo') && contains(t,'zhi')
        v = sscanf(t, '%f %f'); H.zlo = v(1); H.zhi = v(2);
    elseif strncmpi(strtrim(t),'Atoms',5)
        break; % 到 Atoms 段就停（header 读够了）
    end
end

assert(~isempty(H.xlo) && ~isempty(H.ylo), '未解析到 x/y 边界：%s', fname);
end

%% ====== 辅助：读取 Atoms 段（atomic: id type x y z） ======
function [id, typ, xyz] = read_atoms_atomic(fname)
fid = fopen(fname,'r'); assert(fid>0, '无法打开文件：%s', fname);
c = onCleanup(@() fclose(fid));

% 找到 "Atoms" 段
found = false;
while true
    t = fgetl(fid); if ~ischar(t), break; end
    if startsWith(strtrim(t), 'Atoms', 'IgnoreCase', true)
        found = true; break;
    end
end
assert(found, '未找到 "Atoms" 段：%s', fname);

% 跳过空行
pos = ftell(fid); t = fgetl(fid);
while ischar(t) && all(isspace(t)); pos = ftell(fid); t = fgetl(fid); end
fseek(fid, pos, 'bof');

C = textscan(fid, '%f %f %f %f %f', 'CommentStyle','#', 'CollectOutput', true);
M = C{1};
assert(size(M,2) >= 5, 'Atoms 段列数不足（期望 id type x y z）：%s', fname);

id  = M(:,1);
typ = M(:,2);
xyz = M(:,3:5);
end
