%% Edge detection quantitative evaluation : 

% Hi sorry I am not pro in matlab :) if you improved this version, please
% share with me :), I think it needs more efficient coding.
%To start for the first time Run, then: 
% The results are goign to be saved into results dir
% for the simplicity I have termed DexiNed res as dxn
% Pleae make sure that the following dirs have the respective data:
% gt, gt_imgs and edges_pred, if the gt are images, png, copy the same data
% from gt in gt_imgs


clear;
addpath(genpath('F:\data\deep'));
dataset_name = {'BIPED','BSDS','NYUD','BSDS300','PASCAL',...
    'MULTICUE','CLASSIC','CID','DCD'};
models = {'bcn','hed','rcf','can','sob','sed','ced','dxn'}; %1,4,3,5
% show version
model_version = 'f v2 RN 500 epochs'; % 

% ********* Setting and dataset managing *******

% please select the dataset
dataset =dataset_name{1};  % 1=BIPED, 2=BSDS, 3=NYUD,.. 9
% please select the model 
model_name=models{8}; %::: 8
% if your data is already NMS applyed  TRUE
is_pp =false; % with nms
% if you want to use NMS set TRUE
use_nms = true; % just for xdx in false
% If directories already axist TRUE
dirs_ok =true;
dir_name = strcat(lower(dataset),'_',lower(model_name));
is_bw=false; % for the nms the edges have to be in white and non-edges in black
results={};
info_task = ['==> Dataset: ' dataset ' Model: ' model_name...
    ' Use NMS: ' num2str(use_nms) ' is already NMS processed: '...
    num2str(is_pp) ' <=='];
disp(info_task);

base_dir = fullfile('results',dir_name);
%base_dir = 'C:\Users\NING MEI\Desktop\DDW\ev_biped\';
% if ~exist(base_dir,'dir')
dir_ed= fullfile(base_dir,'edges_pred');
dir_values = fullfile(base_dir,'values');
dir_pp = fullfile(base_dir,'edges_nms');
dir_gt = fullfile(base_dir,'gt');
dir_gt_imgs = fullfile(base_dir,'gt_imgs');
dir_imgs = fullfile(base_dir,'imgs');
list_ed = dir(fullfile(dir_ed,'*.png'));
if strcmp(dataset,'BIPED')||strcmp(dataset,'cid')
    list_gt = dir(fullfile(dir_gt,'*.png'));
    list_gt_imgs = dir(fullfile(dir_gt,'*.png'));
    dir_gt_imgs= dir_gt;
    disp(['Base_dir: ' base_dir]);
else
    list_gt = dir(fullfile(dir_gt,'*.mat'));
    list_gt_imgs = dir(fullfile(dir_gt_imgs,'*.png'));
end
list_values = dir(fullfile(dir_values,'*.txt'));
n_res = length(list_ed);

dir_sum_values = fullfile(base_dir,'values-eval');
list_pp = dir(fullfile(dir_pp,'*.png'));
% make dirs
[status, msg, msgID] = mkdir(dir_ed);
disp(msg);
[status, msg, msgID] = mkdir(dir_gt_imgs);
disp(msg);
[status, msg, msgID] = mkdir(dir_values);
disp(msg);
[status, msg, msgID] = mkdir(dir_pp);
disp(msg);
[status, msg, msgID] = mkdir(dir_gt);
disp(msg);
[status, msg, msgID] = mkdir(dir_sum_values);
disp(msg);
[status, msg, msgID] = mkdir(dir_imgs);
disp(msg);
disp(['Base_dir: ' base_dir]);
    

if length(list_gt_imgs)==0 
    disp('Make sure you have data in GT dirs, then *Run*');
    return;
    
end
if length(list_ed)==0
    disp('Make sure you have data in GT/ edge-maps dirs, then *Run*');
    return;
end

if ~is_pp || length(list_pp)==0
    disp('==> Please wait NMS is been applyed to your predictions...');
    if isequal(dataset, 'CLASSIC')
        % for the rest of the dataset
        for i=1:n_res
            
            tmp_edge = imread(fullfile(dir_ed,list_ed(i).name));
            tmp_gt = imread(fullfile(dir_gt,list_gt(i).name));
    %         tmp_edge = imcomplement(rgb2gray(tmp_edge)); % 
            img_size = size(tmp_gt);
            if length(size(tmp_edge))>2
                tmp_edge = rgb2gray(tmp_edge);
            end
            if ~is_bw
                tmp_edge = 1-single(tmp_edge)/255; % image incomplement
            else
                tmp_edge = single(tmp_edge)/255; 
            end
            
            if ~(size(tmp_edge,1)==img_size(1) &&size(tmp_edge,2)==img_size(2))
                tmp_edge= imresize(tmp_edge,img_size(1:2));
            end
            tmp= tmp_edge;
            edg=tmp_edge;
            if ~(sum(img_size(1:2))==sum(size(tmp_edge)))
                error('size of files do not fulfill the rule')
            end

            [Ox, Oy] = gradient2(convTri(tmp_edge,4));
            [Oxx, ~] = gradient2(Ox); [Oxy,Oyy] = gradient2(Oy);

            O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
            tmp_edge = edgesNmsMex(tmp_edge,O,2,5,1.01,8); % the original is 4 

%             %as threshold and 1 after in pytorch result: edgesNmsMex(E,O,1,5,1.01,4);
            tmp_name = list_gt_imgs(i).name(1:end-3);
            tmp_name = strcat(tmp_name,'png');
            edg=imcomplement(edg);
            imwrite(tmp_edge, fullfile(dir_pp,tmp_name));
%             imwrite(edg, fullfile(dir_ed,list_gt(i).name));
        end
% ******* Dataset with Gts ******        
    else
                % for the rest of the dataset
        for i=1:n_res
            
            tmp_edge = imread(fullfile(dir_ed,list_ed(i).name));
            tmp_gt = imread(fullfile(dir_gt_imgs,list_gt_imgs(i).name));
    %         tmp_edge = imcomplement(rgb2gray(tmp_edge)); % 
            img_size = size(tmp_gt);
            if length(size(tmp_edge))>2
                tmp_edge = rgb2gray(tmp_edge);
            end
            if ~is_bw
                tmp_edge = 1-single(tmp_edge)/255; % image incomplement
            else
                tmp_edge = single(tmp_edge)/255; 
            end
            
            if ~(size(tmp_edge,1)==img_size(1) &&size(tmp_edge,2)==img_size(2))
                tmp_edge= imresize(tmp_edge,img_size(1:2));
            end
            tmp= tmp_edge;
            edg=tmp_edge;
            if ~(sum(img_size(1:2))==sum(size(tmp_edge)))
                error('size of files do not fulfill the rule')
            end

            [Ox, Oy] = gradient2(convTri(tmp_edge,4));
            [Oxx, ~] = gradient2(Ox); [Oxy,Oyy] = gradient2(Oy);

            O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
            if isequal(dataset, 'NYUD') && ~isequal(model_name, 'dxn')
                tmp_edge = edgesNmsMex(tmp_edge,O,4,5,1.01,8); % the original is 4 
            elseif isequal(model_name, 'dxn')
                if use_nms
                    tmp_edge = edgesNmsMex(tmp_edge,O,1,5,1.03,8); % O,1,5,1.01,8 bedf=4b8
                end
            else
                if use_nms
                    tmp_edge = edgesNmsMex(tmp_edge,O,1,5,1.01,4); % the original is 4 
            
                end
            end
            edg=imcomplement(edg);
            imwrite(tmp_edge, fullfile(dir_pp,list_gt_imgs(i).name));
            imwrite(edg, fullfile(dir_ed,list_gt_imgs(i).name));
        end
    end
    disp('NMS process finished...');
end
% RCF:(tmp_edge,O,2,5,1.01,8) HED:(tmp_edge,O,1,5,1.01,4)
% DXN:(tmp_edge,O,1,5,1.03,8)later (tmp_edge,O,3,5,1.04,8)
% instead of edgesNmsMex(tmp_edge,O,1,5,1.01,4)
%% Evaluate model ****ODS OIS AP***
% set values and data
% if you want to evaluate please set edgesEvalImg_x()function data_name line 48
disp('...5 seg  to start evaluation');
pause(5);
if isequal(dataset, 'CLASSIC')
    return
end

model.opts.rgbd = 0;
model.opts.nms =0;
% [vis-hed, msi-hed, bsds-hed,]
model.opts.modelFnm = dir_name;  % [vis-hed, msi-hed, bsds-hed,]
model.opts.bsdsDir = 'BSR/BSDS500/data/';
model.opts.imgDir = 'imgs';
model.opts.gtDir = 'gt';
model.opts.predDir= 'edges_pred';
model.opts.mnsDir = 'edges_nms';
model.opts.mainDir = 'results';  %

disp(['Evaluating in: ' model.opts.modelFnm, ' Version: ', model_version]);
% evaluate edges
[results{1:9}] = edgesEval_x(model,'show',1,'name','');


disp('**** Baseline result *******');
res_inf = ['ODS=' num2str(results{1}) ' OIS=' num2str(results{5})...
    ' AP=' num2str(results{8}) ' R50=' num2str(results{9})];
disp(res_inf);
disp('**** Current result *******');
%% ************* plot results *************
pause;
dataset = {'BIPED','BSDS','NYUD','PASCAL','MULTICUE','CLASSIC'};
models = {'hed','rcf','xcp','xdx'}; %1,4,3,5
n_data = 1;  % 1=BIPED, 2=BSDS, 3=NYUD
n_model=4;
is_msi=false;
dir_name = strcat(lower(dataset{n_data}),'_',models{n_model});
base_dir=fullfile('results','z_plots');
dataset =lower(dataset_name{n_data});
base_dir = fullfile(base_dir,'MDBD');
list_dirs =dir(base_dir);
alg_names ={};
k=1;
for i=3:length(list_dirs)
    alg_names{k}=list_dirs(i).name;
    k=k+1;
end

figure;

edgesEvalPlot_x(base_dir,alg_names);   % resDir,name)