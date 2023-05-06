function varargout = edgesEval_x( model, varargin )
% Run and evaluate structured edge detector on BSDS500.
%
% This function first runs the trained structured edge detector on every
% test or validation image in BSDS then call edgesEvalDir.m to perform the
% actual edge evaluation. edgesEval is specific to the structured edge
% detector and BSDS, edgesEvalDir is general purpose edge evaluation code.
% For example usage of edgesEval see edgesDemo.
%
% USAGE
%  varargout = edgesEval( model, prms )
%
% INPUTS
%  model      - structured edge model trained with edgesTrain
%  prms       - parameters (struct or name/value pairs)
%   .dataType   - ['test'] should be either 'test' or 'val'
%   .name       - [''] name to append to evaluation
%   .opts       - {} list of model opts to overwrite
%   .show       - [0] if true plot results using edgesEvalPlot
%   .pDistr     - [{'type','parfor'}] parameters for fevalDistr
%   .cleanup    - [0] if true delete temporary files
%   .thrs       - [99] number or vector of thresholds for evaluation
%   .maxDist    - [.0075] maximum tolerance for edge match
%
% OUTPUTS
%  varargout  - same outputs as edgesEvalDir
%
% EXAMPLE
%
% See also edgesDemo, edgesDetect, edgesEvalDir, edgesEvalPlot
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014.
% Licensed under the MSR-LA Full Rights License [see license.txt]

% get default parameters
da_name = model.opts.modelFnm(1:4);
if strcmp(da_name,'nyud')
    maxDis = 0.011;
else
    maxDis = .0075;
end
dfs={'dataType','test', 'name','', 'opts',{}, 'show',0, ...
  'pDistr',{{'type','parfor'}}, 'cleanup',0, 'thrs',99, 'maxDist',maxDis};
p=getPrmDflt_x(varargin,dfs,1);
detect_edge =false;
% load model and update model.opts acoording to opts
if( ischar(model) ), model=load(model); model=model.model; end
for i=1:length(p.opts)/2, model.opts.(p.opts{i*2-1})=p.opts{i*2}; end
rgbd=model.opts.rgbd; model.opts.nms=1;

% get list of relevant directories (image, depth, gt, results)
name = model.opts.modelFnm;
imgDir = fullfile(model.opts.mainDir,model.opts.modelFnm,...
    model.opts.imgDir);
depDir = fullfile(model.opts.bsdsDir,'depth',p.dataType);
gtDir  = fullfile(model.opts.mainDir,model.opts.modelFnm,...
    model.opts.gtDir);
modelDir = fullfile(model.opts.mainDir,model.opts.modelFnm);
resDir = fullfile(model.opts.mainDir,model.opts.modelFnm,...
    model.opts.mnsDir);
assert(exist(imgDir,'dir')==7); assert(exist(gtDir,'dir')==7);

% run edgesDetect() on every image in imgDir and store in resDir
% ** prepare the whole of documentation
if( ~exist(fullfile([modelDir '/values'],'eval_bdry.txt'),'file') )
  if(~exist(resDir,'dir')), mkdir(resDir);
      disp('Attention, a dir was created');
  end
  
  if detect_edge % model.opts.detect_edge
      
      ids=dir(imgDir); 
      ids=ids([ids.bytes]>0); 
      ids={ids.name}; 
      n=length(ids);
      ext=ids{1}(end-2:end); for i=1:n, ids{i}=ids{i}(1:end-4); end
      res=cell(1,n); for i=1:n, res{i}=fullfile(resDir,[ids{i} '.png']); end
      do=false(1,n); for i=1:n, do(i)=~exist(res{i},'file'); end
      ids=ids(do); res=res(do); m=length(ids);
      for i=1:m, id=ids{i}; % parfor
        I = imread(fullfile(imgDir,[id '.' ext])); D=[];
        if(rgbd), D=single(imread(fullfile(depDir,[id '.png'])))/1e4; end
        if(rgbd==1), I=D; elseif(rgbd==2), I=cat(3,single(I)/255,D); end
        E=edgesDetect(I,model); imwrite(uint8(E*255),res{i});
    %     savinh edge and png files 
    %     fprintf('It is runing this line, save file %s \n', res{i});
    %     pause(0.5)
      end
  end
end
% *** end preparing data **
% perform actual evaluation using edgesEvalDir
varargout=cell(1,max(1,nargout));
[varargout{1:9}] = edgesEvalDir_x('modDir',modelDir,'resDir',resDir,'gtDir',gtDir,...
  'pDistr',p.pDistr,'cleanup',p.cleanup,'thrs',p.thrs,'maxDist',p.maxDist); %maxDist = 0.0075
if( p.show ), figure(p.show); edgesEvalPlot([modelDir '/values'],name); end  % resDir,name)
% last most important part for result

end
