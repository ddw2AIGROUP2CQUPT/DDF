function [thrs,cntR,sumR,cntP,sumP,V] = edgesEvalImg_x( E, G, varargin )
% Calculate edge precision/recall results for single edge image.
%
% Enhanced replacement for evaluation_bdry_image() from BSDS500 code:
%  http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/
% Uses same format and is fully compatible with evaluation_bdry_image.
% Given default prms results are *identical* to evaluation_bdry_image.
%
% In addition to performing the evaluation, this function can optionally
% create a visualization of the matches and errors for a given edge result.
% The visualization of edge matches V has the following color coding:
%  green=true positive, blue=false positive, red=false negative
% If multiple ground truth labels are given the false negatives have
% varying strength (and true positives can match *any* ground truth).
%
% This function calls the mex file correspondPixels. Pre-compiled binaries
% for some systems are provided in /private, source for correspondPixels is
% available as part of the BSDS500 dataset (see link above). Note:
% correspondPixels is computationally expensive and very slow in practice.
%
% USAGE
%  [thrs,cntR,sumR,cntP,sumP,V] = edgesEvalImg( E, G, [prms] )
%
% INPUTS
%  E          - [h x w] edge probability map (may be a filename)
%  G          - file containing a cell of ground truth boundaries
%  prms       - parameters (struct or name/value pairs)
%   .out        - [''] optional output file for writing results
%   .thrs       - [99] number or vector of thresholds for evaluation
%   .maxDist    - [.0075] maximum tolerance for edge match
%   .thin       - [1] if true thin boundary maps
%
% OUTPUTS
%  thrs       - [Kx1] vector of threshold values
%  cntR,sumR  - [Kx1] ratios give recall per threshold
%  cntP,sumP  - [Kx1] ratios give precision per threshold
%  V          - [hxwx3xK] visualization of edge matches
%
% EXAMPLE
%
% See also edgesEvalDir
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014.
% Licensed under the MSR-LA Full Rights License [see license.txt]

% get additional parameters
data_name=G(9:9+4);
dfs={ 'out','', 'thrs',99, 'maxDist',varargin{1,6}, 'thin',1 };
[out,thrs,maxDist,thin] = getPrmDflt(varargin,dfs,1);
if(any(mod(thrs,1)>0)), K=length(thrs); thrs=thrs(:); else
  K=thrs; thrs=linspace(1/(K+1),1-1/(K+1),K)'; end

% load edges (E) and ground truth (G)*********
if(all(ischar(E))), E=double(imread(E))/255; end
[filepath,name,ext] = fileparts(G);
if strcmp(ext,'.png')
    % for GT in PNG image
    tmp=imread(G);
    if length(size(G))>2
        tmp = rgb2gray(tmp); %tmp = imresize(tmp,[480,480]);
    end
    
    tmp(tmp>0)=1;
    G = {double(tmp)}; n = length(G);
    
else
    
    G=load(G); G=G.groundTruth; n=length(G);
    for g=1:n
        G{g}=double(G{g}.Boundaries); end
end
% ******** super important up
% evaluate edge result at each threshold
Z=zeros(K,1); cntR=Z; sumR=Z; cntP=Z; sumP=Z;
if(nargout>=6), V=zeros([size(E) 3 K]); end
for k = 1:K
  % threshhold and thin E
  E1 = double(E>=max(eps,thrs(k))); % so important
  if(thin), E1=double(bwmorph(E1,'thin',inf)); end
  % compare to each ground truth in turn and accumualte
  Z=zeros(size(E)); matchE=Z; matchG=Z; allG=Z;
%   if isequal(data_name, 'vis_hed') || isequal(data_name, 'msi_hed') %byMe
  for g = 1:n
    [matchE1,matchG1] = correspondPixels(E1,G{g},maxDist);
    matchE = matchE | matchE1>0;
    matchG = matchG + double(matchG1>0);
    allG = allG + G{g};
  end
  % compute recall (summed over each gt image)
  cntR(k) = sum(matchG(:)); sumR(k) = sum(allG(:));
  % compute precision (edges can match any gt image)
  cntP(k) = nnz(matchE); sumP(k) = nnz(E1);
  % optinally create visualization of matches
  if(nargout<6), continue; end; cs=[1 0 0; 0 .7 0; .7 .8 1]; cs=cs-1;
  FP=E1-matchE; TP=matchE; FN=(allG-matchG)/n;
  for g=1:3, V(:,:,g,k)=max(0,1+FN*cs(1,g)+TP*cs(2,g)+FP*cs(3,g)); end
  V(:,2:end,:,k) = min(V(:,2:end,:,k),V(:,1:end-1,:,k));
  V(2:end,:,:,k) = min(V(2:end,:,:,k),V(1:end-1,:,:,k));
end

% if output file specified write results to disk
if(isempty(out)), return; end; fid=fopen(out,'w'); assert(fid~=1);
fprintf(fid,'%10g %10g %10g %10g %10g\n',[thrs cntR sumR cntP sumP]');
fclose(fid);
% up is writing a file evaluation data .txt xavysp
end
