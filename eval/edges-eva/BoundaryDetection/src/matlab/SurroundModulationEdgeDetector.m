function EdgeImageResponse = SurroundModulationEdgeDetector(InputImage)
%SurroundModulationEdgeDetector  computed the edges of an image.
%
% inputs
%   InputImage  the input image.
%
% outputs
%   EdgeImageResponse  the edges of input image.
%
% This is the supplementary material of our article presented at the
% IJCV'17 "Feedback and Surround Modulated Boundary Detection".
%

% input image is from retina

% ranging it from 0 to 1
if max(InputImage(:)) > 1
  InputImage = double(InputImage);
  InputImage = InputImage ./ max(InputImage(:));
end

% size of RF in LGN
LgnSigma = 0.5;
InputImage = imfilter(InputImage, GaussianFilter2(LgnSigma), 'replicate');

% convert to opponent image this happens in LGN
OpponentImage = GetOpponentImage(InputImage);

nangles = 6;
EdgeImageResponse = DoV1(OpponentImage, LgnSigma, nangles);
EdgeImageResponse = DoV2(EdgeImageResponse, nangles, OpponentImage);

end

function EdgeImageResponse = DoV2(EdgeImageResponse, nangles, OpponentImage)

ExtraDimensions = [4, 5, 3];
FinalOrientations = [];

for i = 1:numel(ExtraDimensions)
  CurrentDimension = ExtraDimensions(i);
  
  switch CurrentDimension
    case 3
      [EdgeImageResponse, FinalOrientations] = CollapseChannels(EdgeImageResponse, FinalOrientations);
    case 4
      EdgeImageResponse = CollapsePlanes(EdgeImageResponse, OpponentImage);
    case 5
      [EdgeImageResponse, FinalOrientations] = CollapseOrientations(EdgeImageResponse);
  end
  
end

UseNonMax = true;

EdgeImageResponse = EdgeImageResponse ./ max(EdgeImageResponse(:));

if UseNonMax
  FinalOrientations = (FinalOrientations - 1) * pi / nangles;
  FinalOrientations = mod(FinalOrientations + pi / 2, pi);
  EdgeImageResponse = NonMaxChannel(EdgeImageResponse, FinalOrientations);
  EdgeImageResponse([1, end], :) = 0;
  EdgeImageResponse(:, [1, end]) = 0;
  EdgeImageResponse = EdgeImageResponse ./ max(EdgeImageResponse(:));
end

end

function EdgeImageResponse = DoV1(OpponentImage, LgnSigma, nangles)

FarSurroundLevels = 4;

[rows, cols, chns] = size(OpponentImage);
EdgeImageResponse = zeros(rows, cols, chns, FarSurroundLevels, nangles);

% the neurons in V1 are 2 times larger than LGN.
lgn2v1 = 2.7;

for i = 1:FarSurroundLevels
  iiedge = GaussianGradientEdges(OpponentImage, LgnSigma * lgn2v1, nangles, i);
  EdgeImageResponse(:, :, :, i, :) = iiedge;
end

end

function OpponentImage = GetOpponentImage(InputImage)

% gamma correction
SqrIm = sqrt(InputImage);

if size(InputImage, 3) == 3
  % equilibrium single-opponent cells
  OpponentImage(:, :, 1) = SqrIm(:, :, 1) - SqrIm(:, :, 2);
  OpponentImage(:, :, 2) = SqrIm(:, :, 3) - mean(SqrIm(:, :, 2:3), 3);
  OpponentImage(:, :, 3) = (mean(SqrIm(:, :, 1:3), 3));
  
  % imbalanced single-opponent cells
  OpponentImage(:, :, end + 1) = InputImage(:, :, 1) - 0.7 .* InputImage(:, :, 2);
  OpponentImage(:, :, end + 1) = InputImage(:, :, 3) - 0.7 .* mean(InputImage(:, :, 2:3), 3);
  
  % equivalent of the feedback channel to speed up the process
  OpponentImage(:, :, end + 1) = sqrt(CircularLocalStdContrast(rgb2gray(SqrIm), 2.5));
else
  OpponentImage(:, :, 1) = SqrIm;
  OpponentImage(:, :, 2) = sqrt(CircularLocalStdContrast(SqrIm));
end

end

function [EdgeImageResponse, FinalOrientations] = CollapseChannels(EdgeImageResponse, SelectedOrientations)

CurrentDimension = 3;

[rows, cols, ~] = size(EdgeImageResponse);
FinalOrientations = zeros(rows, cols);

SumEdgeResponse = sum(EdgeImageResponse, CurrentDimension);

[~, MaxInds] = max(EdgeImageResponse, [], CurrentDimension);

EdgeImageResponse = SumEdgeResponse;

if ~isempty(SelectedOrientations)
  for c = 1:max(MaxInds(:))
    corien = SelectedOrientations(:, :, c);
    FinalOrientations(MaxInds == c) = corien(MaxInds == c);
  end
end

end

function EdgeImageResponse = CollapsePlanes(inEdgeImageResponse, OpponentImage)

[~, ~, ~, plns, oris] = size(inEdgeImageResponse);

lstd = CircularLocalStdContrast(OpponentImage, 15 / 2);
pstd = 1 - lstd;

EdgeImageResponse = inEdgeImageResponse(:, :, :, 1, :);

for i = 2:plns
  for j = 1:oris
    CurrentChannel = inEdgeImageResponse(:, :, :, i, j) .* (pstd ./ i);
    EdgeImageResponse(:, :, :, 1, j) = EdgeImageResponse(:, :, :, 1, j) + CurrentChannel;
  end
end

EdgeImageResponse = max(EdgeImageResponse, 0);

% normalising the sum of all planes
% it doesn't make the results better, but it makes a logical sense.
for i = 1:size(EdgeImageResponse, 3)
  CurrentChannel = EdgeImageResponse(:, :, i, :, :);
  CurrentChannel = CurrentChannel ./ max(CurrentChannel(:));
  EdgeImageResponse(:, :, i, :, :) = CurrentChannel;
end

end

function [EdgeImageResponse, FinalOrientations] = CollapseOrientations(EdgeImageResponse)

CurrentDimension = 5;
nThetas = size(EdgeImageResponse, CurrentDimension);

StdImg = std(EdgeImageResponse, [], CurrentDimension);

[EdgeImageResponse, FinalOrientations] = max(EdgeImageResponse, [], CurrentDimension);

v1sigma = 0.5 * 2.7;
v1v2 = 2.7;
SurroundEnlagre = 5;

% V2 area pie-wedge shape
for c = 1:size(EdgeImageResponse, 3)
  CurrentChannel = EdgeImageResponse(:, :, c);
  CurrentOrientation = FinalOrientations(:, :, c);
  
  si = CircularLocalStdContrast(CurrentChannel, 45 / 2);
  si = si ./ max(si(:));
  si = max(si(:)) - si;
  si = NormaliseChannel(si, 0.7, 1.0, [], []);
  
  for t = 1:nThetas
    theta = (t - 1) * pi / nThetas;
    theta = theta + (pi / 2);
    
    xsigma = v1sigma * v1v2;
    
    ysigma = xsigma / 8;
    
    v2responsec = imfilter(EdgeImageResponse(:, :, c), GaussianFilter2(xsigma, ysigma, 0, 0, theta), 'symmetric');
    v2responses = imfilter(EdgeImageResponse(:, :, c), GaussianFilter2(xsigma * SurroundEnlagre, ysigma * SurroundEnlagre, 0, 0, theta), 'symmetric');
    
    v2response = max(v2responsec - si .* v2responses, 0);
    CurrentChannel(CurrentOrientation == t) = v2response(CurrentOrientation == t);
  end
  EdgeImageResponse(:, :, c) = CurrentChannel;
end

EdgeImageResponse = EdgeImageResponse .* (StdImg + 1);

end

function d = NonMaxChannel(d, t)

for i = 1:size(d, 3)
  d(:, :, i) = d(:, :, i) ./ max(max(d(:, :, i)));
  d(:, :, i) = nonmax(d(:, :, i), t(:, :, i));
  d(:, :, i) = max(0, min(1, d(:, :, i)));
end

end

function OutEdges = GaussianGradientEdges(InputImage, V1RfSize, nangles, clevel)

[w, h, d] = size(InputImage);

thetas = zeros(1, nangles);
for i = 1:nangles
  thetas(i) = (i - 1) * pi / nangles;
end

OutEdges = zeros(w, h, d, nangles);
for c = 1:d
  OutEdges(:, :, c, :) = gedges(InputImage(:, :, c), V1RfSize, thetas, clevel);
end

end

function rfresponse = gedges(InputImage, sigma, thetas, clevel)

[rows1, cols1, ~] = size(InputImage);

gresize = GaussianFilter2(0.3 * (clevel - 1));

InputImage = imfilter(InputImage, gresize, 'replicate');
InputImage = imresize(InputImage, 1 / (2.0 ^ (clevel - 1)));

ElongatedFactor = 0.5;

[rows2, cols2, ~] = size(InputImage);

nThetas = length(thetas);
rfresponse = zeros(rows2, cols2, nThetas);

for t = 1:nThetas
  theta1 = thetas(t);
  
  dorf = Gaussian2Gradient1(sigma, theta1, ElongatedFactor);
  doresponse = abs(imfilter(InputImage, dorf, 'symmetric'));
  
  rfresponse(:, :, t) = doresponse;
end

CentreSize = size(dorf, 1);
rfresponse = SurroundOrientation(InputImage, CentreSize, rfresponse, sigma);

rfresponse = imresize(rfresponse, [rows1, cols1]);
rfresponse = imfilter(rfresponse, gresize, 'replicate');

rfresponse = rfresponse ./ max(rfresponse(:));

end

function orfresponse = SurroundOrientation(InputImage, GaussianSize, irfresponse, sigma)

irfresponse = irfresponse ./ max(irfresponse(:));

if ~isempty(InputImage)
  AverageSize = GaussianSize(1) / 2;
  
  SurroundContrast = CircularLocalStdContrast(InputImage, GaussianSize(1) / 2);
  SurroundContrast = SurroundContrast ./ max(SurroundContrast(:));
  
  w11 = 1 - SurroundContrast;
  w12 = -SurroundContrast;
  
  w21 = -SurroundContrast;
  w22 = 1 - SurroundContrast;
else
  AverageSize = 7.5;
  
  w11 = GaussianSize;
  w12 = GaussianSize;
  
  w21 = GaussianSize;
  w22 = GaussianSize;
end

ysigma = 0.1;
xsigma = 3 * sigma;
AxesFactor = 4;
CentreZeroSize = [1, 1];

AverageFilter = CircularAverage(AverageSize);
AverageFilter = CentreCircularZero(AverageFilter, AverageSize / 5);
AverageFilter = AverageFilter ./ sum(AverageFilter(:));
FullSurroundOrientation = imfilter(irfresponse, AverageFilter);

orfresponse = irfresponse;

% in the oppositie orientation, orthogonality facilitates and parallelism
% suppresses.
nThetas = size(irfresponse, 3);
for t = 1:nThetas
  theta1 = (t - 1) * pi / nThetas;
  theta2 = theta1 + (pi / 2);
  
  o = t + (nThetas / 2);
  if o > nThetas
    o = t - (nThetas / 2);
  end
  
  oppresponse = irfresponse(:, :, o);
  doresponse = irfresponse(:, :, t);
  
  SameOrientationGaussian = CentreZero(GaussianFilter2(xsigma, ysigma, 0, 0, theta1), CentreZeroSize);
  OrthogonalOrientationGaussian = CentreZero(GaussianFilter2(xsigma / AxesFactor, ysigma, 0, 0, theta2), CentreZeroSize);
  
  axis1(:, :, 1) = imfilter(doresponse, SameOrientationGaussian, 'symmetric');
  axis2(:, :, 1) = imfilter(doresponse, OrthogonalOrientationGaussian, 'symmetric');
  
  axis1(:, :, 2) = imfilter(oppresponse, SameOrientationGaussian, 'symmetric');
  axis2(:, :, 2) = imfilter(oppresponse, OrthogonalOrientationGaussian, 'symmetric');
  
  doresponse = doresponse + w11 .* axis1(:, :, 1) + w12 .* axis1(:, :, 2);
  doresponse = doresponse + w21 .* axis2(:, :, 1) + w22 .* axis2(:, :, 2);
  
  doresponse = doresponse + 0.5 .* FullSurroundOrientation(:, :, o);
  
  orfresponse(:, :, t) = doresponse;
end

orfresponse = max(orfresponse, 0);

orfresponse = orfresponse ./ max(orfresponse(:));

end
