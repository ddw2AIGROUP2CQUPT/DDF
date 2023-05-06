function ImageContrast = CircularLocalStdContrast(InputImage, SurroundRadius, CentreRadius)
%CircularLocalStdContrast  local std in a circular region.
%
% inputs
%   InputImage      the input image with n channel.
%   SurroundRadius  the radius of the neighbourhoud.
%   CentreRadius    if provided the centre is set to 0.
%
% outputs
%   ImageContrast  calculated local std of each channel.
%
% This is the supplementary material of our article presented at the
% IJCV'17 "Feedback and Surround Modulated Boundary Detection".
%

InputImage = double(InputImage);

if nargin < 2
  SurroundRadius = 2.5;
end
if nargin < 3
  CentreRadius = 0;
end

hc = CircularAverage(SurroundRadius);
hc = CentreCircularZero(hc, CentreRadius);
hc = hc ./ sum(hc(:));
MeanCentre = imfilter(InputImage, hc, 'symmetric');
stdv = (InputImage - MeanCentre) .^ 2;
MeanStdv = imfilter(stdv, hc, 'symmetric');
ImageContrast = sqrt(MeanStdv);

end
