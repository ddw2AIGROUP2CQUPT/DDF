function h = GaussianFilter2(sigmax, sigmay, meanx, meany, theta)
%GaussianFilter2  Gaussian kernel.
%   Explanation  http://mathworld.wolfram.com/GaussianFunction.html
%
% inputs
%   sigmax  the sigma in x direction, default 0.5.
%   sigmay  the sigma in y direction, default 0.5.
%   meanx   the mean in x direction, default centre.
%   meany   the mean in y direction, default centre.
%   theta   the rotation angle.
%
% outputs
%   h  the Gaussian kernel.
%
% This is the supplementary material of our article presented at the
% IJCV'17 "Feedback and Surround Modulated Boundary Detection".
%

if nargin < 1 || isempty(sigmax)
  sigmax = 0.5;
end
if nargin < 2 || isempty(sigmay)
  sigmay = sigmax;
end

if sigmax == 0 || sigmay == 0
  h = 1;
  return;
end

MaxSigma = max(sigmax, sigmay);
sizex = CalculateGaussianWidth(MaxSigma);
sizey = sizex;

if nargin < 3 || isempty(meanx)
  meanx = 0;
end
if nargin < 4 || isempty(meany)
  meany = 0;
end

if nargin < 5 || isempty(theta)
  theta = 0;
end

centrex = (sizex + 1) / 2;
centrey = (sizey + 1) / 2;
centrex = centrex + (meanx * centrex);
centrey = centrey + (meany * centrey);

xs = linspace(1, sizex, sizex)' * ones(1, sizey) - centrex;
ys = ones(1, sizex)' * linspace(1, sizey, sizey) - centrey;

a =  cos(theta) ^ 2 / 2 / sigmax ^ 2 + sin(theta) ^ 2 / 2 / sigmay ^ 2;
b = -sin(2 * theta) / 4 / sigmax ^ 2 + sin(2 * theta) / 4 / sigmay ^ 2;
c =  sin(theta) ^ 2 / 2 / sigmax ^ 2 + cos(theta) ^ 2 / 2 / sigmay ^ 2;

h = exp(-(a * xs .^ 2 + 2 * b * xs .* ys + c * ys .^ 2));

h = h ./ sum(h(:));

end
