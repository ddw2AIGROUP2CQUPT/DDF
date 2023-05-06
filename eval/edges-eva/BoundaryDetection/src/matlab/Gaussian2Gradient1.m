function g1 = Gaussian2Gradient1(sigma, theta, seta)
%Gaussian2Gradient1  the first derivative Gaussian kernel.
%   Explanation  http://mathworld.wolfram.com/GaussianFunction.html
%
% inputs
%   sigma  width of the Gaussian kernel.
%   theta  direction of derivative.
%   seta   rspatial aspect ratio, default 0.5;
%
% outputs
%   g1  the first derivative of Gaussian kernel.
%
% This is the supplementary material of our article presented at the
% IJCV'17 "Feedback and Surround Modulated Boundary Detection".
%

if nargin < 3
  seta = 0.50;
end

width = CalculateGaussianWidth(sigma);
width = floor(width / 2);

[xs, ys] = meshgrid(-width:width, -width:width);
x1 = xs .* cos(theta) + ys .* sin(theta);
y1 = -xs .* sin(theta) + ys .* cos(theta);

g1 = -x1 .* exp(-((x1 .^ 2) + (y1 .^ 2) * (seta ^ 2)) / (2 * (sigma ^ 2))) / (pi * (sigma ^ 2));

end
