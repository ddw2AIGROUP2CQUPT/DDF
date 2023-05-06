function FilterWidth = CalculateGaussianWidth(sigma, MaxWidth)
%CalculateGaussianWidth  calculates the descrete width of Gaussian filter.
%
% inputs
%   sigma     the sigma of Gaussian filter.
%   MaxWidth  the maximum allowed width of the filter, default is 100.
%
% outputs
%   FilterWidth  the descrete width of the filter
%
% This is the supplementary material of our article presented at the
% IJCV'17 "Feedback and Surround Modulated Boundary Detection".
%

if nargin < 2
  MaxWidth = 100;
end

threshold = 1e-4;
pw = 1:MaxWidth;
FilterWidth = find(exp(-(pw .^ 2) / (2 * sigma .^ 2)) > threshold, 1, 'last');
if isempty(FilterWidth)
  FilterWidth = 1;
end
FilterWidth = FilterWidth .* 2 + 1;

end
