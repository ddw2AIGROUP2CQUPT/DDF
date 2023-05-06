function SurroundMat = CentreZero(SurroundMat, CentreSize)
%CentreZero  sets the centre indeces to 0.
%
% inputs
%   SurroundMat  the matrix that its centre will be set to 0.
%   CentreSize   the size of centre.
%
% outputs
%   SurroundMat  the centred set to 0 matrix.
%
% This is the supplementary material of our article presented at the
% IJCV'17 "Feedback and Surround Modulated Boundary Detection".
%

[ws, hs] = size(SurroundMat);
wc = CentreSize(1);
hc = CentreSize(2);

if wc == 0 || hc == 0
  return;
end

lw = ceil(ws / 2) - floor(wc / 2);
hw = lw + wc - 1;
lh = ceil(hs / 2) - floor(hc / 2);
hh = lh + hc - 1;

SurroundMat(lw:hw, lh:hh) = 0;

end
