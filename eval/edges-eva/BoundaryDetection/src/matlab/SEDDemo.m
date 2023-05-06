% This script runs our model, Surround-modulation Edge Detection (SED)
% against four sample files from BSDS500.
%
% This is the supplementary material of our article presented at the
% IJCV'17 "Feedback and Surround Modulated Boundary Detection".
%

%%
im1 = imread('../../data/imgs/36046.jpg');
edge1 = SurroundModulationEdgeDetector(im1);
figure, subplot(1, 2, 1); imshow(im1); subplot(1, 2, 2); imshow(edge1, []);

%%
im2 = imread('../../data/imgs/100007.jpg');
edge2 = SurroundModulationEdgeDetector(im2);
figure, subplot(1, 2, 1); imshow(im2); subplot(1, 2, 2); imshow(edge2, []);

%%
im3 = imread('../../data/imgs/112056.jpg');
edge3 = SurroundModulationEdgeDetector(im3);
figure, subplot(1, 2, 1); imshow(im3); subplot(1, 2, 2); imshow(edge3, []);

%%
im4 = imread('../../data/imgs/196027.jpg');
edge4 = SurroundModulationEdgeDetector(im4);
figure, subplot(1, 2, 1); imshow(im4); subplot(1, 2, 2); imshow(edge4, []);
