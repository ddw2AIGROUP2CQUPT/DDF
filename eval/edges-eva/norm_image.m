function i = norm_image(img,img_min, img_max)
    if nargin == 1
        img_min = 0;
        img_max =255;
    end
    i = double(img);
    epsilon = 1e-12;
    i = (i-min(i(:)))*(img_max-img_min)/((max(i(:))-min(i(:)))+epsilon)+img_min;

end