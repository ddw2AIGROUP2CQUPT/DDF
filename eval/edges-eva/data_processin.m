
% Read images
dataset_name = 'BIPED';
base_dir = '/opt/dataset';
if isequal(dataset_name,'BSDS')
    dataset_dir = fullfile(dataset_name,'test');
elseif isequal(dataset_name,'BIPED')
    dataset_dir = fullfile(dataset_name,'edges/imgs/test/rgbr');
elseif isequal(dataset_name,'MULTICUE')
    dataset_dir = fullfile(dataset_name,'test/imgs');
else
    disp('test/imgs');   
    
end

imgs_dir = fullfile(base_dir,dataset_dir);
list_imgs = dir(fullfile(imgs_dir,'*.jpg'));
new_folder = fullfile('imgs_with_noise',dataset_name);
gau0_dir = fullfile(new_folder,'gau0');
gau1_dir = fullfile(new_folder,'gau1');
gau2_dir = fullfile(new_folder,'gau2');
if ~exist(gau0_dir,'dir')
        mkdir(gau0_dir);
end
if ~exist(gau1_dir,'dir')
        mkdir(gau1_dir);
end
if ~exist(gau2_dir,'dir')
        mkdir(gau2_dir);
end

n = length(list_imgs);
for i=1:n
    tmp_img=imread(fullfile(imgs_dir,list_imgs(i).name));
    i_gau0 = imnoise(tmp_img,'gaussian', 0.0,0.0001); % 1 percent
    i_gau1 = imnoise(tmp_img,'gaussian',0.0,0.0025); % 5 percent
    i_gau2 = imnoise(tmp_img,'gaussian',0.0, 0.01); % 10 percent
    % BIPED 0.6060, 
%     i_gamma = gamma_correct(i_ungamma,0.6060,true);
%     i_gamma = uint8(norm_image(i_gamma));
    
    imwrite(i_gau0,fullfile(gau0_dir,list_imgs(i).name));
    imwrite(i_gau1,fullfile(gau1_dir,list_imgs(i).name));
    imwrite(i_gau2,fullfile(gau2_dir,list_imgs(i).name));
    disp(num2str(i));
    disp(list_imgs(i).name);
    
end
return;
%% New noise addition 


% Read images
dataset_name = 'BIPED';
base_dir = '/opt/dataset';
if isequal(dataset_name,'BSDS')
    dataset_dir = fullfile(dataset_name,'test');
elseif isequal(dataset_name,'BIPED')
    dataset_dir = fullfile(dataset_name,'edges/imgs/test/rgbr');
elseif isequal(dataset_name,'MULTICUE')
    dataset_dir = fullfile(dataset_name,'test/imgs');
else
    disp('test/imgs');   
    
end

imgs_dir = fullfile(base_dir,dataset_dir);
list_imgs = dir(fullfile(imgs_dir,'*.jpg'));
new_folder = fullfile('imgs_with_noise',dataset_name);
salt_dir = fullfile(new_folder,'salt');
gau_dir = fullfile(new_folder,'gaussian');
gam_dir = fullfile(new_folder,'gamma');
if ~exist(salt_dir,'dir')
        mkdir(salt_dir);
end
if ~exist(gau_dir,'dir')
        mkdir(gau_dir);
end
if ~exist(gam_dir,'dir')
        mkdir(gam_dir);
end

n = length(list_imgs);
for i=1:n
    tmp_img=imread(fullfile(imgs_dir,list_imgs(i).name));
    i_gau0 = imnoise(tmp_img,'salt & pepper', 0.02);
    i_gau = imnoise(tmp_img,'gaussian');
    i_ungamma = gamma_correct(tmp_img,0.4040,false);
    % BIPED 0.6060, 
    i_gamma = gamma_correct(i_ungamma,0.6060,true);
    i_gamma = uint8(norm_image(i_gamma));
    
    imwrite(i_salt,fullfile(salt_dir,list_imgs(i).name));
    imwrite(i_gau,fullfile(gau_dir,list_imgs(i).name));
    imwrite(i_gamma,fullfile(gam_dir,list_imgs(i).name));
    disp(num2str(i));
    disp(list_imgs(i).name);
    
end