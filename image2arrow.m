%%
clear;
clc;
close all;
%%
pic_path ='D:\deeplearning - diffraction\diffraction';
pic_path_write = 'D:\deeplearning - diffraction\diffraction\Three_write';
pic_dicOutput = dir(fullfile(pic_path,'*.bmp'));
imageFilename = cell(size(pic_dicOutput));
% [X,Y] = meshgrid(1:800);
f = waitbar(0,'¿ªÊ¼°áÔË');
fileLength = length(imageFilename);
tic
parfor i = 1:fileLength
%     waitbar(i/length(pic_dicOutput),f);
    pic_fullPath = fullfile(pic_dicOutput(i).folder,pic_dicOutput(i).name);
    I = gpuArray(im2double(rgb2gray(imread(pic_fullPath))));
    I = imnoise(I,'gaussian',0,0.0001);%     

    
    [u,s,v]=svd(I);
    sval_nums = 2;
    low = 1;
    u1 = gpuArray(zeros(size(u)));
    s1=u1;v1 =u1;
    u1(:,low:sval_nums) =u(:,low:sval_nums);
    s1(:,low:sval_nums) =s(:,low:sval_nums);
    v1(:,low:sval_nums) =v(:,low:sval_nums);
    I1 = u1*s1*v1';
    I1 = imgaussfilt(I1,20);
    I2 = mat2gray(I./I1);
    I = I2;
    I = imgaussfilt(I,1);
    %
    H = gpuArray(fspecial('sobel'));
    dx = gather(imfilter(I,H,'circular','same'));
    dy = gather(imfilter(I,H','circular','same'));
     arrow = mat2gray(atan2(dx,dy));
%     arrow = mat2gray(atan(dx./(dy+eps)));
    arrow = cat(3,arrow,arrow,arrow);
%      figure,imshow(arrow)
    write_path = fullfile(pic_path_write,pic_dicOutput(i).name);
%     I2 =imresize(I2,[256,256]);
%     I = gather( cat(3,I2,I2,I2));
    imwrite(arrow,write_path);
%     figure,imshow(I,[])
end 
toc
close(f)
% I = gpuArray(im2double(imread('D:\0001.bmp')));
% I = imgaussfilt(I,3);
% H = gpuArray(fspecial('sobel'));
% dx = gather(imfilter(I,H,'circular','same'));
% dy = gather(imfilter(I,H','circular','same'));
% arrow = mat2gray(atan2(dy,dx));
% figure,imshow(arrow)
