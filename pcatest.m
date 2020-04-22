clear;close all;
I = im2double((imread('D:\0001.bmp')));
phi = I;
I = I(1:1600,1:1600);
I = im2double(rgb2gray(imread('D:\deeplearning - diffraction\diffraction\110_1.bmp')));
% mean_phi=repmat(mean(phi,2),1,800);
% X=phi-mean_phi;
figure,imshow(I,[])
% [eigenvector,eigenvalue]=eig(phi);
% A1 = eigenvector'*eigenvalue*eigenvector;
% figure,imshow(abs(A1),[])

[u,s,v]=svd(I);
sval_nums = 2;
low = 1;
u1 = zeros(size(u));
s1=u1;v1 =u1;
u1(:,low:sval_nums) =u(:,low:sval_nums);
s1(:,low:sval_nums) =s(:,low:sval_nums);
v1(:,low:sval_nums) =v(:,low:sval_nums);
I1 = u1*s1*v1';
% I1 = imgaussfilt(I1,20);
figure,imshow(I1,[])
I2 = I./I1;
figure,imshow(I2,[])
% I2 =I1;
% figure,imshow(I2,[])
% I2 = gpuArray(I2);
% I2 = imgaussfilt(I2,5);
% H = gpuArray(fspecial('sobel'));
% % x=-3:3;y=x;
% % [X,Y]=meshgrid(x,y);
% % H = X./(pi*2^2).*exp(-(X.^2+Y.^2)./2./2^2);
% dx = gather(imfilter(I2,H,'circular','same'));
% dy = gather(imfilter(I2,H','circular','same'));
% arrow = mat2gray(atan2(dx,dy));
% figure,imshow(arrow)
