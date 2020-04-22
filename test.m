
clear;
close all;

%%
global m n m1 n1 dx dy;
m = 2048;
n = m;
m1 =1500;
n1 = m1;
dx = 5.5e-3;
dy=dx;
diffraction_init;

%%
% load 'E:\Anacondadoc\hot-imging-detect\code\detector.mat';
load yolov2detector3
%%
% path = 'C:\Users\hp\Desktop\损伤点检测\图片\多个点\Image010.bmp';
% I1 = imreadOnechannel(path);
% I1 = I1(10:1010,300:1300);
%Ein = imread('D:\deeplearning - diffraction\diffraction\30_1.bmp');
path = 'D:\0007.bmp';
%path = 'D:\deeplearning - diffraction\target\diffraction\原图\16_2_8_300.bmp';
% path = 'D:\deeplearning - diffraction\target\diffraction\测试图\Amplitude_120_2_15.bmp';
% path = 'C:\Users\hp\Desktop\损伤点检测\图片\多个点\Image007.bmp';
I0 = imreadOnechannel(path);
% I0 = I0(10:1010,300:1300);
% I0= mat2gray(I1-I0);
% I0 = I0(10:1010,200:1200);
I0 = imresize(I0,[256,256]);


I = I0;
    [u,s,v]=svd(I0);
    sval_nums = 2;
    low = 1;
    u1 =zeros(size(u));
    s1=u1;v1 =u1;
    u1(:,low:sval_nums) =u(:,low:sval_nums);
    s1(:,low:sval_nums) =s(:,low:sval_nums);
    v1(:,low:sval_nums) =v(:,low:sval_nums);
    I1 = u1*s1*v1';
    I1 = imgaussfilt(I1,20);
    I0 = (mat2gray(I./I1));
    I = I0;
    I = imgaussfilt(I,1);
    %
    H = (fspecial('sobel'));
    dx1 = (imfilter(I,H,'circular','same'));
    dy1 = (imfilter(I,H','circular','same'));
    arrow = im2uint8 (mat2gray(atan2(dx1,dy1)));
    arrow = cat(3,arrow,arrow,arrow);
    
%      arrow = imread('D:\deeplearning - diffraction\diffraction\Three_write\70_48.bmp');
figure,imshow(arrow)
I = cat(3,I0,I0,I0);
I1 = imreadOnechannel(path);
% I1=I1(10:1010,300:1300);
% I1 = I1(10:1010,200:1200);
I1 = cat(3,I1,I1,I1);
[bboxes, scores, labels] = detect(detector, arrow);
bboxes = bboxes(scores>0.5,:)./255.*size(I1,1);
scores = scores(scores>0.5);

I1=insertObjectAnnotation(I1,'rectangle',bboxes,scores, 'LineWidth',4,'Font','Times New Roman', 'FontSize',30);

figure,imshow(I1),title('yolov2')
%%
% m = 2048;
% n = m;
% m1 = 1700;
% n1 = 1700;
%%
diffraction_z = 10:1:600;
Ein = imreadOnechannel(path);
% [m1,n1]=size(Ein);
% m = 2048;n=m;
bboxes = floor(bboxes);
I_final = zeros(size(diffraction_z,1),size(bboxes,1));
finalwrite = zeros(size(diffraction_z,2),4);

 I_buffer = Ein;
figure,
for j= 1:length(diffraction_z)
       z_diffraction_out = diffraction_z(j);
       Uf = fresnel_prog(exp(1i*6*pi*I_buffer),z_diffraction_out);
       If = abs(Uf).^2;
       for i = 1:size(bboxes,1)
        I_final(j,i) = max(max(If(bboxes(i,2)+round(bboxes(i,4)/5*2):bboxes(i,2)+round(bboxes(i,4)/5*3),...
                                bboxes(i,1)+round(bboxes(i,3)/5*2):bboxes(i,1)+round(bboxes(i,3)/5*3))));
       end
       imagesc(If);
       title([num2str(z_diffraction_out),'mm    ',num2str(I_final(j))])
       pause(0.01)
end
%%
I_finalCopy = I_final;
 for i = 1:size(bboxes,1)
  I_final(:,i) = mapminmax(PLDpreprocess(I_final(:,i))',0,1);
    figure,plot(diffraction_z,I_final(:,i) );
    [max_I,index_I] = max(I_final(:,i) );
    finalwrite(:,i) =  I_final(:,i) ;
    fprintf('\n峰值是%f，峰值在%dmm',max_I,diffraction_z(index_I));
end
%%
I_max = abs(fresnel1(exp(1i*I_buffer),300)).^2;
finalwrite(:,end)=diffraction_z;
csvwrite('测试结果.csv',finalwrite)
figure,imagesc(I_max(1:m1,1:n1)),colormap('hot'),axis off
%%
function Uf = fresnel1(U_initial,z_diffraction)
% compute fresnel diffraction
   % tic
    global lambda fx fy k m n m1 n1;
    U_initia = zeros(m,n);
    U_initia(1:m1,1:n1)=U_initial;
    U_initia = gpuArray(U_initia);
    Uf = (fft2(U_initia));   
    xx = lambda*fx;
    yy = lambda*fy;
    trans=fftshift(exp(1i*k*z_diffraction*sqrt(1-xx.^2-yy.^2)));                         %角谱衍射传递函数
    %trans=fftshift(exp(-1i*pi*lambda*z_diffraction*(fx.^2+fy.^2))); 
    result=Uf.*trans;
    Uf=gather(ifft2((result)));    
   % toc
end
