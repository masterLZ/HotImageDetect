%%
%用于产生数据集
%指定位置指定大小
%用于写文章
clear;
clc;
close all;
lib_origin = '.\target\origin\';
lib_diffraction = '.\target\diffraction\';
lib_table = '.\target\table\';
cd 'D:\deeplearning - diffraction'
load yolov2detector1
lib_GDM = '.\target\diffraction\原图\';
%%
%unit mm
global lambda k;
global fx fy;
global R C m n;
global U_gaussian I_gaussian;
global xx yy;
%%
lambda = 632.8*1e-6;                                                   %波长
k = 2*pi/lambda;                                                           %波矢
dx = 6.45e-3;                                                  %采样区间:6.6048mm*6.6048mm(注：单个像素尺寸为6.45um，矩阵大小为1024*1024，因此采样宽度等于6.45*1024*10^(-6)=0.0066048m)
m = 2048;
n = 2048;
r=-n/2:1:n/2-1;     c=-m/2:1:m/2-1;     [R,C] = meshgrid(r,c);
%% 
% 频域

dfx=1/n/dx;     dfy=1/m/dx;     fx=gpuArray(R*dfx);       fy=gpuArray(C*dfy);
xx = lambda*fx;
yy = lambda*fy;
%% 
% 空域坐标

x = r*dx;       y = c*dx;       [X,Y] = meshgrid(x,y);
RR = sqrt(X.^2 + Y.^2);
N_gaussian = 2;
w0 = dx*m;
U_gaussian = exp(-(sqrt(X.^2+Y.^2)/w0).^N_gaussian);
%  U_gaussian = ones(size(U_gaussian));
%% 
global N1;
N1 = 800;
z_diffraction = [60,120,180,240];
% sizelib =[2,4,6,8,10];
sizelib = [2,4,6,8];
global center_distance;
global head;
%  head = 'Amplitude_';
head = 'phase_';
for factor = 0.1:0.1:1
    center_distance = round(factor*160);
    for i=1:1
        diffraction_distance = z_diffraction(i);
        E0 = zeros(m,n);
         Eout = E0;
         boundary_diffraction = zeros(1,4);
         numi = 0;
        
         for j = 2:2
             size = sizelib(j);
            [E0,Eout,boundary_diffraction, numi] = gengerate_damage(2,size,diffraction_distance);
            str = [lib_origin,head,num2str(diffraction_distance),'_',num2str(factor),'_',num2str(size),'.tiff'];
            E0 = im2uint16(imresize(abs(E0),[N1,N1]));
            imwrite(E0,str);
            str = [lib_diffraction,head,num2str(diffraction_distance),'_',num2str(factor),'_',num2str(size),'.tiff'];
            Eout =imresize(mat2gray(abs(Eout).^2),[N1,N1]);
            imwrite(Eout,str);
            str = [lib_GDM,head,num2str(diffraction_distance),'_',num2str(factor),'_',num2str(size),'.tiff'];
            imwrite(Eout,str);
            diffraction_save = Eout;
            diffraction_save = diffraction_save./max((max(diffraction_save)));
            diffraction_save = cat(3,diffraction_save,diffraction_save,diffraction_save);
            [bboxes, scores, labels] = detect(detector, diffraction_save);
            fprintf("%d\n",length(labels))
    %         scores = scores(scores>0.8);
    %         bboxes = bboxes(scores>0.8);
            
             I1=insertObjectAnnotation(diffraction_save,'rectangle',bboxes,scores);
             str = [lib_diffraction,head,'detect',num2str(diffraction_distance),'_',num2str(factor),'_',num2str(size),'.tiff'];
             imwrite(I1,str);
             
%             str = [lib_diffraction,'原图\',num2str(center_distance),'_',num2str(2),'_',num2str(size),'_',num2str(diffraction_distance),'.bmp'];
%             imwrite(diffraction_save,str);
            str = [lib_diffraction,'table\测试_',head,num2str(diffraction_distance),'_',num2str(factor),'_',num2str(size),'.csv'];
            csvwrite(str,bboxes);
            str = [lib_diffraction,'table\原图_',head,num2str(diffraction_distance),'_',num2str(factor),'_',num2str(size),'.csv'];
            csvwrite(str,boundary_diffraction);
%             str = [lib_table,num2str(diffraction_distance),'_',num2str(size),'.csv'];
%             csvwrite(str,boundary_diffraction);
            
         end
    end
end
run(' D:\deeplearning - diffraction\target\diffraction\原图\GDM_in_frequency_domain_final.m')
%%
function [Uf,Uf_boundary] = fresnel1(Phase_in,z_diffraction)
% compute fresnel diffraction
   % Uf 计算出射物光+背景
   % Uf_boundary 用于计算边界
    global xx yy k m n head U_gaussian;
    Amplitude =(U_gaussian);
    %U_initial(m/4:m/4+m/2-1,n/4:n/4+n/2-1) = U_initial1;
    Phase = zeros(size(Amplitude));
    Phase(m/4:m/4+m/2-1,n/4:n/4+n/2-1)=Phase_in;
    if strcmp(head,'phase_')
        U_initial = Amplitude.*exp(1i.*Phase);
    else
        U_initial = Amplitude.*(1-Phase);
    end
    
    U_initial = gpuArray(U_initial);
    Uf = (fft2(U_initial));       
    trans=fftshift(exp(1i*k*z_diffraction*sqrt(1-xx.^2-yy.^2)));                         %角谱衍射传递函数
    %trans=fftshift(exp(-1i*pi*lambda*z_diffraction*(fx.^2+fy.^2))); 
    result=Uf.*trans;
    Uf=(ifft2((result)));
    
    Amplitude = ones(size(Amplitude));
    U_initial = Amplitude.*exp(1i.*Phase);
    U_initial = gpuArray(U_initial);
    Uf_boundary = (fft2(U_initial));       
    trans=fftshift(exp(1i*k*z_diffraction*sqrt(1-xx.^2-yy.^2)));
    result=Uf_boundary.*trans;
    Uf_boundary=(ifft2((result)));
    
    Uf =gather(Uf(m/4:m/4+m/2-1,n/4:n/4+n/2-1));
    Uf_boundary =gather(Uf_boundary(m/4:m/4+m/2-1,n/4:n/4+n/2-1));
   % toc
end
function center_out = generate_damgepoint(num,z_diffraction) 
    global center_distance;
    L = round(center_distance/2);
    center_out=[L,-L;-L,L];
end

function [damage_out,damage_diffraction_out, boundary_diffraction, num] = gengerate_damage(num1, size, z_diffraction)
    global R C m n;
    R1 = R(m/4:m/4+m/2-1,n/4:n/4+n/2-1);
    C1 = C(m/4:m/4+m/2-1,n/4:n/4+n/2-1);
    damage_out = zeros(m/2, n/2);
    num = num1;
    center_out = generate_damgepoint(num,z_diffraction);
    size_out = size*ones(num,1);
    boundary_diffraction = zeros(num,4);
    for i = 1:num
        index = ((R1-(center_out(i,1))).^2+(C1-(center_out(i,2))).^2<size_out(i).^2);
        damage_out(index) = 1;
        damge_out_buffer = zeros(m/2, n/2);
        damge_out_buffer(index)=1;
        [~,Uf_boundary] = fresnel1((damge_out_buffer),z_diffraction);
        boundary_diffraction(i,:)=compute_boundary(Uf_boundary,center_out(i,:),z_diffraction);
    end
    [damage_diffraction_out,~] = fresnel1((damage_out),z_diffraction);
end

function boundary = compute_boundary(Eout,center,z_diffraction)
   global N1 m n;
   [M,N] = size(Eout);
   if(z_diffraction<80)
       z_diffraction = round(z_diffraction*3);
   else
       z_diffraction = round(z_diffraction*1.3);
   end
    center(1) = round((center(1)+m/4+1)/M*N1);
    center(2) = round((center(2)+n/4+1)/N*N1);   
    boundary = [(center(1)-z_diffraction/2),(center(2)-z_diffraction/2)...
        ,z_diffraction,z_diffraction];
    if (center(1)-z_diffraction/2)< 1
        boundary(1)=1;
    end
    if (center(2)-z_diffraction/2)< 1
        boundary(2)=1;
    end
    if(boundary(1)>N1)
        boundary(1)=N1;
    end
    if(boundary(2)>N1)
        boundary(2)=N1;
    end
    if (center(1)+z_diffraction/2)>N1
        boundary(3)=N1-boundary(1)-1;
    end
    if (center(2)+z_diffraction/2)>N1
        boundary(4)=N1-boundary(2)-1;
    end
    boundary = round(boundary);
end
