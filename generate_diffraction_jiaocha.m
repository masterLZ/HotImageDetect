%%
%用于创建一些新交叠在一起的点作为训练集
%对参数进行重新训练
%%
clear;
clc;
close all;
lib_origin = '.\origin\jiaocha\';
lib_diffraction = '.\diffraction\jiaocha\';
lib_table = '.\table\jiaocha\';
lib_buffer = '.\buffer\jiaocha\';
cd 'D:\deeplearning - diffraction'
%%
%unit mm
global lambda k;
global fx fy;
global R C m n;
global U_gaussian ;
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
N_gaussian = 1;
w0 = dx*m;
U_gaussian = exp(-(sqrt(X.^2+Y.^2)/w0).^N_gaussian);
%%
global N1;
N1 = 800;
I_write = zeros(N1,N1,3);
%初始损伤点
figure,
 for diffraction_distance =10:30:60
     fprintf('\n%d',diffraction_distance);
     E0 = zeros(m,n);
     Eout = E0;
     boundary_diffraction = zeros(1,4);
     numi = 0;
     for j = 1:400
        [E0,Eout,boundary_diffraction, numi,Uf_out] = gengerate_damage(3,8,diffraction_distance);
        str = [lib_origin,num2str(diffraction_distance),'_',num2str(j),'.bmp'];
        imwrite(imresize(abs(E0),[N1,N1]),str);
        str = [lib_diffraction,num2str(diffraction_distance),'_',num2str(j),'.bmp'];
        I_write1 = imresize(abs(Eout).^2,[N1,N1]);
        I_write(:,:,1)=I_write1;
        I_write(:,:,2)=I_write1;
        I_write(:,:,3)=I_write1;
        diffraction_save = (I_write);
        diffraction_save = diffraction_save./max((max(diffraction_save)));
        imwrite(diffraction_save,str);
        str = [lib_table,num2str(diffraction_distance),'_',num2str(j),'.csv'];
        csvwrite(str,boundary_diffraction);

%         imshow(diffraction_save)
%         for i = 1:numi
%             rectangle('Position',boundary_diffraction(i,:), 'Edgecolor','r');
%         end
%         str = [lib_buffer,num2str(diffraction_distance),'_',num2str(j),'.bmp'];
%         title([num2str(diffraction_distance),'-',num2str(j)])
%         saveas(gcf,str,'bmp');
     end
 end

% figure,imagesc(imresize(abs(E0),[N1,N1]))
% figure,imagesc(imresize(abs(Eout),[N1,N1]))
% 
% hold on ;
% for i = 1:numi
%   rectangle('Position',boundary_diffraction(i,:), 'Edgecolor','r');  
% end
% %%
I_buffer = mat2gray(abs(Eout).^2).*2*pi;
%
%%
I_max = abs(fresnel1(exp(1i*I_buffer),210)).^2;
figure,imagesc(I_max)
I_buffer = mat2gray(abs(Uf_out).^2).*2*pi;
I_max = abs(fresnel1(exp(1i*I_buffer),210)).^2;
figure,imagesc(I_max)
% Using_faster_Rcnn
%%
function [Uf,Uf_boundary] = fresnel1(Phase_in,z_diffraction)
% compute fresnel diffraction
   % Uf 计算出射物光+背景
   % Uf_boundary 用于计算边界
    global xx yy k m n U_gaussian;
%     tic
    Amplitude =(U_gaussian);
    %U_initial(m/4:m/4+m/2-1,n/4:n/4+n/2-1) = U_initial1;
    Phase = zeros(size(Amplitude));
    Phase(m/4:m/4+m/2-1,n/4:n/4+n/2-1)=Phase_in;
    U_initial = Amplitude.*exp(1i.*Phase);
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
%     toc
end
function center_out = generate_damgepoint(num)
%首先获得一个损伤点，然后将第一个损伤点周围n个size内距离的点设为第二个
%依次往下迭代
    global m n;
    center_out = zeros(num, 2);
    for i =1:num
        if(i==1)
            center_out(i,1) = randi(round(m/2), 1, 1)-m/4;
            center_out(i,2) = randi(round(n/2), 1, 1)-n/4;
        else 
            neg = 1;
            if randi([0,1])
                neg = -1;
            end
            cx = center_out(i-1,1)+ neg*randi([90,120]);
            
            neg = 1;
            if randi([0,1])
                neg = -1;
            end
            cy = center_out(i-1,2)+ neg*randi([90,120]);
            
            if cx <=(-m/4+1)
                cx =-m/4+1;
            end
            if cx >m/4
                cx = m/4;
            end
            if cy < -n/4+1
                cy = -n/4+1;
            end
            if cy >n/4
                cy = n/4;
            end
            center_out(i,1) = cx;
            center_out(i,2) = cy;
        end
    end
end

function [damage_out,damage_diffraction_out, boundary_diffraction, num,Uf_out] = gengerate_damage(num1, size, z_diffraction)
    
    global R C m n;
    R1 = R(m/4:m/4+m/2-1,n/4:n/4+n/2-1);
    C1 = C(m/4:m/4+m/2-1,n/4:n/4+n/2-1);
    damage_out = zeros(m/2, n/2);
    num = randi([2,num1]);
    size_out = randi([3,size],num,1);
    center_out = generate_damgepoint(num);    
    boundary_diffraction = zeros(num,4);
    
    for i = 1:num
        index = ((R1-(center_out(i,1))).^2+(C1-(center_out(i,2))).^2<size_out(i).^2);
        damage_out(index) = 1;
        damge_out_buffer = zeros(m/2, n/2);
        damge_out_buffer(index)=1;
        [~,Uf_boundary] = fresnel1((damge_out_buffer),z_diffraction);
        boundary_diffraction(i,:)=compute_boundary(Uf_boundary,center_out(i,:),z_diffraction);
    end
    [damage_diffraction_out,Uf_out] = fresnel1((damage_out),z_diffraction);
    
end

function boundary = compute_boundary(Eout,center,z_diffraction)
    global N1 m n;
%    I0 = gpuArray(mat2gray(abs(Eout).^2));
%     I0 = mat2gray (I0 - I_gaussian);
%     figure,imshow(I0);
%     I0 = imresize(I0,[N1,N1]);
%     I0 = uint8(I0.*255);
%     if(size>5 && z_diffraction<100)
%             BW = imbinarize(I0,'adaptive','Sensitivity' ,0.55);
%     elseif(size<=5)
%             BW = imbinarize(I0,'adaptive','Sensitivity' ,0.5);
%     else
%             BW = imbinarize(I0,'adaptive','Sensitivity' ,0.4);
%     end
%%    
%     BW = imbinarize(I0,'adaptive','Sensitivity',0.5);
%     CC = bwconncomp(BW,4);
%     stats = regionprops(CC,'BoundingBox','Area');
%     Area = [stats.Area];
%     [~,index]=max(Area);
%     boundary_b = {stats.BoundingBox};
%     boundary = round(cell2mat(boundary_b(index)));
  %  boundary = round(stats(index,:));
%%
    [M,N] = size(Eout);
    z_diffraction = round(z_diffraction*4);
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
    if (center(1)+z_diffraction/2)>N1
        boundary(3)=N1-boundary(1)-1;
    end
    if (center(2)+z_diffraction/2)>N1
        boundary(4)=N1-boundary(2)-1;
    end
    boundary = round(boundary);
end
