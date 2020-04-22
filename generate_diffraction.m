
clear;
clc;
close all;
lib_origin = '.\origin\';
lib_diffraction = '.\diffraction\';
lib_table = '.\table\';
lib_buffer = '.\buffer\';
cd 'D:\deeplearning - diffraction'
global isJiaoDie;
isJiaoDie =0;
if isJiaoDie
   lib_origin = '.\origin\jiaocha\';
    lib_diffraction = '.\diffraction\jiaocha\';
    lib_table = '.\table\jiaocha\';
    lib_buffer = '.\buffer\jiaocha\';
end
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
N_gaussian = 1;
w0 = dx*m;
U_gaussian = exp(-(sqrt(X.^2+Y.^2)/w0).^N_gaussian);
%%
global N1;
N1 = 800;
%初始损伤点
figure,
 for diffraction_distance =30:40:260
     fprintf('\n%d',diffraction_distance);
     E0 = zeros(m,n);
     Eout = E0;
     boundary_diffraction = zeros(1,4);
     numi = 0;
     for j = 1:500
        [E0,Eout,boundary_diffraction, numi] = gengerate_damage(5,15,diffraction_distance);
        str = [lib_origin,num2str(diffraction_distance),'_',num2str(j),'.bmp'];
        imwrite(imresize(abs(E0),[256,256]),str);
        str = [lib_diffraction,num2str(diffraction_distance),'_',num2str(j),'.bmp'];
        diffraction_save = (imresize(abs(Eout).^2,[256,256]));
        diffraction_save = diffraction_save./max((max(diffraction_save)));        
        imwrite(cat(3,diffraction_save,diffraction_save,diffraction_save),str);
        str = [lib_table,num2str(diffraction_distance),'_',num2str(j),'.csv'];
        csvwrite(str,boundary_diffraction);
%         imshow(diffraction_save)
%         for i = 1:numi
%             rectangle('Position',boundary_diffraction(i,:), 'Edgecolor','r');
%         end
%         pause(0.01)
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
% I_buffer = mat2gray(abs(Eout).^2).*2*pi;
% 
% I_max = abs(fresnel1(exp(1i*I_buffer),10)).^2;
% figure,imagesc(I_max)
%%

%%
function [Uf,Uf_boundary] = fresnel1(Phase_in,z_diffraction)
% compute fresnel diffraction
   % Uf 计算出射物光+背景
   % Uf_boundary 用于计算边界
    global xx yy k m n U_gaussian;
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
   % toc
end
function center_out = generate_damgepoint(num,z_diffraction)
    global m n isJiaoDie;
    center_out = zeros(num, 2);
%     point_1D = randi(m*n, num, 1);
%     [center_out(:,1), center_out(:,2)] = ind2sub([m,n], point_1D);
    center_out(:,1) = randi(round(m/4), num, 1)-m/4+m/8;
    center_out(:,2) = randi(round(n/4), num, 1)-m/4+m/8;
   if z_diffraction>100
       z_diffraction=100;
   end
    if isJiaoDie
        for i=2:num 
            while 1
                center_out(i,1)=(randi([z_diffraction,z_diffraction+30])*randsrc)+center_out(i-1,1);
                center_out(i,2)=(randi([z_diffraction,z_diffraction+30])*randsrc)+center_out(i-1,2);
                if center_out(i,1)>(-m/8) && center_out(i,2)>(-m/8) && center_out(i,1)<(m/8) && center_out(i,2)<(m/8)
                    break;
                end                
            end
        end
    end
end

function [damage_out,damage_diffraction_out, boundary_diffraction, num] = gengerate_damage(num1, size, z_diffraction)
    global R C m n;
    R1 = R(m/4:m/4+m/2-1,n/4:n/4+n/2-1);
    C1 = C(m/4:m/4+m/2-1,n/4:n/4+n/2-1);
    damage_out = zeros(m/2, n/2);
    num = randi(num1);
    center_out = generate_damgepoint(num,z_diffraction);
    size_out = randi([3,size],num,1);
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
       z_diffraction = round(z_diffraction*3.5);
   else
       z_diffraction = round(z_diffraction*1.5);
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

