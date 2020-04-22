function Uf = fresnel_prog(U_initial,z_diffraction)
% compute fresnel diffraction
   % tic
   % m1 n1 是图像大小
   %m n是计算区域 
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
    Uf = Uf(1:m1,1:n1);
   % toc
end
