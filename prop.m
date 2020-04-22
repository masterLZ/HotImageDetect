function [H] = prop(im,pSize,wLength,z)
% propagate 'im' for z distance

im = gpuArray(im);
[M,N]=size(im);
k0 = gpuArray(2*pi/wLength);
kmax = pi / pSize;
kxm0 = gpuArray(linspace(-kmax,kmax,M));
kym0 = gpuArray(linspace(-kmax,kmax,N));
[kxm,kym] = (meshgrid(kxm0,kym0));
kzm = double(sqrt(k0^2-kxm.^2-kym.^2));
imFT = fftshift(fft2(im));
PropFunction = exp(1i.*kzm.*z).*((k0^2-kxm.^2-kym.^2)>0);
OutputFT = imFT.*PropFunction;
H = ifft2(ifftshift(OutputFT));
H = gather(H);

end

