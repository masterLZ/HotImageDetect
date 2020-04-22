
global lambda k;
global fx fy;
global R C ;
%unit mm

lambda = 632.8*1e-6;                                                   %²¨³¤
k = 2*pi/lambda;                                                           %²¨Ê¸
% dx = 6.45e-3/800*1024;      
% dx = 5.5e-3;
% % dx = 6.45e-3;
% dy = dx;
% m = 2048;
% n = m;
r=-n/2:1:n/2-1;     c=-m/2:1:m/2-1;     [R,C] = meshgrid(r,c);
dfx=1/n/dx;     dfy=1/m/dy;     fx=gpuArray(R*dfx);       fy=gpuArray(C*dfy);
x = r*dx;       y = c*dx;       [X,Y] = meshgrid(x,y);
RR = sqrt(X.^2 + Y.^2);
