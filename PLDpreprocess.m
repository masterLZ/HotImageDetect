function y = PLDpreprocess(x)
%  Preprocess input x
%    This function expects an input vector x.

% Generated by MATLAB(R) 9.7 and Signal Processing Toolbox 8.3.
% Generated on: 18-Dec-2019 15:46:02

windowLength = 14;
[y,~] = envelope(x,windowLength,'rms');