function [img] = imreadOneChannel(Path)
%读取路径下图像，返回单通道0-1double类型数据
%   此处显示详细说明
img=imread(Path);
if size(img,3)==1
    img = im2double(img);
else
    img = im2double(rgb2gray(img));
end
end

