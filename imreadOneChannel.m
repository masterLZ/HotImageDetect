function [img] = imreadOneChannel(Path)
%��ȡ·����ͼ�񣬷��ص�ͨ��0-1double��������
%   �˴���ʾ��ϸ˵��
img=imread(Path);
if size(img,3)==1
    img = im2double(img);
else
    img = im2double(rgb2gray(img));
end
end

