clear;
pic_path ='D:\deeplearning - diffraction\diffraction';
write_lib = 'D:\deeplearning - diffraction\diffraction\Three';
pic_dicOutput = dir(fullfile(pic_path,'*.bmp'));
imageFilename = cell(size(pic_dicOutput));
f = waitbar(0,'¿ªÊ¼°áÔË');
for i = 1:length(pic_dicOutput)
    pic_fullPath = fullfile(pic_dicOutput(i).folder,pic_dicOutput(i).name);
    I = imread(pic_fullPath);
    write_path = fullfile(write_lib,pic_dicOutput(i).name);
    imwrite(cat(3,I,I,I),write_path);
    waitbar(i/length(pic_dicOutput),f);
end
close(f)
