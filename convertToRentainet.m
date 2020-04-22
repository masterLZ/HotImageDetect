clear;
clc;
close all;
cd('D:\deeplearning - diffraction\code')

lib_diffraction = 'D:\deeplearning - diffraction\diffraction\Three_write';
lib_boundary = '..\table';
lib_imagewrite = '../csv/images';
lib_imagewriteABS = 'E:/Anacondadoc/keras-retinanet/tests/test-data/csv/images';
path_annotations = '..\csv\annotations.csv';
path_annotationsTest = '..\csv\val-annotations.csv';
path_classes = '..\csv\classes.csv';
Lmax = 800;%原尺寸
Lmin = 256;%缩放尺寸
%% 

fileFolder=fullfile(lib_diffraction);
dirOutput = dir(fullfile(fileFolder,'*.bmp'));
filenames = {dirOutput.name};
file_length = length(filenames);
tableWrite_name = cell(file_length*3,1);
boundaryWrite = zeros(file_length*3,4);

rng = 0;
index = sort(randperm(file_length,floor(file_length*0.8)));%分割训练集和测试集
n_train =1;
train_dataset_index = [];
k=1;
f = waitbar(0,'开始搬运');
% figure,
for  i = 1:file_length
    file = filenames{i};
    
    I = imread([lib_diffraction,'\',file]);
    file_head = strsplit(file,'.bmp');
    boundary_path = [lib_boundary,'\',file_head{1},'.csv'];
    boundary = round(csvread(boundary_path)./Lmax*Lmin);
    boundary1 = zeros(size(boundary));
    boundary1(:,1)=boundary(:,1);
    boundary1(:,2)=boundary(:,2);
    buffer=boundary(:,1)+boundary(:,3);
    buffer(buffer>Lmin-1)=Lmin-1;
    buffer(buffer<1)=1;
    boundary1(:,3)=buffer;    
    buffer=boundary(:,2)+boundary(:,4);
    buffer(buffer>Lmin-1)=Lmin-1;
    buffer(buffer<1)=1;
    boundary1(:,4)=buffer;    
    boundaryLength = size(boundary,1);
    k1 = k+boundaryLength;
    write_bmp_name = sprintf('%s/%06d.bmp',lib_imagewriteABS,i);
    write_bmp_name1 = sprintf('%s/%06d.bmp',lib_imagewrite,i);
    imwrite(I,write_bmp_name1);
    tableWrite_name(k:k1-1)=repmat({write_bmp_name},boundaryLength,1);
    boundaryWrite(k:k1-1,:)=boundary1;
    if i==index(n_train)
        train_dataset_index=[train_dataset_index,k:k1-1];
        n_train=n_train+1;
        if(n_train>floor(file_length*0.8))
            n_train = floor(file_length*0.8);
        end
    end
    k=k1;
    waitbar(i/file_length,f);
%     imshow(I)
%     for j = 1:boundaryLength
%          rectangle('Position',boundary(j,:), 'Edgecolor','r');
%     end
%     pause(0.01)
end
close(f)
%%
tableWrite_name=tableWrite_name(1:k-1);
boundaryWrite =num2cell(boundaryWrite(1:k-1,:)); 
%class_number = num2cell(zeros(size(tableWrite_name)));
class_name = repmat({'diffraction'},size(tableWrite_name));
%%
%划分测试和训练
tableWrite_nameTrain = tableWrite_name(train_dataset_index);
boundaryWriteTrain = boundaryWrite(train_dataset_index,:);
class_nameTrain = class_name(train_dataset_index);
test_dataset_index = 1:k-1;
test_dataset_index(train_dataset_index)=[];
tableWrite_nameTest = tableWrite_name(test_dataset_index);
boundaryWriteTest = boundaryWrite(test_dataset_index,:);
class_nameTest= class_name(test_dataset_index);
%%
%写测试集
A1 = [tableWrite_nameTrain,boundaryWriteTrain,class_nameTrain];
writecell(A1,path_annotations);
%写训练集
A2 = [tableWrite_nameTest,boundaryWriteTest,class_nameTest];
writecell(A2,path_annotationsTest);
