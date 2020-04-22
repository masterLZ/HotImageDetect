function [filenames,file_length] = eurmFiles(Path,type)
%列举该文件夹下指定类型所有文件
%type = '.csv','.bmp'
fileFolder=fullfile(Path);
dirOutput = dir(fullfile(fileFolder,['*',type]));
filenames = {dirOutput.name};
file_length = length(filenames);
end

