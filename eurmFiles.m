function [filenames,file_length] = eurmFiles(Path,type)
%�оٸ��ļ�����ָ�����������ļ�
%type = '.csv','.bmp'
fileFolder=fullfile(Path);
dirOutput = dir(fullfile(fileFolder,['*',type]));
filenames = {dirOutput.name};
file_length = length(filenames);
end

