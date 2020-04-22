myClear;
global dx dy m n m1 n1;
dx = 6.45e-3;
dy =dx;
m = 1024;
n =m;
m1 = 800;
n1 = m1;

diffraction_init;


lib = 'D:\deeplearning - diffraction\target\diffraction\测试图';
[filenames,file_length] = eurmFiles(lib,'.tiff');

diffracitonZ = 10:1:400;
Z_length = length(diffracitonZ);
final_inv = zeros(Z_length,file_length);
final_invCopy = final_inv;
%%
for i = 1:file_length
    file = filenames{i};
    filename = strsplit(file,'.tiff');
    filename = filename{1};
    I_buffer = imreadOneChannel([lib,'\',file]);
    parfor j = 1:Z_length
        Uf = (prop(exp(1i*6*pi*I_buffer),dx,lambda,diffracitonZ(j)));
%         Uf = (fresnel_prog(exp(1i*6*pi*I_buffer),diffracitonZ(j)));
        If = abs(Uf).^2;
        final_inv(j,i)= max(max(If(500:600,500:600)));
%         imagesc(If);
%         title([num2str(diffracitonZ(j)),'mm    ',num2str(final_inv(i,j))])
%         pause(0.01)
    end
   
    final_invCopy(:,i) = final_inv(:,i);
%     final_inv(:,i) = PLDpreprocess((mapminmax(final_inv(:,i)',0,1))');
    final_inv(:,i) = mapminmax(PLDpreprocess(final_inv(:,i))',0,1);
    [max_I,index_I] = max(final_inv(:,i));
    fprintf('\n%s\t峰值是%f，峰值在%dmm',filename,max_I,diffracitonZ(index_I));
    figure,plot(diffracitonZ,final_inv(:,i));
    title(filename)
    
end
T = table(final_inv','RowNames',filenames);
writetable(T,'模式结果输出.csv','WriteRowNames',true)
%%
[max_I,index_I] = max(final_inv,[],1);
index = diffracitonZ(index_I);
