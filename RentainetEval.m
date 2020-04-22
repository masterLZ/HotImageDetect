clear;clc;
%   load ('D:\deeplearning - diffraction\code\fastrcnn.mat')
T1 = readtable('D:\deeplearning - diffraction\predict-annotations.csv');
T2 = readtable('D:\deeplearning - diffraction\val-annotations.csv');
T2.Var4 = T2.Var4-T2.Var2-1;
T2.Var5 = T2.Var5-T2.Var3-1;
[Tout] = tableToEval(T1);
%%
Tout2 = tableToEval(T2);
expectTout = cell2table(Tout2.Boxes,'VariableNames',{'boundary_cell'});
[ap, recall, precision] = evaluateDetectionPrecision(Tout, expectTout);
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
writeTable = table(recall,precision);
writetable(writeTable,'D:\deeplearning - diffraction\val_out.csv')
%%
function [Tout]=tableToEval(T1)
    boundaryLib = table2array(T1(:,2:5));
    namesLib = T1.Var1;
    [unique_names,ia,~] = unique(namesLib,'stable');
    scoresLib =  T1.Var6;
    dia = diff(ia);
    L = length(unique_names);
    dia(L)=0;
    boundary_cell = cell(L,1);
    Scores = cell(L,1);
    Names = cell(L,1);
%     lables = Names;
    for i=1:L
        boundary_cell{i}=boundaryLib(ia(i):ia(i)+dia(i)-1,:);
        Scores{i}=scoresLib(ia(i):ia(i)+dia(i)-1);
       % Names{i}=repmat('boundary_cell',dia(i),1);
%         Names{i}=namesLib(ia(i));
%         lables{i}=T1.Var8(ia(i));
    end

    numImages = length(boundary_cell);
    Tout = table('Size',[numImages 2],...
        'VariableTypes',{'cell','cell'},...
        'VariableNames',{'Boxes','Scores'});
%     Tout1 = table('Size',[numImages 2],...
%         'VariableTypes',{'cell','cell'},...
%         'VariableNames',{'names','lables'});
    for i = 1:numImages
    %     results.Boxes{i} = mat2cell(boundary_cell{i},ones(size(boundary_cell{i},1),1));
        Tout.Boxes{i} = boundary_cell{i};
        Tout.Scores{i} = Scores{i};
%         Tout1.names{i} = Names{i};
%         Tout1.lables{i} = lables{i};
%         Tout.Labels{i} = categorical(cellstr(Names{i}));
    end
end
