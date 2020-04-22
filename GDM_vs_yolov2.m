clear;
close all;
lib = 'D:\deeplearning - diffraction\target\diffraction\table\';

criterionLength = 160;

for i = 0.1:0.1:1
    distanceTwoPoints = round(criterionLength*i);
    for j = 1:4      
        GDMpath = sprintf('%sGDM_%d_%d_6.csv',lib,distanceTwoPoints,j);
        Truethpath = sprintf('%s原图_%d_%d_6.csv',lib,distanceTwoPoints,j);
        yolov2Path = sprintf('%s测试_%d_%d_6.csv',lib,distanceTwoPoints,j);
        GDM = csvread(GDMpath);
        TruethCor = csvread(Truethpath);
        yolov2Cor = csvread(yolov2Path);
        Trueth = [TruethCor(:,1)+TruethCor(:,3)/2,TruethCor(:,2)+TruethCor(:,4)/2]./800;
        yolov2 = [yolov2Cor(:,1)+yolov2Cor(:,3)/2,yolov2Cor(:,2)+yolov2Cor(:,4)/2]./800;
        I = round(i*10);
        if size(Trueth,1)==size(GDM,1)
            GDMerr(I,j) = findDist(Trueth,GDM);
        else
            GDMerr(I,j) = findMinDist(Trueth,GDM);
        end
        if size(Trueth,1)==size(yolov2,1)
            yolov2err(I,j)= findDist(Trueth,yolov2);
        else
            yolov2err (I,j)= findMinDist(Trueth,yolov2);
        end 
    end
end

%%
function errout = findMinDist(Trueth,error)
%适用与点的个数不匹配的时候计算最小的误差
    errout1 = inf;
    errout2 = inf;
        for j = 1:size(error,1)
            error1 = norm(Trueth(1,:)-error(j,:));
            error2 = norm(Trueth(2,:)-error(j,:));
            if(error1<errout1)
                errout1 = error1;
            end
            if(error2<errout2)
                errout2 = error2;
            end
        end
    errout = (errout1+errout2)/2+abs(size(error,1)-2)*0.2;
end
function Error = findDist(Trueth,error)
%适用与调整点与点之间的位置
    error1 = norm(Trueth(1,:)-error(1,:));
    error2 = norm(Trueth(2,:)-error(1,:));
    if(error1<error2)
        error2 = error1;
    end
    error3 = norm(Trueth(1,:)-error(2,:));
    error4 = norm(Trueth(2,:)-error(2,:));
    if(error3<error4)
        error4 = error3;
    end
    Error = mean([error2,error4]);
end
