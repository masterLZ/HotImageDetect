%%
%²âÊÔÖØµþÇøÓòyolov2ÐÔÄÜ
myClear;
lab_read = 'C:\Users\hp\Desktop\ËðÉËµã¼ì²â\Í¼Æ¬\¶à¸öµã\';
lab_write = 'C:\Users\hp\Desktop\ËðÉËµã¼ì²â\Í¼Æ¬\¶à¸öµã\¼ì²âÊä³ö\';
lab_back = 'C:\Users\hp\Desktop\ËðÉËµã¼ì²â\Í¼Æ¬\¶à¸öµã\È¥±³¾°Í¼\';
s0 = sprintf('Image%03d.bmp',10);
path = [lab_read,s0];
I_background = imreadOnechannel(path);
I_background = I_background(10:1010,300:1300);
load yolov2detector3
%%

for i =0:9
    fprintf('\n%d/9',i);
    s0 = sprintf('Image%03d.bmp',i);
    path = [lab_read,s0];
    I0 = imreadOnechannel(path);
    I0 = I0(10:1010,300:1300);
    I01= mat2gray(I0-I_background);
    I0 = imresize(I01,[256,256]);
    I = I0;
    [u,s,v]=svd(I0);
    sval_nums = 2;
    low = 1;
    u1 =zeros(size(u));
    s1=u1;v1 =u1;
    u1(:,low:sval_nums) =u(:,low:sval_nums);
    s1(:,low:sval_nums) =s(:,low:sval_nums);
    v1(:,low:sval_nums) =v(:,low:sval_nums);
    I1 = u1*s1*v1';
    I1 = imgaussfilt(I1,20);
    I0 = (mat2gray(I./I1));
    I = I0;
    I = imgaussfilt(I,1);
    %
    H = (fspecial('sobel'));
    dx1 = (imfilter(I,H,'circular','same'));
    dy1 = (imfilter(I,H','circular','same'));
    arrow = im2uint8 (mat2gray(atan2(dx1,dy1)));
    arrow = cat(3,arrow,arrow,arrow);
    I1 = imreadOnechannel(path);
    I1=I1(10:1010,300:1300);
    [bboxes, scores, labels] = detect(detector, arrow);
    bboxes = bboxes(scores>0.5,:)./255.*size(I1,1);
    scores = scores(scores>0.5);
    I1=insertObjectAnnotation(I01,'rectangle',bboxes,scores, 'LineWidth',4,'Font','Times New Roman', 'FontSize',30);
%     center = [(bboxes(:,1)+bboxes(:,3)/2),(bboxes(:,2)+bboxes(:,4)/2)];
%     I1 = insertMarker(I1,center,'x','Size' ,8,'color','red');
%     I1 = insertMarker(I1,center,'+','Size' ,8,'color','red');
    s0 = sprintf('yolov2_Image%03d.bmp',i);
    path = [lab_write,s0];
    imwrite(I1,path);
    path = [lab_back,s0];
    imwrite(I01,path); 
end

