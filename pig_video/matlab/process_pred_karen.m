%% import prediction file

% 
% M=240;
% N=320;

M=1648;
N=2208;



%% read video framesl
path_name = 'z:\_pig_video\'
filename_list=dir('z:\_pig_video\*.avi');
%filename_list=dir('*.avi');
my_filename=filename_list(1).name;
disp(my_filename);

% Construct a multimedia reader object associated with file 'xylophone.mpg' with
% user tag set to 'myreader1'.
readerobj = VideoReader(my_filename);
% Read in all video frames.
vidFrames = read(readerobj);

% Get the number of frames.
numFrames = get(readerobj, 'NumberOfFrames');

range_mat=range(single(vidFrames(:,:,1,:)),4);


%% process pred mat
load range_mat_3;


figure(1);

fn = './vid1/pred_test_rf.csv';
pred = csvread(fn,1);
pred_mat = mat2gray(reshape(pred(:,1),M, N));
pred_mat_bi = (pred_mat<.77);
%subplot(2,2,1);imagesc(imcomplement(pred_mat)); title('I');
subplot(2,2,1);imagesc(pred_mat); title('I');


I = pred_mat;
BW = im2bw(I, graythresh(I));
[B,L] = bwboundaries(BW,'noholes');
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
end



%% blood vessel detection

I = pred_mat;
J=mat2gray(range_mat);
imagesc(mat2gray(range_mat));


alpha=0.8;
M = I*alpha+J*(1-alpha);
imagesc(mat2gray(M));

BW = im2bw(M, graythresh(M));
[B,L] = bwboundaries(BW,'noholes');
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
end
title(strcat('alpha=',num2str(alpha)));

stat = regionprops(L, 'Area', 'Eccentricity');


%filter out small region

thresh =10000;
for i = 1:length(B)
    disp(i);
    count = sum(L(:)==i);
    if count < thresh
        [row,col] = find(L ==i);
        L(row, col)=0;        
    end
end
imshow(label2rgb(L, @jet, [.5 .5 .5]))

%% skeleton detection
BW1 = L;
BW2 = bwmorph(BW1,'remove');
BW3 = bwmorph(BW1,'skel', 100);
BW3 = bwmorph(BW3,'clean');
BW3 = bwmorph(BW3, 'spur');
imshow(BW3)




%%
I2 = L;

%# Perform morphological thinning to get the skeleton
I3 = bwmorph(I2, 'thin',Inf);

%# prune the skeleton (remove small branches at the endpoints)
I4 = bwmorph(I3, 'spur', 2);

%# Remove small components
I5 = bwareaopen(I4, 30);

%# dilate image
I6 = imdilate(I5, strel('square',2*2+1));

I7 = bwmorph(I6, 'thin',Inf);
IM=I7;

save('skeleton_mat', 'IM');

% %%
% mn=bwmorph(y,'branchpoints');
% [row column] = find(mn);
% branchPts    = [row column];
% endImg    = bwmorph(y, 'endpoints');
% [row column] = find(endImg);
% endPts       = [row column];
% figure;imshow(y);
% hold on ; 
% plot(branchPts(:,2),branchPts(:,1),'rx');
% hold on; plot(endPts(:,2),endPts(:,1),'*');
% 
% 
% 
% 
% %%  define a bounding box
% IM = I7;
% imshow(IM);
% [x1, y1] = ginput(1);[x2, y2] = ginput(1);
% w=abs(x2-x1);h=abs(y1-y2);
% rect=floor([x1 y1 w h]);
% hold on;
% rectangle('Position',rect, 'LineWidth',5, 'EdgeColor','r');
% 
% %% find ending points that are left and right of the points closes to the bounding box
% 
% % extract on pixel index from bounding box
% x1=floor(x1); x2=floor(x1+w);
% y1=floor(y1); y2=floor(y1+h);
% 
% M=1648;
% N=2208;
% 
% box_mat=repmat(0,M,N); 
% box_mat(y1:y2, x1:x2)=1; % rectangle box mask marix
% 
% IM2 = IM.*box_mat;
% %imshow(IM2);
% 
% [ii, jj]=ind2sub(size(IM2),find(IM2==1));
% px1=min(jj);px2=max(jj);
% 
% tmp=find(jj==min(jj));
% py1=ii(tmp(1));
% tmp=find(jj==max(jj));
% py2=ii(tmp(1));
% 
% hold on;
% plot(px1,py1,'ro',px2,py2,'ro');
% 

%% extract ts at 3*3 box







%sample data points on veins center line within a region of interest




%%
%training on video 1 test on video2

fn = './video2/vid1/pred_test_rf.csv';
pred = csvread(fn,1);
pred_mat = mat2gray(reshape(pred(:,1),M, N));
pred_mat_bi = (pred_mat<.77);
%subplot(2,2,2);imagesc(imcomplement(pred_mat)); title('I/max(I)');
subplot(2,2,1);imagesc(pred_mat); title('raw');



fn = './video2/vid2/pred_test_rf.csv';
pred = csvread(fn,1);
pred_mat = mat2gray(reshape(pred(:,1),M, N));
pred_mat_bi = (pred_mat<.77);
%subplot(2,2,2);imagesc(imcomplement(pred_mat)); title('I/max(I)');
subplot(2,2,2);imagesc(pred_mat); title('I/max(I)');



fn = './video2/vid3/pred_test_rf.csv';
pred = csvread(fn,1);
pred_mat = mat2gray(reshape(pred(:,1),M, N));
pred_mat = reshape(pred(:,1),M, N);
pred_mat_bi = (pred_mat<.5);
%subplot(2,2,3);imagesc(imcomplement(pred_mat)); title('I/max(I)-mean');
subplot(2,2,3);imagesc(pred_mat); title('I/max(I)-mean')


%subplot(2,2,4); imagesc(imcomplement(range_mat)); title('range mat');
subplot(2,2,4); imagesc(range_mat); title('range mat');



%training on video 1 and test on video 1
fn = './vid1/pred_test_rf.csv';
pred = csvread(fn,1);
pred_mat = mat2gray(reshape(pred(:,1),M, N));
pred_mat_bi = (pred_mat<.77);
%subplot(2,2,2);imagesc(imcomplement(pred_mat)); title('I/max(I)');
subplot(2,2,1);imagesc(pred_mat); title('raw');



fn = './vid2/pred_test_rf.csv';
pred = csvread(fn,1);
pred_mat = mat2gray(reshape(pred(:,1),M, N));
pred_mat_bi = (pred_mat<.77);
%subplot(2,2,2);imagesc(imcomplement(pred_mat)); title('I/max(I)');
subplot(2,2,2);imagesc(pred_mat); title('I/max(I)');



fn = './vid3/pred_test_rf.csv';
pred = csvread(fn,1);
pred_mat = mat2gray(reshape(pred(:,1),M, N));
pred_mat = reshape(pred(:,1),M, N);
pred_mat_bi = (pred_mat<.5);
%subplot(2,2,3);imagesc(imcomplement(pred_mat)); title('I/max(I)-mean');
subplot(2,2,3);imagesc(pred_mat); title('I/max(I)-mean')


%subplot(2,2,4); imagesc(imcomplement(range_mat)); title('range mat');
subplot(2,2,4); imagesc(range_mat); title('range mat');












figure(1);
subplot(2,2,1); imshow(vidFrames(:,:,1,1)); colormap('gray'); title('raw first frame');
subplot(2,2,2); imshow(pred_mat); colormap('jet'), title('prediction');
subplot(2,2,3); imshow(mat2gray(range_mat)); title('range map');
subplot(2,2,4); imshow(pred_mat_bi); title('binarized prediction map');




fn='./vid1/data_classifer_test.csv';
feature = csv2struct(fn);

f_mean = mat2gray(reshape(feature(1).mean_5, M, N));
f_range= mat2gray(reshape(feature(1).range_5, M, N));

raw1 = mat2gray(vidFrames(:,:,1,1));

figure(1);
subplot(2,2,1); imshow(raw1); title('raw first image');
subplot(2,2,2); imshow(f_mean); title('feature: mean scale 5');
subplot(2,2,3); imshow(f_range); title('feature: range scale 5');
subplot(2,2,4); imshow(pred_mat); title('prediction');

colormap('hsv');