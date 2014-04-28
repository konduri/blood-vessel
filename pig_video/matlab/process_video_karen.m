%% load video
% 
% path_name = '..\_pig_video\'
% filename_list=dir('..\_pig_video\*.avi');
% %filename_list=dir('*.avi');
% my_filename=filename_list(1).name;
% disp(my_filename);

% Construct a multimedia reader object associated with file 'xylophone.mpg' with
% user tag set to 'myreader1'.

readerobj = VideoReader('../upmc-ss_pigs-pig_ss15-20131118-105514113.avi');
% readerobj = VideoReader(my_filename);
% Read in all video frames.
vidFrames = read(readerobj);
% Get the number of frames.
numFrames = get(readerobj, 'NumberOfFrames');
%% read in a small video from matlab
%  readerobj = VideoReader('xylophone.mpg', 'tag', 'myreader1');
%  vidFrames = read(readerobj);
%  numFrames = get(readerobj, 'NumberOfFrames'); 
% %%  play video
%%
% Create a MATLAB movie struct from the video frames.
for k = 1 : numFrames
     mov(k).cdata = vidFrames(:,:,:,k);
     mov(k).colormap = [];
end

% Create a figure
hf = figure; 

% Resize figure based on the video's width and height
set(hf, 'position', [150 150 readerobj.Width readerobj.Height])

% Playback movie once at the video's frame rate
movie(hf, mov, 1, readerobj.FrameRate);

%% plot time series of value at location x,y

x=1;y=1;
ts = reshape(vidFrames(x,y,1,:),1,numFrames);
plot(ts)



%% get a small portion of the video

smallVideo=vidFrames(1:100,1:100,1,:);
movie(hf,smallVideo,1,readerobj.FrameRate);

%% create different version of frames

M = size(vidFrames,1);
N = size(vidFrames,2);
numFrames = size(vidFrames,4);

vidFrames1 = zeros(M,N,numFrames);
vidFrames2 = zeros(M,N,numFrames);
vidFrames3 = zeros(M,N,numFrames);
%for k = 1:5
for k = 1 : numFrames
     disp(k);
     this_frame = double(vidFrames(:,:,1,k));
     vidFrames1(:,:,k) = this_frame;
     %vidFrames2(:,:,k) = this_frame/max(this_frame(:));
     %vidFrames3(:,:,k) = vidFrames2(:,:,k)-mean2(this_frame);     
end




 
%% compute mean frames at various scale and compute faeture thereafter

vid = vidFrames1;
scale = 5;
M = size(vid,1);
N = size(vid,2);
numFrames = size(vid,3);

mean_frame = zeros(M,N,numFrames);

for k = 1 : numFrames
     disp(k);
     mean_frame(:,:,k) = meanFilter(vid(:,:,k),scale); 
     
end

stat_mean = mean(mean_frame,3);
stat_range = range(mean_frame,3);
stat_var = var(mean_frame,0,3);

stat_mat = reshape(cat(3, stat_mean, stat_var, stat_range), M*N, 3);


path_name = 'Z:\_pig_video\feature\';

out = struct('mean',stat_mean(:), 'var', stat_var(:), 'range', stat_range(:));
fn = strcat(path_name, 'feature_scale', num2str(scale),'.csv');
struct2csv(out, fn);
 



%% normalize to elimiate illumiation effect against max
figure(1);
this_frame = vidFrames(:,:,1,1);
subplot(2,2,1);
imshow(this_frame);
title('original');
this_frame = double(this_frame);
norm_frame1 = mat2gray(this_frame/max(this_frame(:)));

subplot(2,2,3);
hist(this_frame(:));

subplot(2,2,2);
imshow(norm_frame1);
title('normalized by max');

subplot(2,2,4);
hist(norm_frame1(:));

%% normalize to elimiate illumiation effect against max substracted mean
figure(1);
this_frame = vidFrames(:,:,1,1);
subplot(2,2,1);
imshow(this_frame);
title('original');

% histogram of original image
subplot(2,2,3);
this_frame = double(this_frame);
hist(this_frame(:));


%norm_frame = uint8(floor(this_frame/max(max(this_frame))*255));
norm_frame1 = this_frame/max(this_frame(:));
norm_frame2 = norm_frame1 - mean(norm_frame1(:));

subplot(2,2,4);
norm_frame2 = mat2gray(norm_frame2);
hist(norm_frame2(:));

%histogram of transformed image
subplot(2,2,2);
imshow(norm_frame2);
title('normalized by max, subtracted by mean');



%% normalize each frame

vidFrames2 = vidFrames;
%for k = 1:5
for k = 1 : numFrames
     disp(k);
     this_frame = double(vidFrames(:,:,1,k));
     norm_frame1 = uint8(mat2gray(this_frame/max(this_frame(:)))*255);     
     vidFrames2(:,:,1,k) = norm_frame1;     
end


%% extract global feature frame by frame and save it in a matrix

nn = numFrames;
gM = zeros(nn, 1);
gV = zeros(nn, 1);
gE = zeros(nn, 1);

for k = 1: nn
    disp(k);
    this_frame = double(vidFrames(:,:,1,k));
    [M, V, E] = get_global_feature(this_frame); 
    gM(k) = M;
    gV(k) = V;
    gE(k) = E;    
end
    
  
 out = struct('mean',gM, 'var', gV, 'entropy', gE);
 fn = strcat(path_name, 'global_feature.csv');
 struct2csv(out, fn);

%% image filtering
scale = 100;
I_raw = double(vidFrames(:,:,1,1));
I = I_raw/max(I_raw(:));


J = meanFilter(I,9);
K = J - meanFilter(J,100);
L = I - meanFilter(I,100);

figure(1);
subplot(2,2,1);
imshow(mat2gray(I));
title('I=original normalized by max ');

subplot(2,2,2);
imshow(mat2gray(J));
title('J=smoothed - mean(I,9)');

subplot(2,2,3);
imshow(mat2gray(L));
title('I-mean(I,100)');

subplot(2,2,4);

imshow(mat2gray(K));
title ('J- mean(J,100)');

C = histeq(I);
imshow(I);

S = imsharpen(I,'Radius',50,'Amount',0.8);
imshow(mat2gray(S));
% 
imhist(K);

K2 = K>0.01;
imshow(K2);
thresh = [0.01 0.2];
BWs=edge(J,'canny',thresh);
imshow(BWs);


%% compute a map of various statistcs at each pixel
var_mat=var(single(vidFrames(:,:,1,:)),0,4);
range_mat=range(single(vidFrames(:,:,1,:)),4);
range_mat2=range(single(vidFrames2(:,:,1,:)),4);

mean_mat=mean(single(vidFrames(:,:,1,:)),4);
ratio_mat=range_mat/mean_mat;
imwrite(uint8(range_mat), 'range_mean_ratio.jpg', 'jpg');


%save the cropped video
% aviObj = VideoWriter('cropped_video.avi','Uncompressed AVI');
% open(aviObj);
% writeVideo(aviObj,smallVideo);
% close(aviObj);


%% run edge detection on range_mat
thresh = [0.15 0.2];
BW=edge(range_mat,'canny',thresh);
imshow(BW)

%%
% function to compute feature at a region centered at point [x y]

figure(1)
imagesc(range_mat);
[x0 y0]=ginput(1);
hold on;
plot(x0,y0,'ro')
hold off;
disp([x0 y0]);

%x0=800;y0=800;
regSize=10;

xrange=1:size(vidFrames,1);
yrange=1:size(vidFrames,2);

xidx = xrange (xrange>=x0-regSize/2 & xrange<=x0+regSize/2);
yidx = yrange (yrange>=y0-regSize/2 & yrange<=y0+regSize/2);

sub_region = vidFrames(xidx,yidx,1,:);
%sub_region = vidFrames(:, :,1,:);

%xlim*ylim by t
mat_t = reshape(single(sub_region),size(sub_region, 1)*size(sub_region,2),numFrames);  
mat_g = reshape(single(sub_region),size(sub_region, 1)*size(sub_region,2)*numFrames,1);
var_g = var(mat_g);
var_t = var(mat_t);
mean_g = mean(mat_g);
mean_t = mean(mat_t);


figure(2)
subplot(2,2,1); 
plot(mean_t, 'g');hold on; plot(smooth(mean_t,20), 'r');
ylim([90 200]);
title ('mean');
hold off;
subplot(2,2,2); plot(var_t, 'g'); hold on; plot(smooth(var_t,20), 'r');
ylim([20 350]);
title ('variance');
hold off;

x = mean_t;
m = length(x);
y = fft(x,m);
f = (0:(m-1))/m;
power = y.*conj(y)/m; 
subplot(2,2,3); 
semilogy(f, power);
title ('mean ts fft');
%plot(f, power);

%xlim([0, 0.01]);

x = var_t;
m = length(x);
y = fft(x,m);
f = (0:(m-1))/m;
power = y.*conj(y)/m; 
subplot(2,2,4); 
semilogy(f, power);
title ('var ts fft');
%plot(f, power); 
%xlim([0, 0.01]);



%% use function get_feature
figure(1)
%imshow(vidFrames(:,:,1,1));
load range_mat_3;
imagesc(imcomplement(range_mat));
colormap 'gray'

hold on;

%pType=0; % 1 positive 0 negative

NN = 30; % number pairs of points
xy_mat_0 = zeros(NN,5);
xy_mat_1 = zeros(NN,5);
%point(NN) = struct('x0',[], 'y0',[], 'value',[], 'label',[]);

for pType=[1 0]
    for ii=1:NN
        [x0, y0]=ginput(1);

        if pType==1
            h =plot(x0,y0,'ro');
            x0=floor(x0); y0=floor(y0);
            linearInd = double(sub2ind(size(range_mat), y0, x0));
            value = range_mat(y0,x0);
            xy_mat_1(ii,:)=[y0 x0  linearInd 1 value];            
        else 
            h= plot(x0,y0,'bo');
            x0=floor(x0); y0=floor(y0);
            linearInd = double(sub2ind(size(range_mat), y0, x0));
            value = range_mat(y0,x0);
            xy_mat_0(ii,:)=[y0 x0 linearInd 0 value];
        end   
    end 
end
hold off;

xy_mat = cat(1, xy_mat_0, xy_mat_1);



mt_mat=zeros(size(vidFrames,4),NN);
vt_mat=zeros(size(vidFrames,4),NN);
mg_vec=zeros(NN,1);
vg_vec=zeros(NN,1);
pmax_vec=zeros(NN,1);
rg_vec=zeros(NN,1);


regsize_list=[5 10 20 50 100];
    
for kk = 1:length(regsize_list)
    regSize = regsize_list(kk);
    for ii = 1:NN
        x0=xy_mat(ii,1);y0=xy_mat(ii,2);
        [mt,mg,vt,vg,rg,pmax]= get_feature(vidFrames,x0,y0,regSize);
        mt_mat(:,ii)=mt;
        vt_mat(:,ii)=vt;   
        mg_vec(ii)=mg;
        vg_vec(ii)=vg;
        pmax_vec(ii)=pmax;
        rg_vec(ii)=rg;
    end
    out = struct('mean',mg_vec, 'var', vg_vec, 'pmax', pmax_vec, 'range', rg_vec);
    fn = strcat(path_name, 'feature_scale', num2str(regSize),'.csv');
    struct2csv(out, fn);
    
    ts_fn=strcat(path_name, 'meanTS_scale', num2str(regSize), '.csv');
    csvwrite(ts_fn, mt_mat);
end




%% compute features for a large region without user input

[x,y]=ginput(2);
minx = min(x);
maxx= max(x);
miny= min(y);
maxy = max(y);

xrange=floor(minx):1:floor(maxx);
yrange=floor(miny):1:floor(maxy);

xy_mat=zeros(length(xrange)*length(yrange),2);

count=0;
for ii=xrange
    for jj=yrange 
        count=count+1;
        xy_mat(count,:)=[ii,jj];
    end     
end

NN = size(xy_mat,1);

save 'xy_mat' xy_mat;

mg_vec=zeros(NN,1);
vg_vec=zeros(NN,1);
pmax_vec=zeros(NN,1);
rg_vec=zeros(NN,1);


regsize_list=[5 10 20 50 100];

path_name = 'Z:\_pig_video\ROIfeature\';
    
for kk = 1:length(regsize_list)
    regSize = regsize_list(kk);
    for ii = 1:NN
        x0=xy_mat(ii,1);y0=xy_mat(ii,2);
        [mt,mg,vt,vg,rg,pmax]= get_feature(vidFrames,x0,y0,regSize);
        mt_mat(:,ii)=mt;
        vt_mat(:,ii)=vt;   
        mg_vec(ii)=mg;
        vg_vec(ii)=vg;
        pmax_vec(ii)=pmax;
        rg_vec(ii)=rg;
    end
    out = struct('mean',mg_vec, 'var', vg_vec, 'pmax', pmax_vec, 'range', rg_vec);
    fn = strcat(path_name, 'feature_scale', num2str(regSize),'.csv');
    struct2csv(out, fn);
    
%     ts_fn=strcat(path_name, 'meanTS_scale', num2str(regSize), '.csv');
%     csvwrite(ts_fn, mt_mat);
end
