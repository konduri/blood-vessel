movie_obj = VideoReader('upmc-ss_pigs-pig_ss15-20131118-105514113.avi');
nFrames   = movie_obj.NumberOfFrames;
%%
image1 = read(movie_obj,1);
image1 = im2double(image1(:,:,1)); %all data bw 0 & 1
hgram  = imhist(im2double(image1(:,:,1)));
%%

figure

for i = 1:nFrames
    buffer_temp = read(movie_obj, i);
    image_orig       = im2double(buffer_temp(:,:,1));
    image_hist  = histeq(image_orig,hgram);
    image_mean  = (image_orig-mean(image_orig(:))) ./ sqrt(var(image_orig(:)));
    pause(0.01);
    subplot(1,3,1)
    imshow(image_orig)
    subplot(1,3,2)
    imagesc(image_hist)
    subplot(1,3,3)
    imagesc(image_mean)
    
end

%%
train_img = image1(936:1638,1:1100);
imshow(train_img)
%%
temp =[]
points = []

%%
%%
clear temp
close
h = figure
imshow(train_img);
[x,y]=getpts(h)
temp = [x,y];
points = [points;temp];

%%
hold on
plot(points(:,1),points(:,2),'*')
%%



figure
imshow(train_img)




%%
figure
temp =  zeros(size(train_img));
points2 = abs(round(points));

for i = 1:length(points)
    temp(points2(i,2),points2(i,1))= 1;
end
imshow(temp)

pause(1)

se = strel('line',10,10);
I2 = imdilate(temp,se);
imshow(I2)

%%

imwrite(I2,'train_ans_small.png')
imwrite(train_img,'train_image_small.png')

%%

movie_obj = VideoReader('upmc-ss_pigs-pig_ss15-20131118-105514113.avi');
nFrames   = movie_obj.NumberOfFrames;
%%
image1 = read(movie_obj,1);
image1 = im2double(image1(:,:,1)); %all data bw 0 & 1
hgram  = imhist(im2double(image1(:,:,1)));


%%
% 
% outputVideo = VideoWriter('sample_two.avi');
% outputVideo.FrameRate = 8;
% open(outputVideo);
dummy = read(movie_obj,n);
dummy = im2double(dummy(:,:,1));
testingImg = dummy(936:1638,1:1100);
    



%%
scale = 0.5; 
mean_seg = zeros(352,550);
figure
for n = 1:150
    image2 = read(movie_obj,n);
    image2 = im2double(image2(:,:,1));
    image2  = histeq(image2,hgram);

    
    testingImg = image2(936:1638,1:1100);
    testingImg  = imresize(testingImg ,scale,'nearest');

    
    
    
    
    features = zeros([size(testingImg),numel(sigmas),numel(thetas),numel(Fs)]);
for k = 1:numel(sigmas)
for j = 1:numel(Fs)
for i = 1:numel(thetas)
    
    sigma = sigmas(k);    
    F = Fs(j);
    theta = thetas(i);

    % setup the "Gabor transform"
    [x,y]=meshgrid(-filter_size_halfed:filter_size_halfed,-filter_size_halfed:filter_size_halfed);
    g_sigma = (1./(2*pi*sigma^2)).*exp(((-1).*(x.^2+y.^2))./(2*sigma.^2));
    real_g = g_sigma.*cos((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));
    im_g = g_sigma.*sin((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));

    % perform Gabor transform
    uT =sqrt(conv2(testingImg,real_g,'same').^2+conv2(testingImg,im_g,'same').^2);
    
    % normalize transformed image
    uT = (uT-mean(uT(:)))./std(uT(:));

    % append tranformed images to 'features'
    features(:,:,k,j,i) = uT;
    
end
end
end

% feed the features to GLM.
szG = size(features);
features = reshape(features,[prod(szG(1:2)),prod(szG(3:end))]);
Ctest = glmval(b,features,'logit');
Ctest = reshape(Ctest,szG(1:2));
% subplot(1,3,1)
% imagesc(Ctest)
% title('result of GM+ML')
% subplot(1,3,2)
% imshow(testingImg)
% title('testing Image in the vide')
% pause(0.05)
% subplot(1,3,3)
seg = Ctest > 0.519;
seg = im2double(seg);
% imshow(seg)

size(seg);
mean_seg = mean_seg + seg;    
imshow(mean_seg./n)
% writeVideo(outputVideo,seg);
drawnow
n    
end

mean_seg = mean_seg./n;
figure
imshow(mean_seg)
title('the mean image')
% close(outputVideo);