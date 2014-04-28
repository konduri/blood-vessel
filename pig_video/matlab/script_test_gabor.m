% %The current code that you are working on.
% 
% movie_obj = VideoReader('upmc-ss_pigs-pig_ss15-20131118-105514113.avi');   %load the image 
% nFrames   = movie_obj.NumberOfFrames;                                      %find the number of frames
% %%
% %loading video sequence gives each frame as nxmx3 matrix, all three of which
% %are the same. pick one
% 
% image1    = read(movie_obj,1);
% image1    = im2double(image1(:,:,1));                                      
% hgram     = imhist(im2double(image1(:,:,1)));
% 
% %%
% 
% % To see the comparison between the diferent forms of normalization
% 
% % figure
% % 
% % for i = 1:nFrames
% %     buffer_temp  = read(movie_obj, i);
% %     image_orig   = im2double(buffer_temp(:,:,1));
% %     image_hist   = histeq(image_orig,hgram);
% %     image_mean   = (image_orig-mean(image_orig(:))) ./ sqrt(var(image_orig(:)));
% %     drawnow;
% %     subplot(1,3,1)
% %     imshow(image_orig)
% %     title('original image')
% %     subplot(1,3,2)
% %     imagesc(image_hist)
% %     title('histogram corrected image')
% %     subplot(1,3,3)
% %     imagesc(image_mean)
% %     title('mean and variance corrected image')    
% % end
% 
% 
% %%
% 
% train_img = image1(936:1638,1:1100);
% imshow(train_img)
% 
% %%
% %Following piece was written for the ground truth detection in image
% 
% % temp   = []
% % points = []
% % 
% % %%
% % 
% % clear temp
% % close
% % h      = figure
% % imshow(train_img);
% % [x,y]  = getpts(h)
% % temp   = [x,y];
% % points = [points;temp];
% % %%
% % hold on
% % plot(points(:,1),points(:,2),'*')
% %%
% 
% 
% figure
% imshow(train_img)
% %%
% %when you want to use the data above, now yo can simple load the truth
% %image that you have
% figure
% temp    = zeros(size(train_img));
% points2 = abs(round(points));
% 
% for i = 1:length(points)
%     temp(points2(i,2),points2(i,1))= 1;
% end
% imshow(temp)
% pause(1)
% se = strel('line',10,10);
% I2 = imdilate(temp,se);
% imshow(I2)
% imwrite(I2       ,'train_ans_small.png  ')
% imwrite(train_img,'train_image_small.png')

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%From here we can write the actual code forthe problem, assuming that we
%now have the ground truths and stuff
movie_obj = VideoReader('upmc-ss_pigs-pig_ss15-20131118-105514113.avi');
nFrames   = movie_obj.NumberOfFrames;
%%
image1    = read(movie_obj,1);
image1    = im2double(image1(:,:,1)); %all data bw 0 & 1
image1    = (image1-mean(image1(:)))./sqrt(var(image1(:)));

% hgram   = imhist(im2double(image1(:,:,1)));
%image_mean   = (image_orig-mean(image_orig(:))) ./ sqrt(var(image_orig(:)));
%%
scale       = 0.5; 
trainingImg = image1(936:1638,1:1100);
trainingAns = im2double(imread('train_ans_small.png'));
trainingAns = trainingAns(1:end-3,:);                                      %% make sure that for new ground truth this is not a problem
% testingImg  = image2(936:1638,1:1100);
trainingAns = imresize(trainingAns,scale,'nearest');
trainingImg = imresize(trainingImg,scale,'nearest');    %testing image will come later
% testingImg  = imresize(testingImg ,scale,'nearest');

%% 
% outputVideo = VideoWriter('sample_two.avi');
% outputVideo.FrameRate = 8;
% open(outputVideo);
% dummy = read(movie_obj,n);
% dummy = im2double(dummy(:,:,1));
% testingImg = dummy(936:1638,1:1100);
%%
%fit to a generalized linear model

filter_size        = 40.*scale;
filter_size_halfed = round((filter_size)/2);
Fs                 = 0.1:0.05:0.11;
sigmas             = [2:1:8].*scale;
thetas             = pi/8:pi/8:pi+pi/8;
% initialize array for storing features
features = zeros([size(trainingImg),numel(sigmas),numel(thetas),numel(Fs)]);
h1 = figure;

%%%%%%%%%%%

% outputVideo = VideoWriter('sample_gabor.avi');
% outputVideo.FrameRate = 2;
% open(outputVideo);

%%%%%%%%%%%%

% perform multiple Gabor transforms with varying parameters 
for k = 1:numel(sigmas)
for j = 1:numel(Fs)
for i = 1:numel(thetas)
    sigma = sigmas(k);    
    F     = Fs(j);
    theta = thetas(i);
    % setup the Gabor transform
    [x,y]   = meshgrid(-filter_size_halfed:filter_size_halfed,-filter_size_halfed:filter_size_halfed);
    g_sigma = (1./(2*pi*sigma^2)).*exp(((-1).*(x.^2+y.^2))./(2*sigma.^2));
    real_g  = g_sigma.*cos((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));
    im_g    = g_sigma.*sin((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));
    % perform Gabor transform
    size(real_g)
    size(im_g)
    uT =sqrt(conv2(trainingImg,real_g,'same').^2+conv2(trainingImg,im_g,'same').^2);

    % normalize transformed image
%     uT = (uT-mean(uT(:)))./std(uT(:));               %%%%%%%%%%%might not have to do this. check it out first                       

    % append tranformed images to 'features'
    features(:,:,k,j,i) = uT;
    
    % visualize filtered image and the varying filters
    subplot(2,1,1);
    imagesc([trainingImg mat2gray(uT).*255],[0 255]);
    imagesc([trainingImg mat2gray(uT)]);
    colormap('gray'); axis image; axis off;
    title('testing image and the Gabor transformed image');
    subplot(2,1,2);
    imagesc([real_g im_g]);
    colormap('gray'); axis image; axis off;
    title(sprintf('Gabor filter F:%1.2f t:%1.2f k:%1.f',F,theta,sigma));
    drawnow;
%     fig = getframe(h1)
%     writeVideo(outputVideo,fig);
end
end
end
% close(outputVideo);
% until here, we should be having the features to train the regression
% model


%% Fit GLM  with features and location of the vessels

% reshape feature array
szG      = size(features);
features = reshape(features,[prod(szG(1:2)),prod(szG(3:end))]);
%%
% fit GLM with the features and the location of the vessels
b = glmfit(features,trainingAns(:),'normal');  
%%
% see the output of the model based on the training features
CTrain = glmval(b,features,'logit');
CTrain = reshape(CTrain,szG(1:2));

% visualize 
h2= figure;
% imagesc([trainingImg trainingAns.*255 CTrain.*255]);
imagesc([trainingImg trainingAns CTrain]);
colormap('gray');axis image;
title('testing image, answer, output from GLM');


%%





% scale    = 0.5; 
mean_seg = zeros(352,550);
final    = zeros(352,550,150);
figure
for n = 1:150
    image2 = read(movie_obj,n);
    image2 = im2double(image2(:,:,1));
%     image2  = histeq(image2,hgram);
    image2      = (image2-mean(image2(:)))./sqrt(var(image2(:)));
    testingImg  = image2(936:1638,1:1100);
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
%     uT = (uT-mean(uT(:)))./std(uT(:));

    % append tranformed images to 'features'
    features(:,:,k,j,i) = uT;
    
end
end
end

% feed the features to GLM.
szG = size(features);
features = reshape(features,[prod(szG(1:2)),prod(szG(3:end))]);
Ctest    = glmval(b,features,'logit');
Ctest    = reshape(Ctest,szG(1:2));
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
final(:,:,n) = seg;
%for median
temp          = final;
temp(temp==0) = NaN  ;
temp_two      = nanmedian(temp,3);   %median
temp_three    = nanmean(temp,3);     %mean
temp_four     = mode(temp,3);        %mode
%for mean
mean_seg = mean_seg./n;
figure
imshow(mean_seg)
title('the mean image')
% close(outputVideo);



%%
im = read(movie_obj,1);
sz = size(im(:,:,1))
edge_struct = uint8(zeros(sz(1), sz(2), nFrames));
for i = 1:nFrames
    im = read(movie_obj,i);
    im          = im(:,:,1);
    temp        = edge(medfilt2(im,[25 25]),'canny');
    edge_struct(:,:,i) = temp;
    i
end
    %%
imshow(mean(edge_struct,3))
save('edges.mat','edge_struct')


%%
im = read(movie_obj,1);
sz = size(im(:,:,1))
edge_struct = uint8(zeros(sz(1), sz(2), nFrames));
for i = 1:nFrames
    im = read(movie_obj,i);
    im          = im(:,:,1);
    temp        = edge(medfilt2(im,[25 25]),'canny');
    edge_struct(:,:,i) = temp;
    i
end
    %%
imshow(mean(edge_struct,3))
save('edges.mat','edge_struct')





