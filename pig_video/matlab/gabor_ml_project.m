%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%From here we can write the actual code forthe problem, assuming that we
%now have the ground truths and stuff
movie_obj = VideoReader('upmc-ss_pigs-pig_ss15-20131118-105514113.avi');
nFrames   = movie_obj.NumberOfFrames;
%%
image1      = read(movie_obj,1);
image1      = im2double(image1(:,:,1)); %all data bw 0 & 1
image1      = (image1-mean(image1(:)))./sqrt(var(image1(:)));

%image_mean   = (image_orig-mean(image_orig(:))) ./ sqrt(var(image_orig(:)));
%%
scale       = 0.5; 
trainingImg = image1(936:1638,1:1100);
trainingAns = im2double(imread('train_ans_small.png'));
trainingAns = trainingAns(1:end-3,:);                                      %% make sure that for new ground truth this is not a problem
trainingAns = imresize(trainingAns,scale,'nearest');
trainingImg = imresize(trainingImg,scale,'nearest');    %testing image will come later

%fit to a generalized linear model
filter_size        = 40.*scale;
filter_size_halfed = round((filter_size)/2);
Fs                 = 0.1:0.05:0.11;
sigmas             = [2:1:8].*scale;
thetas             = pi/8:pi/8:pi+pi/8;
% initialize array for storing features
features = zeros([size(trainingImg),numel(sigmas)*numel(thetas)*numel(Fs)]);


no_filters = numel(sigmas)*numel(Fs)*numel(thetas);
len_mesh   = length(-filter_size_halfed:filter_size_halfed);
real_filters = zeros(len_mesh^2,no_filters);
imag_filters = zeros(len_mesh^2,no_filters);
% perform multiple Gabor transforms with varying parameters 

count = 0;                          %to keep track of number of loops that we have gone through
for k = 1:numel(sigmas)
for j = 1:numel(Fs)
for i = 1:numel(thetas)
    count = count +1;               %increment every loop, matlab doesn't support 0 indexing.
    sigma = sigmas(k);    
    F     = Fs(j);
    theta = thetas(i);
    % setup the Gabor transform
    [x,y]   = meshgrid(-filter_size_halfed:filter_size_halfed,-filter_size_halfed:filter_size_halfed);
    g_sigma = (1./(2*pi*sigma^2)).*exp(((-1).*(x.^2+y.^2))./(2*sigma.^2));
    real_g  = g_sigma.*cos((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));
    im_g    = g_sigma.*sin((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));
    real_filters(:,count) = real_g(:);
    imag_filters(:,count) = im_g(:);
end
end
end

%%
h1 = figure;
for loop =  1:no_filters
    real_g = reshape(real_filters(:,loop),[len_mesh len_mesh]);
    im_g   = reshape(imag_filters(:,loop),[len_mesh len_mesh]);
    uT =sqrt(conv2(trainingImg,real_g,'same').^2+conv2(trainingImg,im_g,'same').^2);
    features(:,:,loop) = uT;
    
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
end

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

    for loop =  1:no_filters
        real_g = reshape(real_filters(:,loop),[len_mesh len_mesh]);
        im_g   = reshape(imag_filters(:,loop),[len_mesh len_mesh]);
        uT =sqrt(conv2(testingImg,real_g,'same').^2+conv2(testingImg,im_g,'same').^2);
        features(:,:,loop) = uT;
    end

% feed the features to GLM.
szG = size(features);
features = reshape(features,[prod(szG(1:2)),prod(szG(3:end))]);
Ctest    = glmval(b,features,'logit');
Ctest    = reshape(Ctest,szG(1:2));
seg = Ctest > 0.519;
seg = im2double(seg);
size(seg);
mean_seg = mean_seg + seg;    
imshow(mean_seg./n)
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





