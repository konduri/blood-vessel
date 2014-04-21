%following piece of code is just to show how the gabor filter looks like

[x,y]   = meshgrid(-50:50,-50:50); 
sigma   = 10; 
theta   = pi/3; 
F       = 0.04;         
g_sigma = (1./(2*pi*sigma^2)).*exp(((-1).*(x.^2+y.^2))./(2*sigma.^2));     
real_g  = g_sigma.*cos((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));           %real and img part of gaussian
im_g    = g_sigma.*sin((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));

figure;

imagesc([real_g im_g]);
colormap('gray');
axis image;

title('real and imaginary parts of a Gabor filter');


%% Load images and create training and testing images.
% Load an image of the retina and an image indicating where the vessels are in the image.

% let us cleanse Matlab first by clearing the workspace.
clc;clear all;close all;


%%


movie_obj = VideoReader('upmc-ss_pigs-pig_ss15-20131118-105514113.avi');
nFrames   = movie_obj.NumberOfFrames;
%%
image1 = read(movie_obj,1);
image1 = im2double(image1(:,:,1)); %all data bw 0 & 1
hgram  = imhist(im2double(image1(:,:,1)));

image2  = read(movie_obj,2);
image2  = im2double(image2(:,:,1));
image2  = histeq(image2,hgram);


%%
clear movie_obj
%%
%%
% load image (image from wikipedia)
% img = im2double(imread('train_image_small.png'));

% shrink image to decrease computation time.
scale = 0.5; 
% img = imresize(img,scale);

% load answer (location of vessels)
% bwImg = double(imread('train_ans_small.png'));   
% both of the images should be of the same type. here double. try using im2double to sort of normalize




% bwImg = imresize(bwImg,scale,'nearest');
% bwImg(bwImg==255) = 1;
% bwImg(bwImg==0) = 0;

% get training and testing image and vessel location from the above images

trainingImg = image1(936:1638,1:1100);
trainingAns = im2double(imread('train_ans_small.png'));
trainingAns = trainingAns(1:end-3,:);
testingImg = image2(936:1638,1:1100);


trainingAns = imresize(trainingAns,scale,'nearest');
trainingImg = imresize(trainingImg,scale,'nearest');
testingImg  = imresize(testingImg ,scale,'nearest');

imshow(trainingImg)
% testingAns = bwImg(1:175,:);

%% Extract features from training image.

% initialize parameters for Gabor transforms
filter_size = 40.*scale;
filter_size_halfed = round((filter_size)/2);
Fs     = 0.1:0.05:0.11;
sigmas = [2:1:8].*scale;
thetas =pi/8:pi/8:pi+pi/8;

% initialize array for storing features
features = zeros([size(trainingImg),numel(sigmas),numel(thetas),numel(Fs)]);

h1 = figure;

%%%%%%%%%%%

outputVideo = VideoWriter('sample_gabor.avi');
outputVideo.FrameRate = 2;
open(outputVideo);



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
    uT =sqrt(conv2(trainingImg,real_g,'same').^2+conv2(trainingImg,im_g,'same').^2);
    
    % normalize transformed image
    uT = (uT-mean(uT(:)))./std(uT(:));               %%%%%%%%%%%might not have to do this. check it out first                       

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
    
    drawnow;%pause(0.5);
    
    fig = getframe(h1)
    writeVideo(outputVideo,fig);
end
end
end
close(outputVideo);

%% Fit GLM  with features and location of the vessels

% reshape feature array
szG = size(features);
features = reshape(features,[prod(szG(1:2)),prod(szG(3:end))]);

% fit GLM with the features and the location of the vessels
b = glmfit(features,trainingAns(:),'normal');  

% see the output of the model based on the training features
CTrain = glmval(b,features,'logit');
CTrain = reshape(CTrain,szG(1:2));

% visualize 
h2= figure;
% imagesc([trainingImg trainingAns.*255 CTrain.*255]);
imagesc([trainingImg trainingAns CTrain]);
colormap('gray');axis image;
title('testing image, answer, output from GLM');

%% Perform cross validation for Gabor+GLM
% Note that this is a pusedo cross-validation*,** as we used only
% half of an image for training and will be using half of an image for testing.

% Again, perform multiple Gabor transforms with varying parameters.

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

% calculate sensitivity and specificity by thresholding
% the output of GLM 'Ctest' and comparing the thresholded image with the answer.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sensitivity = [];
specificity = [];
rgs = 0:0.01:1;
for i = rgs

    tmpBwImg = im2bw(Ctest,i);
    
    tp = sum(tmpBwImg == 1 & testingAns ==1);
    fn = sum(tmpBwImg == 0 & testingAns ==1);
    tn = sum(tmpBwImg == 0 & testingAns ==0);
    fp = sum(tmpBwImg == 1 & testingAns ==0);
    
    sensitivity = [sensitivity tp/(tp+fn)]; %true positive rate
    specificity = [specificity tn/(tn+fp)]; %true negative rate
    
end

% plot roc curve
h3 = figure;
plot(1-specificity,sensitivity,'k-','linewidth',2);
xlabel('False Positive Rate (1-Specificity)');
ylabel('True Positive Rate (Sensitivity)');
axis([0 1 0 1]);grid on;

% calculate auc.
[fprSort, fprSortInd] = sort([1-specificity],'ascend');
auc = trapz([0 fprSort 1],[0 sensitivity(fprSortInd) 1]);
title(sprintf('ROC curve, AUC: %1.2f',auc));

% get optimal threshold by maximizing Youden's index
[trsh, thInd] = max(sensitivity + specificity - 1);
th = rgs(thInd);

%% Visualize testing image and the detected vessels

h4 = figure;
imagesc([testingImg Ctest.*255 (Ctest > th).*255]);
colormap('gray');axis image;
title('original image, output from GLM, optimally thresholded output from GLM');
