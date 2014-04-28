%The following piece of code is to show how the gabor filter looks like

[x,y]   = meshgrid(-50:50,-50:50); 
sigma   = 10; 
theta   = pi/3; 
F       = 0.04;
g_sigma = (1./(2*pi*sigma^2)).*exp(((-1).*(x.^2+y.^2))./(2*sigma.^2));
real_g  = g_sigma.*cos((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));
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

% load image (image from wikipedia)
img = double(imread('Fundus_photograph_of_normal_left_eye.tif'));

% shrink image to decrease computation time.
scale = 0.25; 
img = imresize(img,scale);

% load answer (location of vessels)
bwImg = double(imread('Fundus_photograph_of_normal_left_eye_binary.tif'));

bwImg = imresize(bwImg,scale,'nearest');
bwImg(bwImg==255) = 1;
bwImg(bwImg==0) = 0;

% get training and testing image and vessel location from the above images
testingImg = img(1:175,:);
testingAns = bwImg(1:175,:);

trainingImg = img(176:end,:);
trainingAns = bwImg(176:end,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testingImg = double(rgb2gray(imread('../flow.png')));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Extract features from training image.

% initialize parameters for Gabor transforms
filter_size = 40.*scale;
filter_size_halfed = round((filter_size)/2);
Fs = 0.1:0.1:0.3;
sigmas = [2:2:8].*scale;
thetas=pi/8:pi/8:pi-pi/8;

% initialize array for storing features
features = zeros([size(trainingImg),numel(sigmas),numel(thetas),numel(Fs)]);

h1 = figure;
% perform multiple Gabor transforms with varying parameters 
for k = 1:numel(sigmas)
for j = 1:numel(Fs)
for i = 1:numel(thetas)


    sigma = sigmas(k);    
    F = Fs(j);
    theta = thetas(i);

    % setup the Gabor transform
    [x,y]=meshgrid(-filter_size_halfed:filter_size_halfed,-filter_size_halfed:filter_size_halfed);
    g_sigma = (1./(2*pi*sigma^2)).*exp(((-1).*(x.^2+y.^2))./(2*sigma.^2));
    real_g = g_sigma.*cos((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));
    im_g = g_sigma.*sin((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));

    % perform Gabor transform
    uT =sqrt(conv2(trainingImg,real_g,'same').^2+conv2(trainingImg,im_g,'same').^2);
    
    % normalize transformed image
    uT = (uT-mean(uT(:)))./std(uT(:));

    % append tranformed images to 'features'
    features(:,:,k,j,i) = uT;
    
    % visualize filtered image and the varying filters
    subplot(2,1,1);
    imagesc([trainingImg mat2gray(uT).*255],[0 255]);
    colormap('gray'); axis image; axis off;
    title('testing image and the Gabor transformed image');
    subplot(2,1,2);
    imagesc([real_g im_g]);
    colormap('gray'); axis image; axis off;
    title(sprintf('Gabor filter F:%1.2f t:%1.2f k:%1.f',F,theta,sigma));
    
    drawnow;%pause(0.5);
    
end
end
end



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
imagesc([trainingImg trainingAns.*255 CTrain.*255]);
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

sensitivity = [];
specificity = [];
rgs = 0:0.01:1;
% for i = rgs
% 
%     tmpBwImg = im2bw(Ctest,i);
%     
%     tp = sum(tmpBwImg == 1 & testingAns ==1);
%     fn = sum(tmpBwImg == 0 & testingAns ==1);
%     tn = sum(tmpBwImg == 0 & testingAns ==0);
%     fp = sum(tmpBwImg == 1 & testingAns ==0);
%     
%     sensitivity = [sensitivity tp/(tp+fn)]; %true positive rate
%     specificity = [specificity tn/(tn+fp)]; %true negative rate
%     
% end
% 
% % plot roc curve
% h3 = figure;
% plot(1-specificity,sensitivity,'k-','linewidth',2);
% xlabel('False Positive Rate (1-Specificity)');
% ylabel('True Positive Rate (Sensitivity)');
% axis([0 1 0 1]);grid on;

% calculate auc.
[fprSort, fprSortInd] = sort([1-specificity],'ascend');
auc = trapz([0 fprSort 1],[0 sensitivity(fprSortInd) 1]);
title(sprintf('ROC curve, AUC: %1.2f',auc));

% get optimal threshold by maximizing Youden's index
[trsh, thInd] = max(sensitivity + specificity - 1);
th = rgs(thInd);

%% Visualize testing image and the detected vessels
% Shows the testing image, the output image from the GLM and a
% thresholded image with a threshold that 
% has a relatively good sensitivity and specificity.
%%
imshow(Ctest > 0.6);
%%
h4 = figure;
imagesc([testingImg Ctest.*255 (Ctest > th).*255]);
colormap('gray');axis image;
title('original image, output from GLM, optimally thresholded output from GLM');

% Sandberg, Berta, Tony Chan, and Luminita Vese. "A level-set and 
% gabor-based active contour algorithm for segmenting textured images." 
% UCLA Department of Mathematics CAM report. 2002.
% http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.7.3145