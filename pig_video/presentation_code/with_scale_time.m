%%
% load the image into matlab
movie_obj = VideoReader('upmc-ss_pigs-pig_ss15-20131118-105514113.avi');
nFrames   = movie_obj.NumberOfFrames;
%%
%Filters that you will be using, various parameters of Gabor filters
scale              = 0.5;
filter_size        = 40.*scale;
filter_size_halfed = round((filter_size)/2);
Fs                 = 0.1;                   %frequencies
sigmas             = [1:0.5:8].*scale;                  %sigmas used
thetas             = pi/8:pi/8:pi+pi/8;               % orientations
no_filters   = numel(sigmas)*numel(Fs)*numel(thetas); %number of filters used 

len_mesh     = length(-filter_size_halfed:filter_size_halfed);  %length of mesh of gabor filter that we generate
real_filters = zeros(len_mesh^2,no_filters);                    % real filters of gabor filter
imag_filters = zeros(len_mesh^2,no_filters);                    % imaginary ones
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
real_filters = single(real_filters);
imag_filters = single(imag_filters);
%%
load('training_data.mat','training_img'); %should contain images, and truth
training_img = single(training_img);
% training_img = training_img(584:end,1:1350,:);
%%
h1    = figure;
count = 0;
tr_img       = training_img(:,:,1);
tr_img       = imresize(tr_img,scale);   %#################
features     = single(zeros([size(tr_img,1) size(tr_img,2) no_filters])); %vairable used to store features

%%
for i = 1:size(training_img,3)
    tr_img = training_img(:,:,i);
    tr_img = single(mat2gray(tr_img));
    tr_img = imresize(tr_img,0.5);  % ################################
    tr_img = (tr_img-mean(tr_img(:)))./sqrt(var(tr_img(:)));
    for loop =  1:no_filters
        count = count + 1;
        real_g = reshape(real_filters(:,loop),[len_mesh len_mesh]);
        im_g   = reshape(imag_filters(:,loop),[len_mesh len_mesh]);
        uT     = sqrt(conv2(tr_img,real_g,'same').^2+conv2(tr_img,im_g,'same').^2);
        features(:,:,count) = uT;

%         visualize filtered image and the varying filters
        subplot(2,1,1);
        imagesc([tr_img mat2gray(uT).*255],[0 255]);
        imagesc([tr_img mat2gray(uT)]);
        colormap('gray'); axis image; axis off;
        title('testing image and the Gabor transformed image');
        subplot(2,1,2);
        imagesc([real_g im_g]);
        colormap('gray'); axis image; axis off;
        title(sprintf('Gabor filter F:%1.2f t:%1.2f k:%1.f',F,theta,sigma));
        drawnow;
    end
    i
end
%% Fit GLM  with features and location of the vessels
% reshape feature array
szG      = size(features);
features = reshape(features,[prod(szG(1:2)),prod(szG(3:end))]);
%%
% fit GLM with the features and the location of the vessels
load('training_data.mat','training_ans');
%%
% training_ans = training_ans(584:end,1:1350,:);
training_ans = imresize(training_ans,0.25);
sz           = size(training_ans);
temp         = sum(training_ans,3);
training_ans = [reshape(temp,[sz(1)*sz(2) 1]) sz(3)*ones([sz(1)*sz(2) 1])];
%%
clear temp; 
%%
b        = glmfit(features,training_ans(:,1),'normal');  
%%
clear training_ans
%%
% see the output of the model based on the training features
CTrain   = glmval(b,features,'logit');
CTrain   = reshape(CTrain,szG(1:2));
% visualize 
h2= figure;
% imagesc([trainingImg trainingAns.*255 CTrain.*255]);
imagesc([tr_img trainingAns CTrain]);
colormap('gray');axis image;
title('testing image, answer, output from GLM');
%%
img_temp = read(movie_obj,1);
img_temp = img_temp(:,:,1);
sz       = size(img_temp);
mean_seg = single(zeros([sz(1) sz(2)]));
final    = single(zeros([sz(1) sz(2) nFrames]));
figure

%%
full_count = 0;                        %counter for image number read from video
for n = 1:floor(nFrames/length(training_data))
    feature_counter = 0;               %counter for number of features taken  
    features = zeros([size(img_temp),no_filters]);
    for i = 1:length(training_data)    %length of trainging data is the batch we are considering in time.
        full_count  = full_count+1;
        image2      = read(movie_obj,full_count);
        image2      = im2single(image2(:,:,1));
        image2      = (image2-mean(image2(:)))./sqrt(var(image2(:)));
        test_img  = image2;
        for loop =  1:no_filters
            feature_counter = feature_counter+1;
            real_g = reshape(real_filters(:,loop),[len_mesh len_mesh]);
            im_g   = reshape(imag_filters(:,loop),[len_mesh len_mesh]);
            uT =sqrt(conv2(test_img,real_g,'same').^2+conv2(test_img,im_g,'same').^2);
            features(:,:,feature_counter) = uT;
        end 
    end
    szG      = size(features);
    features = reshape(features,[prod(szG(1:2)),prod(szG(3:end))]);
    Ctest    = glmval(b,features,'logit');
    Ctest    = reshape(Ctest,szG(1:2));
%         seg = Ctest > 0.519;
    seg = im2single(seg);
    mean_seg = mean_seg + seg;    
    imshow(mean_seg./n)
    drawnow
    n    
    final(:,:,n) = seg;
end


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





