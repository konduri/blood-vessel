movie_obj = VideoReader('upmc-ss_pigs-pig_ss15-20131118-105514113.avi');
nFrames   = movie_obj.NumberOfFrames;
%%
image1 = read(movie_obj,1);
hgram  = imhist(im2double(image1(:,:,1)));
%%
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