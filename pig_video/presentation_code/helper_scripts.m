training_data = {}
for i = 1:15
    temp = read(movie_obj,i);
    training_data{i} = temp(:,:,1);
end
    save('training_data.mat','training_data')
    
%%
clear training_ans;
clear training_img;
a = imread('mask_1.bmp');
sz = size(a);
training_img = zeros([sz(1) sz(2) 5]);
training_ans = zeros([sz(1) sz(2) 5]);
for i = 1:5
    imagename = sprintf('%s_%i.bmp','image',i);
    maskname  = sprintf('%s_%i.bmp','mask',i);
    training_img(:,:,i)  = imread(imagename);
    training_ans(:,:,i)  = imread(maskname);
end
save('training_data.mat','training_img','training_ans');
    
%%
% outputVideo           = VideoWriter('edge_thres.avi');
% outputVideo.FrameRate = 20;
% open(outputVideo);
load('final_cropped.mat');
h1 = figure;
final_thr = final;          %create a copy 
sz        = size(final);    
blok      = 5 ;                   %size of block of images we are considering {over time froma}
avg_img   = zeros([sz(1) sz(2) floor(sz(3)/blok)]);
for i = 1:floor(sz(3)/blok)
    avg_img(:,:,i) = mean(final_thr(:,:,blok*(i-1) + 1:blok*i),3);
%     subplot(1,3,1)
%     imshow(avg_img(:,:,i)*1.1);
%     title('Output of test image from glmfit')
    im = (avg_img(:,:,i) > 0.27);  %threshold to convert to b/w img 
%     subplot(1,3,2)
%     imshow(im);
%     title('Thresholded image')
    im = bwareaopen(im, 50);      %remove areas of less than 50 pixels (double edge)
%     subplot(1,3,3)
%     imshow(im)
%     title('After morphological operations')
    [edgelist, labelededgeim] = edgelink(im, 150); %edge link funtion, see http://www.csse.uwa.edu.au/~pk/research/matlabfns/#edgelink
    tol = 0;                      %std deviation of pixels about edges to draw    
    seglist = lineseg(edgelist, tol);
%     subplot(2,2,4)
%     close
    drawedgelist(seglist, size(im), 2, [0 0 0], 3);
%     axis off
    title('edge-linked image')
%     drawnow
    f = getframe;
%     writeVideo(outputVideo,f);
    i                                 %to see where in loop
end
% close(outputVideo);
%%
outputVideo = VideoWriter(fullfile(workingDir,'edge_link.avi'));
outputVideo.FrameRate = 8;
open(outputVideo);
%%%%%%%%%%%%%%%%%%%%
writeVideo(outputVideo,img);
%%%%%%%%%%%%%%%%%%%%
close(outputVideo);


%%
% Attempts to get rid of hardcoded threshold

% 
% outputVideo = VideoWriter('edge_link_fraction.avi');
% outputVideo.FrameRate = 8;
% open(outputVideo);
% load('final_cropped.mat');
h1        = figure;
final_thr = final;          %create a copy 
sz        = size(final);    
blok      = 3 ;                   %size of block of images we are considering {over time froma}
avg_img   = zeros([sz(1) sz(2) floor(sz(3)/blok)]);
for i = 1:floor(sz(3)/blok)
    temp           = mean(final_thr(:,:,blok*(i-1) + 1:blok*i),3);
    avg_img(:,:,i) = temp;
    im = (temp > 2*mean(temp(:)));  %threshold to convert to b/w img 
%     score one is 2*mean(temp(:))
%       see next section for score two
    %     imshow(im)
    [edgelist, labelededgeim] = edgelink(im, 150); %edge link funtion, see http://www.csse.uwa.edu.au/~pk/research/matlabfns/#edgelink
    tol = 0;                                       %std deviation of pixels about edges to draw    
    seglist = lineseg(edgelist, tol);
    drawedgelist(seglist, size(im), 2, [0 0 0], 3);
    axis off
    title('edge-linked image')
    drawnow
    f = getframe;
%     writeVideo(outputVideo,f);
    i
end
% close(outputVideo);

%%
% for getting second score to find out the adaptive thresholds


outputVideo = VideoWriter('edge_link_fraction_true.avi');
outputVideo.FrameRate = 8;
open(outputVideo);

temp = imread('mask_2.bmp');
frac_truth = 2.5*sum(temp(:))/(size(temp,1)*size(temp,2)); %2.5 times the fraction of true pixels in the ground truth image
load('final_cropped.mat');
h1        = figure;
final_thr = final;          %create a copy 
sz        = size(final);    
blok      = 3 ;                   %size of block of images we are considering {over time froma}
avg_img   = zeros([sz(1) sz(2) floor(sz(3)/blok)]);
for i = 1:floor(sz(3)/blok)
    temp           = mean(final_thr(:,:,blok*(i-1) + 1:blok*i),3);
    avg_img(:,:,i) = temp;

    
    [freq, value] =  hist(temp(:),100);
    sum_nice = 0;
    sum_tots = sum(temp(:));
    len      = length(freq);
    for j = len:-1:1
    sum_nice = sum_nice + freq(j)*value(j);
        if sum_nice/sum_tots > frac_truth
            thres = value(j);
%             imshow(temp > thres);
            break
        end
    end
    im = (temp > thres);  %threshold to convert to b/w img 
%     score one is 2*mean(temp(:))
%       see next section for score two
    %     imshow(im)
    [edgelist, labelededgeim] = edgelink(im, 150); %edge link funtion, see http://www.csse.uwa.edu.au/~pk/research/matlabfns/#edgelink
    tol = 0;                                       %std deviation of pixels about edges to draw    
    seglist = lineseg(edgelist, tol);
    drawedgelist(seglist, size(im), 2, [0 0 0], 3);
    axis off
    title('edge-linked image')
    drawnow
    f = getframe;
    writeVideo(outputVideo,f);
    i
end


close(outputVideo);



