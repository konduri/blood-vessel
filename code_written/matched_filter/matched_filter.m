close all
clc

im =  imread('../flow.png');
% im = imread('../Input.bmp');
im =  rgb2gray(im);

threshold   = 30;   %chosen based on results needs improvement
L           = 6;        %from paper http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=34715&tag=1
sigma       = 2:0.5:3.5;        
mean_filter = fspecial('average', [4 4]); %to remove noise


% imshow(im)
% im  = imfilter(im, mean_filter);
% imshow(im)

% gaus   = [0 4 3 2 1 -2 -5 -6 -5 -2 1 2 3 4 0]/110;

size_img    = size(im);
no_filters  = 180/15; %divide into sets of 15 degrees   

filtered    = zeros([size_img no_filters]); %filter and store result


 
for k = 1:length(sigma)
    gaus  = fspecial('gaussian',[1 16], sigma(k));
    gaus = -gaus + mean(gaus);
    gaus  = repmat(gaus,L,1);
    gaus_filter = gaus;


    for i = 1:no_filters
        filtered(:,:,(k-1)*no_filters + i) = imfilter(im,imrotate(gaus_filter,(i-1)*15 + 90));
    end


end


% ans = []
vessel = zeros(size_img);

for i = 1:size_img(1)
    for j =  1:size_img(2)
        [a,ind] = max(filtered(i,j,:));
        if a > threshold
            vessel(i,j) = a;
        end
    end
end





subplot(1,2,1)
imshow(im);
subplot(1,2,2)
imshow(vessel);
% imshow(vessel)
%%
%  vessel = vessel;


% %  imshow(vessel)
 BW2 = bwareaopen(vessel,150);
 imshow(BW2)
