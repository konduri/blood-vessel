close all
clc

im =  imread('../flow.png');
im =  rgb2gray(im);

threshold   = 27;   %chosen based on results needs improvement
L           = 12;        %from paper http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=34715&tag=1
sigma       = 7;        
mean_filter = fspecial('average', [4 4]); %to remove noise



imshow(im);
% im  = imfilter(im, mean_filter);
imshow(im)

% gaus_filter = fspecial('gaussian',[L L], sigma);
% gaus_filter = 10*(gaus_filter - mean(gaus_filter(:)));


gaus   = [0 4 3 2 1 -2 -5 -6 -5 -2 1 2 3 4 0]/110;
gaus   = repmat(gaus,9,1);

gaus_filter = gaus;




size_img    = size(im);
no_filters  = 180/15; %divide into sets of 15 degrees   

filtered    = zeros([size_img no_filters]); %filter and store result
for i = 1:no_filters
    filtered(:,:,i) = imfilter(im,imrotate(gaus_filter,(i-1)*15));
end
   
vessel = zeros(size_img); 

for i = 1:size_img(1)
    for j =  1:size_img(2)
        [a,ind] = max(filtered(i,j,:));
        if a > threshold
            vessel(i,j) = a;
        end
    end
end

figure
imshow(vessel)
beep on
beep
bw = hysthresh_2(vessel, 27, 30);

imshow(bw);
%  vessel = vessel;
 imshow(vessel)
 BW2 = bwareaopen(vessel,75);
 imshow(BW2)
