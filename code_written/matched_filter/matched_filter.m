im =  imread('../flow.png');
im =  rgb2gray(im);

threshold = 45;   %chosen based on results needs improvement
L     = 9;        %from paper http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=34715&tag=1
sigma = 2;        
mean_filter = fspecial('average', [5 5]); %to remove noise

imshow(im);
im  = imfilter(im, mean_filter);
imshow(im)

gaus_filter = fspecial('gaussian',[L L], sigma);
gaus_filter = 10*(gaus_filter - mean(gaus_filter(:)));

size_img   = size(im);
no_filters = 180/15; %divide into sets of 15 degrees   

filtered   = zeros([size_img no_filters]); %filter and store result
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
