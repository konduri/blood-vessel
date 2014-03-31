function output_image = kirsch_templates(img)

threshold = 12;

if length(size(img))==3
%     img = mean(img, 3);
    img = rgb2gray(img);
end

kirsch = [-3 -3 -3 -3 -3 5 5 5];

result = zeros([size(img) length(kirsch)]);

for i =  1:length(kirsch)
    temp = circshift(kirsch,i);
    template_i = [temp(1:4) 0 temp(5:end)]/15;
    filter     = reshape(template_i,3,3);
    result(:,:,i) = filter2(filter,img);
end

bloodvessel = zeros(size(img));

for i = 1:size(img,1)
    for j = 1:size(img,2)
        [a,ind] = max(result(i,j,:));
        if a > threshold
            bloodvessel(i,j) = result(i,j,ind);
        end
    end
end

output_image = bloodvessel;

    