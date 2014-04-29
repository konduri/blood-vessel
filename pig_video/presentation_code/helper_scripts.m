training_data = {}
for i = 1:15
    temp = read(movie_obj,i);
    training_data{i} = temp(:,:,1);
end
    save('training_data.mat','training_data')
    
%%