function vocab = build_rgb_vocabulary(image_paths, vocab_size, useLocations)

rng(0, 'twister');

no_of_images = 1500;
no_of_samples = 250;
pos = 1;
feat_list = cell(no_of_images, 1);

for i=1:no_of_images

    img = imread(image_paths{i,1}); % read the image
    
    [R_locations, R_SIFT_Features] = vl_dsift(im2single(img(:,:,1)));
    [G_locations, G_SIFT_Features] = vl_dsift(im2single(img(:,:,2))); 
    [B_locations, B_SIFT_Features] = vl_dsift(im2single(img(:,:,3)));
    
    if useLocations == 1
        R_SIFT_Features = cat(1,R_SIFT_Features, R_locations); % 130
        G_SIFT_Features = cat(1,G_SIFT_Features, G_locations);
        B_SIFT_Features = cat(1,B_SIFT_Features, B_locations);
    end

    %combining all RGB sift features
    SIFT_features = cat(1, R_SIFT_Features, G_SIFT_Features);
    SIFT_features = cat(1, SIFT_features, B_SIFT_Features);
    

    [x, y] = size(SIFT_features);

    
    if y ~= 0  % y being equal to no of siftfeatures
        % random sample
        SIFT_features = SIFT_features';
        resample = cell(no_of_samples, 1); %constructing x by 1 cell array of empty matrices

        r = randi([1 y], 1, no_of_samples);
        for index=1:no_of_samples
            if useLocations == 1
                test = SIFT_features(r(index),1:390); % ask about the 384? 390
            else
                test = SIFT_features(r(index),1:384);
            end
            
            resample{index} = test;
        end

        resample = cell2mat(resample);
        feat_list{pos, 1} = resample';

        pos = pos +1;
    end
end

feat_list = feat_list';
feat_list = feat_list(1:pos-1);
feat_list = cell2mat(feat_list);

[centers, ~] = vl_kmeans(single(feat_list), vocab_size);

disp("Number of features: ")
disp(length(feat_list'))

vocab = centers';

% SIFT_features = 128xN
% vocab = vocab_sizex128

end