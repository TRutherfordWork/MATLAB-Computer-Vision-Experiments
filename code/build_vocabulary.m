% Based on James Hays, Brown University

%This function will sample SIFT descriptors from the training images,
%cluster them with kmeans, and then return the cluster centers.

function vocab = build_vocabulary(image_paths, vocab_size, useLocations)

rng(0, 'twister')

no_of_images = 1500;
no_of_samples = 250;
pos = 1;
feat_list = cell(no_of_images, 1);

for i=1:no_of_images

    img = imread(image_paths{i,1}); % read the image
    img = im2gray(img);
    img = im2single(img);
    [locations, SIFT_features] = vl_dsift(img);
    % appending locations onto sift_features vector
    if useLocations == 1
        SIFT_features = cat(1,SIFT_features, locations);
    end
    [x, y] = size(SIFT_features);

    if y ~= 0  % y being equal to no of siftfeatures
        % random sample
        SIFT_features = SIFT_features';
        resample = cell(no_of_samples, 1); % constructing x by 1 cell array of empty matrices

        r = randi([1 y], 1, no_of_samples);
        for index=1:no_of_samples
            if useLocations == 1
                test = SIFT_features(r(index),1:130);
            else
                test = SIFT_features(r(index), 1:128);
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

% The inputs are images, a N x 1 cell array of image paths and the size of 
% the vocabulary.

% The output 'vocab' should be vocab_size x 128. Each row is a cluster
% centroid / visual word.

%{ 
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be thrown away here
  (but possibly used for extra credit in get_bags_of_sifts if you're making
  a "spatial pyramid").
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

[centers, assignments] = vl_kmeans(X, K)
 http://www.vlfeat.org/matlab/vl_kmeans.html
  X is a d x M matrix of sampled SIFT features, where M is the number of
   features sampled. M should be pretty large! Make sure matrix is of type
   single to be safe. E.g. single(matrix).
  K is the number of clusters desired (vocab_size)
  centers is a d x K matrix of cluster centroids. This is your vocabulary.
   You can disregard 'assignments'.

  Matlab has a build in kmeans function, see 'help kmeans', but it is
  slower.
%}

% Load images from the training set. To save computation time, you don't
% necessarily need to sample from all images, although it would be better
% to do so. You can randomly sample the descriptors from each image to save
% memory and speed up the clustering. Or you can simply call vl_dsift with
% a large step size here, but a smaller step size in make_hist.m. 

% For each loaded image, get some SIFT features. You don't have to get as
% many SIFT features as you will in get_bags_of_sift.m, because you're only
% trying to get a representative sample here.

% Once you have tens of thousands of SIFT features from many training
% images, cluster them with kmeans. The resulting centroids are now your
% visual word vocabulary.