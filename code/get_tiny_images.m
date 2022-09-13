function [features] = get_tiny_images(paths, img_size, colour)

length = 0; % length of the feature vector

if strcmp(colour,'gray')
    length = img_size*img_size; % size of feature vector if image is grayscale
elseif strcmp(colour, 'rgb')
    length = (img_size*img_size)*3; % size of the feature vector if the image is rgb
else
    error('Error occurred')
end

features = zeros(1500,length);  % initialise array for image feature vectors
[rows] = size(paths); % gets the number of images in dataset being parsed


for i = 1:rows % for every image (1 to 1500)

    img = imread(paths{i,1}); % read the image
    [height, width, colour] = size(img); % gets height, width, and colour space values of image

    if height > width
        target = [width,width]; % make window according to dimensions of width

    else
        target = [height,height]; % make window according to dimensions of height
    end

    crop_win = centerCropWindow2d(size(img), target); % create window
    img = imcrop(img,crop_win); % crop image to center square
    img = imresize(img,[img_size,img_size]); % resize by dimension specified in parameter

    if length == (img_size*img_size)
        img = rgb2gray(img); % turn to grayscale
    end

    img = double(img);
    img = img(:); % converts 2D image into 1D vector
    img = img'; % makes the vector go horizontal, instead of vertical
    img = normalize(img); % normalizing image unit length

    [rows2, columns2] = size(img);
   
    % make it to a array
    for z=1:columns2
        features(i,z) = img(z);
    end

end

end