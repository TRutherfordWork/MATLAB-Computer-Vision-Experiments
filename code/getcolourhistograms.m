train_image_feats = get_colour_histograms(train_image_paths,16);
test_image_feats = get_colour_histograms(test_image_paths,16);

function[features] = get_colour_histograms(paths, quant)

length = (quant+1)^3;
features = zeros(1500,length);  % initialise array for image feature vectors

% rgb
% YCbCR
% HSV

[rows] = size(paths);

for i=1:rows
    img = imread(paths{i,1});
    %img = rgb2ycbcr(img);
    %img_ycbcr = double(img_ycbcr);
    %[a,b,c] = size(img);
    %disp(c)
    img = double(img);

    img_quant = img/255;
    img_quant = round(img_quant*quant-1) + 1;

    feature_vec = zeros(quant+1,quant+1,quant+1);

    [a,b,c] = size(img_quant);

    for x=1:a
        for d=1:b
            r = img_quant(x,d,1) + 1; % +1 because indexes from 1, not 0
            g = img_quant(x,d,2) + 1;
            b = img_quant(x,d,3) + 1;

            feature_vec(r,g,b) = feature_vec(r,g,b) + 1; % increment the number of values
        end
    end

    feature_vec = feature_vec(:);
    feature_vec = feature_vec';

    [rows2, columns2] = size(feature_vec);

    for z=1:columns2
        features(i,z) = feature_vec(z);
    end
end
    
end