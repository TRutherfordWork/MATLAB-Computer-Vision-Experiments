% Based on James Hays, Brown University

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters. 

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

predicted_categories = cell(1500,1);  % initialise cell array

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in coursework_starter,
%because unique() sorts them. This shouldn't really matter, though.
categories = unique(train_labels); 
num_categories = length(categories);

hyperplanes = cell(15,2);

for i=1:num_categories
    cat_true_false = strcmp(categories{i,1}, train_labels);
    labels = zeros(1500, 1);

    for z=1: length(cat_true_false)
        if cat_true_false(z) == 1
            labels(z,1) = 1;
        else
            labels(z,1) = -1;
        end
    end

    [W, B] = vl_svmtrain(train_image_feats', labels, 0.000000001); % change here - last
    hyperplanes{i, 1} = W;
    hyperplanes{i, 2} = B;
end

[rows, columns] = size(test_image_feats);

for i=1:rows
    best = cell(15, 1);
    for z=1: length(hyperplanes)
        W = cell2mat(hyperplanes(z,1));
        X = test_image_feats(i,1:columns)'; % 64, 192, 4913 - change this
        B = hyperplanes(z,2);
        B = B{1,1};

        value = dot(W,X)+B;
        best{z, 1} = value;
    end

    for q=1:15 % all 15 categories
        if q == 1 % if first category
            best_one = best{q, 1}; % value
            best_name = categories(q); % name of best cat
        elseif best{q, 1} > best_one % ??? - not sure about this
            best_one = best{q, 1};
            best_name = categories(q);
        end
    end

    predicted_categories{i, 1} = char(best_name);
end

% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

end