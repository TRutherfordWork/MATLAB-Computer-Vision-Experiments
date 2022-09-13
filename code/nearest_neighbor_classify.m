function predictions = nearest_neighbor_classify(train_features,train_lab, test_features)
% Input: Training image feature vectors, test image feature vectors,training image labels
% Output: Predictions of what each test image is

k = 10; % K-value for nearest neighbour (value of 3 best for colour histogram feature extraction)

cats = {'Kitchen','Store','Bedroom','LivingRoom','House','Industrial','Stadium','Underwater','TallBuilding','Street','Highway','Field','Coast','Mountain','Forest'};
predictions = cell(1500,1);  % initialise cell array

[rows_test, columns_test] = size(test_features); % gets number of rows & columns in testing feature vector cell array
[rows_train, columns_train] = size(train_features); % gets number of rows & columns in training feature vector cell array


for i=1:rows_test % for every test image (1 to 1500)
    test = test_features(i,1:columns_test); % get feature vector of test image
    closest2test = cell(k,3); % cell array holding info regarding training images closest to test image

    for x=1:rows_train % for every train image
        train = train_features(x,1:columns_train); % get feature vec of train image
        distance = pdist2(test,train); % distance between test and train images

        if x <= k % if training image is <= k, populate the cell array with first k values
            closest2test{x,1} = x; % set 1st train img as closest
            closest2test{x,2} = distance; % distance of closest
            closest2test{x,3} = train_lab{x,1}; % name of category

        else % if greater than k, check if distance between test and train image is smaller than image already stored in closest2test

            for r=1:k

                if distance < closest2test{r,2} % if distance of current test/train image is less than value stored in 

                    % update values to store new closer training image
                    closest2test{r,1} = x;
                    closest2test{r,2} = distance;
                    closest2test{r,3} = train_lab{x,1};
                    break
                end

            end
        end

        if x == rows_train % if all training images been compared
            labels = closest2test(:,3); % gets the identified labels of closest images
            final_closest = cell(1,2);

            for cat_i = 1:length(cats) % for all 15 categories of images
                count = nnz(ismember(labels,cats{1,cat_i})); % count how many times it appears in closest training images to test

                if cat_i == 1 % if the first category

                    % populate cell array with values
                    final_closest{1,1} = cats{1,cat_i};
                    final_closest{1,2} = count;

                elseif count > final_closest{1,2} % if the count of current category higher than previous closest category
                    % update values
                    final_closest{1,1} = cats{1,cat_i};
                    final_closest{1,2} = count;
                end
            end

            predictions{i,1} = final_closest{1,1}; % add closest to predictions cell array
        end

    end
end

end