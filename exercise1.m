% EXPERIMENT1: basic training and testing of a classifier

% add the required search paths
setup ;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% load the positive (relevant) data
%positives = load('data/airplane-histograms.mat') ;
positives = load('data/face-histograms.mat') ;
%positives = load('data/motorbike-histograms.mat') ;
%positives = load('data/car-histograms.mat') ;

% load the negative (distractor) data
negatives = load('data/background-histograms.mat') ;

% extract the names of the images, the histograms, and the labels
% (positive and negative)
names = {positives.names{:}, negatives.names{:}};
histograms = [positives.histograms, negatives.histograms] ;
labels = [ones(1,numel(positives.names)), - ones(1,numel(negatives.names))] ;

% Use half of the images for test.
selTest = vl_colsubset(1:numel(labels), .5, 'uniform') ;

% Make active only a fraction of the training images.
trainSetFraction = 5/100 ;
% trainSetFraction = 10/100 ; For stage F
% trainSetFraction = +inf ; % use 100%
selTrain = vl_colsubset(setdiff(1:numel(labels), selTest), ...
                        trainSetFraction, 'uniform') ;

% count how many images are there
fprintf('number of training images: %d positive, %d negative\n', ...
        sum(labels(selTrain) > 0), sum(labels(selTrain) < 0)) ;
fprintf('number of testing images: %d positive, %d negative\n', ...
        sum(labels(selTest) > 0), sum(labels(selTest) < 0)) ;

% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% For Stage E: Vary the image representation
% histograms = removeSpatialInformation(histograms) ;

% For Stage F: Vary the classifier (Hellinger kernel)
% histograms = sqrt(histograms) ;

% L2 normalize the histograms before running the linear SVM
histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;

% Train the linear SVM
x = histograms(:, selTrain) ;
y = labels(selTrain) ;
C = 10 ;
[w,bias] = trainLinearSVM(x, y, C) ;

% Evaluate the scores on the training data
scores = w'*x + bias ;

% Visualize the ranked list of images
figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
displayRankedImageList(names(selTrain), scores)  ;

% --------------------------------------------------------------------
% Stage C: Classify the test images and assess the performance
% --------------------------------------------------------------------

% Test the linar SVM
x = histograms(:, selTest) ;
y = labels(selTest) ;
scores = w'*x + bias ;

% Visualize the precision-recall curve
figure(2) ; clf ; set(2,'name','Precision-recall on test data') ;
vl_pr(y, scores) ;

% Visualize the ranked list of images
figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
displayRankedImageList(names(selTest), scores)  ;
