% EXERCSIE1: basic training and testing of a classifier

% add the required search paths
setup ;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% Load training data
pos = load('data/aeroplane_train_hist.mat') ;
%pos = load('data/motorbike_train_hist.mat') ;
%pos = load('data/person_train_hist.mat') ;
neg = load('data/background_train_hist.mat') ;
names = {pos.names{:}, neg.names{:}};
histograms = [pos.histograms, neg.histograms] ;
labels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% Load testing data
pos = load('data/aeroplane_val_hist.mat') ;
%pos = load('data/motorbike_val_hist.mat') ;
%pos = load('data/person_val_hist.mat') ;
neg = load('data/background_val_hist.mat') ;
testNames = {pos.names{:}, neg.names{:}};
testHistograms = [pos.histograms, neg.histograms] ;
testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% For stage G: throw away part of the training data
% fraction = .1 ;
% fraction = .5 ;
fraction = +inf ;

sel = vl_colsubset(1:numel(labels), fraction, 'uniform') ;
names = names(sel) ;
histograms = histograms(:,sel) ;
labels = labels(:,sel) ;
clear sel ;

% count how many images are there
fprintf('Number of training images: %d positive, %d negative\n', ...
        sum(labels > 0), sum(labels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
        sum(testLabels > 0), sum(testLabels < 0)) ;

% For Stage E: Vary the image representation
% histograms = removeSpatialInformation(histograms) ;
% testHistograms = removeSpatialInformation(testHistograms) ;

% For Stage F: Vary the classifier (Hellinger kernel)
% ** insert code here for the Hellinger kernel using  **
% ** the training histograms and testHistograms       **

% L2 normalize the histograms before running the linear SVM
histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;
testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1))) ;

% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% Train the linear SVM. The SVM paramter C should be
% cross-validated. Here for simplicity we pick a valute that works
% well with all kernels.
C = 100 ;
[w, bias] = trainLinearSVM(histograms, labels, C) ;

% Evaluate the scores on the training data
scores = w' * histograms + bias ;

% Visualize visual words by relevance on the first image
% displayRelevantVisualWords(names{1},w)

% Visualize the ranked list of images
figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
displayRankedImageList(names, scores)  ;

% Visualize the precision-recall curve
figure(2) ; clf ; set(2,'name','Precision-recall on train data') ;
vl_pr(labels, scores) ;

% --------------------------------------------------------------------
% Stage C: Classify the test images and assess the performance
% --------------------------------------------------------------------

% Test the linar SVM
testScores = w' * testHistograms + bias ;

% Visualize the ranked list of images
figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
displayRankedImageList(testNames, testScores)  ;

% Visualize the precision-recall curve
figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
vl_pr(testLabels, testScores) ;

% Print results
[drop,drop,info] = vl_pr(testLabels, testScores) ;
fprintf('Test AP: %.2f\n', info.auc) ;

[drop,perm] = sort(testScores,'descend') ;
fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;
