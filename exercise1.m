% EXERCISE1: basic training and testing of a classifier

% setup MATLAB to use our software
setup ;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% Load training data
encoding = 'vggm128' ;

category = 'motorbike' ;
%category = 'aeroplane' ;
%category = 'person' ;

pos = load(['data/' category '_train_' encoding '.mat']) ;
neg = load(['data/background_train_' encoding '.mat']) ;

names = {pos.names{:}, neg.names{:}};
descriptors = [pos.descriptors, neg.descriptors] ;
labels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% Load testing data
pos = load(['data/' category '_val_' encoding '.mat']) ;
neg = load(['data/background_val_' encoding '.mat']) ;

testNames = {pos.names{:}, neg.names{:}};
testdescriptors = [pos.descriptors, neg.descriptors] ;
testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% For stage G: throw away part of the training data
% fraction = .1 ;
% fraction = .5 ;
fraction = +inf ;

sel = vl_colsubset(1:numel(labels), fraction, 'uniform') ;
names = names(sel) ;
descriptors = descriptors(:,sel) ;
labels = labels(:,sel) ;
clear sel ;

% count how many images are there
fprintf('Number of training images: %d positive, %d negative\n', ...
        sum(labels > 0), sum(labels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
        sum(testLabels > 0), sum(testLabels < 0)) ;

% For Stage E: Vary the image representation
% descriptors = removeSpatialInformation(descriptors) ;
% testdescriptors = removeSpatialInformation(testdescriptors) ;

% For Stage F: Vary the classifier (Hellinger kernel)
% ** insert code here for the Hellinger kernel using  **
% ** the training descriptors and testdescriptors       **

% L2 normalize the descriptors before running the linear SVM
descriptors = bsxfun(@times, descriptors, 1./sqrt(sum(descriptors.^2,1))) ;
testdescriptors = bsxfun(@times, testdescriptors, 1./sqrt(sum(testdescriptors.^2,1))) ;

% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% Train the linear SVM. The SVM paramter C should be
% cross-validated. Here for simplicity we pick a valute that works
% well with all kernels.
C = 10 ;
[w, bias] = trainLinearSVM(descriptors, labels, C) ;

% Evaluate the scores on the training data
scores = w' * descriptors + bias ;

% Visualize the ranked list of images
figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
displayRankedImageList(names, scores)  ;

% Visualize the precision-recall curve
figure(2) ; clf ; set(2,'name','Precision-recall on train data') ;
vl_pr(labels, scores) ;

% --------------------------------------------------------------------
% Stage C: Classify the test images and assess the performance
% --------------------------------------------------------------------

% Test the linear SVM
testScores = w' * testdescriptors + bias ;

% Visualize the ranked list of images
figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
displayRankedImageList(testNames, testScores)  ;

% Visualize the salinecy map
encoder = loadEncoder() ;
[~,best] = max(testScores) ;
displaySaliencyMap(testNames{best},encoder,w)

% Visualize the precision-recall curve
figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
vl_pr(testLabels, testScores) ;

% Print results
[drop,drop,info] = vl_pr(testLabels, testScores) ;
fprintf('Test AP: %.2f\n', info.auc) ;

[drop,perm] = sort(testScores,'descend') ;
fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;
