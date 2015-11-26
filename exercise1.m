% EXERCISE1: basic training and testing of a classifier

% setup MATLAB to use our software
setup ;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% Choose an encoding
encoding = 'vggm128-conv4' ;
%encoding = 'vggm128-conv5' ;
%encoding = 'vggm128-fc7' ;

% Choose an object category
%category = 'aeroplane' ;
category = 'motorbike' ;
%category = 'car' ;
%category = 'person' ;

% For stage G: choose the number of training samples (set to +inf to use
% all)
numPos = +inf ;
numNeg = +inf ;

% For stage X: choose which data augmentation to use
transforms = {'none'} ;
%transforms = {'none', 'flip'} ;

% Load the training data
names = {} ;
descriptors = [] ;
labels = [] ;
for transform = transforms
  switch char(transform)
    case 'none', suffix = '' ;
    case 'zoom', suffix = '_zoom' ;
    case 'flip', suffix = '_flip' ;
  end
  pos = load(['data/' category '_train_' encoding suffix '.mat']) ;
  neg = load(['data/background_train_' encoding suffix '.mat']) ;
  selp = vl_colsubset(1:numel(pos.names),numPos,'beginning') ;
  seln = vl_colsubset(1:numel(neg.names),numNeg,'beginning') ;
  names = horzcat(names, pos.names(selp)', neg.names(seln)') ;
  descriptors = horzcat(descriptors, pos.descriptors(:,selp), neg.descriptors(:,seln)) ;
  labels = horzcat(labels, ones(1,numel(selp)), - ones(1,numel(seln))) ;
end
clear pos neg ;

% Load testing data
pos = load(['data/' category '_val_' encoding '.mat']) ;
neg = load(['data/background_val_' encoding '.mat']) ;
testNames = {pos.names{:}, neg.names{:}};
testDescriptors = [pos.descriptors, neg.descriptors] ;
testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% Count how many images are there
fprintf('Number of training images: %d positive, %d negative\n', ...
        sum(labels > 0), sum(labels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
        sum(testLabels > 0), sum(testLabels < 0)) ;
            
% For Stage E: Vary the classifier (Hellinger kernel)
% L2 normalize the descriptors before running the linear SVM
descriptors = bsxfun(@times, descriptors, 1./sqrt(sum(descriptors.^2,1))) ;
testDescriptors = bsxfun(@times, testDescriptors, 1./sqrt(sum(testDescriptors.^2,1))) ;

% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% Train the linear SVM. The SVM paramter C should be
% cross-validated. Here for simplicity we pick a valute that works
% well with all kernels.
C = 1 ;
[w, bias] = trainLinearSVM(descriptors, labels, C) ;

% Evaluate the scores on the training data
scores = w' * descriptors + bias ;

% Visualize the ranked list of images
figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
displayRankedImageList(names, scores)  ;

% Visualize the precision-recall curve
figure(2) ; clf ; set(2,'name','Precision-recall on training data') ;
vl_pr(labels, scores) ;

% --------------------------------------------------------------------
% Stage C: Classify the test images and assess the performance
% --------------------------------------------------------------------

% Test the linear SVM
testScores = w' * testDescriptors + bias ;

% Visualize the ranked list of images
figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
displayRankedImageList(testNames, testScores)  ;

% Visualize the salinecy map
encoder = loadEncoder() ;
[~,best] = max(testScores) ;
displaySaliencyMap(testNames{best},encoder,w) ;

% Visualize the precision-recall curve
figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
vl_pr(testLabels, testScores) ;

% Print results
[drop,drop,info] = vl_pr(testLabels, testScores) ;
fprintf('Test AP: %.2f\n', info.auc) ;

[drop,perm] = sort(testScores,'descend') ;
fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;
