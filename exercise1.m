% EXERCISE1: basic training and testing of a classifier

% setup MATLAB to use our software
setup ;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% For stage G: Change the encoding
%encoding = 'vggm128-conv1' ;
%encoding = 'vggm128-conv2' ;
encoding = 'vggm128-conv3' ;
%encoding = 'vggm128-conv4' ;
%encoding = 'vggm128-conv5' ;
%encoding = 'vggm128-fc7' ;

% For stage E: Change the object category
category = 'motorbike' ;
%category = 'aeroplane' ;
%category = 'person' ;

% For stage G: Change the number of training samples
numPos = 20 ;
numNeg = +inf ;

% For stage I: Change the data augmentation
transforms = {'none'} ;
%transforms = {'none', 'flip'} ;
testTransforms = {'none'} ;
%testTransforms = {'none', 'flip'} ;

% Load the training data
names = {} ;
descriptors = [] ;
labels = [] ;
for transform = transforms
  switch char(transform)
    case 'none', suffix = '' ;
    case 'flip', suffix = '_flip' ;
  end
  pos = load(['data/' category '_train_' encoding suffix '.mat']) ;
  neg = load(['data/background_train_' encoding suffix '.mat']) ;
  selp = vl_colsubset(1:numel(pos.names),numPos,'beginning') ;
  seln = vl_colsubset(1:numel(neg.names),numNeg,'beginning') ;
  names = horzcat(names, pos.names(selp)', neg.names(seln)') ;
  descriptors = horzcat(descriptors, pos.descriptors(:,selp), neg.descriptors(:,seln)) ;
  labels = horzcat(labels, ones(1,numel(selp)), - ones(1,numel(seln))) ;
  clear pos neg ;
end

% Load testing data
testNames = {} ;
testDescriptors = [] ;
testLabels = [] ;
for transform = testTransforms
  switch char(transform)
    case 'none', suffix = '' ;
    case 'flip', suffix = '_flip' ;
  end
  pos = load(['data/' category '_val_' encoding suffix '.mat']) ;
  neg = load(['data/background_val_' encoding suffix '.mat']) ;
  testNames = {pos.names{:}, neg.names{:}};
  testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
  testDescriptors = cat(3, testDescriptors, [pos.descriptors, neg.descriptors]) ;
  clear pos neg ;
end
testDescriptors = mean(testDescriptors,3) ;

% Count how many images are there
fprintf('Number of training images: %d positive, %d negative\n', ...
        sum(labels > 0), sum(labels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
        sum(testLabels > 0), sum(testLabels < 0)) ;

% For stage H: Change the descriptor normalization
descriptors = bsxfun(@times, descriptors, 1./sqrt(sum(descriptors.^2,1))) ;
testDescriptors = bsxfun(@times, testDescriptors, 1./sqrt(sum(testDescriptors.^2,1))) ;

%descriptors = bsxfun(@times, descriptors, 1./sum(abs(descriptors),1)) ;
%testDescriptors = bsxfun(@times, testDescriptors, 1./sum(abs(testDescriptors),1)) ;

% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% For stage F: change the value of the C parameter in the SVM (this
% parameter should be cross-validated).
C = 10 ;
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

% For stage J: Visualize the salinecy map
% encoder = loadEncoder(encoding) ;
% [~,best] = max(testScores) ;
% displaySaliencyMap(testNames{best},encoder,max(w,0)) ;

% Visualize the precision-recall curve
figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
vl_pr(testLabels, testScores) ;

% Print results
[drop,drop,info] = vl_pr(testLabels, testScores) ;
fprintf('Test AP: %.2f\n', info.auc) ;

[drop,perm] = sort(testScores,'descend') ;
fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;
