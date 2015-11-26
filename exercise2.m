% EXERCISE2: learn your own model

% add required search paths
setup ;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% Whether to use data augmentation or not
augmentation = true ;

% Compute positive descriptors from your own images
encoding = 'vggm128' ;
encoder = loadEncoder(encoding) ;
pos.names = getImageSet('data/myImages', augmentation) ;
if numel(pos.names) == 0, error('Please add some images to data/myImages before running this exercise') ; end
pos.descriptors = encodeImage(encoder, pos.names, ['data/cache_' encoding]) ;

% Add default background images
neg = load(sprintf('data/background_train_%s.mat',encoding)) ;
names = {pos.names{:}, neg.names{:}};
descriptors = [pos.descriptors, neg.descriptors] ;
labels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% Load testing data
pos = load(sprintf('data/horse_val_%s.mat',encoding)) ;
%pos = load(sprintf('data/car_val_%s.mat',encoding)) ;
neg = load(sprintf('data/background_val_%s.mat',encoding)) ;
testNames = {pos.names{:}, neg.names{:}};
testDescriptors = [pos.descriptors, neg.descriptors] ;
testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% count how many images are there
fprintf('Number of training images: %d positive, %d negative\n', ...
        sum(labels > 0), sum(labels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
        sum(testLabels > 0), sum(testLabels < 0)) ;

% L2 normalize the descriptors before running the linear SVM
descriptors = bsxfun(@times, descriptors, 1./sqrt(sum(descriptors.^2,1))) ;
testDescriptors = bsxfun(@times, testDescriptors, 1./sqrt(sum(testDescriptors.^2,1))) ;

% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% Train the linear SVM
C = 10 ;
[w, bias] = trainLinearSVM(descriptors, labels, C) ;

% Evaluate the scores on the training data
scores = w' * descriptors + bias ;

% Visualize the ranked list of images
% figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
% displayRankedImageList(names, scores)  ;

% Visualize the precision-recall curve
% figure(2) ; clf ; set(2,'name','Precision-recall on train data') ;
% vl_pr(labels, scores) ;

% --------------------------------------------------------------------
% Stage C: Classify the test images and assess the performance
% --------------------------------------------------------------------

% Test the linar SVM
testScores = w' * testDescriptors + bias ;

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


