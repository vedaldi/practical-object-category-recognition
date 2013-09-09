% EXERCISE2: learn your own model

% add required search paths
setup ;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

encoding = 'bovw' ;
%encoding = 'vlad' ;
%encoding = 'fv' ;

encoder = load(sprintf('data/encoder_%s.mat',encoding)) ;

% Compute positive histograms from your own images
pos.names = getImageSet('data/myImages') ;
if numel(pos.names) == 0, error('Please add some images to data/myImages before running this exercise') ; end
pos.histograms = encodeImage(encoder, pos.names, ['data/cache_' encoding]) ;

% Add default background images
neg = load(sprintf('data/background_train_%s.mat',encoding)) ;
names = {pos.names{:}, neg.names{:}};
histograms = [pos.histograms, neg.histograms] ;
labels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% Load testing data
pos = load(sprintf('data/horse_val_%s.mat',encoding)) ;
%pos = load(sprintf('data/car_val_%s.mat',encoding)) ;
neg = load(sprintf('data/background_val_%s.mat',encoding)) ;
testNames = {pos.names{:}, neg.names{:}};
testHistograms = [pos.histograms, neg.histograms] ;
testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
clear pos neg ;

% count how many images are there
fprintf('Number of training images: %d positive, %d negative\n', ...
        sum(labels > 0), sum(labels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
        sum(testLabels > 0), sum(testLabels < 0)) ;

% Hellinger's kernel
histograms = sign(histograms).*sqrt(abs(histograms)) ;
testHistograms = sign(testHistograms).*sqrt(abs(testHistograms)) ;

% L2 normalize the histograms before running the linear SVM
histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;
testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1))) ;

% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% Train the linear SVM
C = 10 ;
[w, bias] = trainLinearSVM(histograms, labels, C) ;

% Evaluate the scores on the training data
scores = w' * histograms + bias ;

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


