% EXPERIMENT1: basic training and testing of a classifier

% add required search paths
setup ;

% load a dataset to be used as positives
positives = load('data/face-histograms.mat') ;
%positives = load('data/motorbike-histograms.mat') ;
%positives = load('data/car-histograms.mat') ;
%positives = load('data/airplane-histograms.mat') ;

% load a dataset to be used as negatives
negatives = load('data/background-histograms.mat') ;
%negatives = load('data/face-histograms.mat') ;
%negatives = load('data/motorbike-histograms.mat') ;
%negatives = load('data/car-histograms.mat') ;
%negatives = load('data/airplane-histograms.mat') ;

names = {positives.names{:}, negatives.names{:}};
histograms = [positives.histograms, negatives.histograms] ;
labels = [ones(1,numel(positives.names)), - ones(1,numel(negatives.names))] ;

% L2 normalize the histograms before running the linear SVM
histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;

% split the data into train and test
selTrain = vl_colsubset(1:numel(labels), .5, 'uniform') ;
selTest = setdiff(1:numel(labels), selTrain) ;

% train the linear SVM
x = histograms(:, selTrain) ;
y = labels(selTrain) ;
C = 10 ;
[w,bias] = trainLinearSVM(x, y, C) ;

% test the linar SVM
x = histograms(:, selTest) ;
y = labels(selTest) ;
scores = w'*x + bias ;

% evaluation: compute the precision-recall curve
figure(1) ; clf ;
vl_pr(y, scores) ;

% evaluation: visualize ranked list of images
figure(2) ; clf ;
displayRankedImageList(names(selTest), scores)  ;
