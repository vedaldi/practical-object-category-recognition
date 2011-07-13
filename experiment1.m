% load data
positives = load('data/face-histograms.mat') ;
negatives = load('data/background-histograms.mat') ;
names = {positives.names{:}, negatives.names{:}};
patterns = [positives.histograms, negatives.histograms] ;
labels = [ones(1,numel(positives.names)), - ones(1,numel(negatives.names))] ;

% normalize
patterns = bsxfun(@times, patterns, 1./sqrt(sum(patterns.^2,1))) ;

% split
selTrain = vl_colsubset(1:numel(labels), .5, 'uniform') ;
selTest = setdiff(1:numel(labels), selTrain) ;

% train
x = patterns(:, selTrain) ;
y = labels(selTrain) ;
C = 10 ;
w = trainLinearSVM(x, y, C) ;
save('data/face-model.mat','w') ;

% test
x = patterns(:, selTest) ;
y = labels(selTest) ;
scores = w'*x ;
figure(1) ; clf ; vl_pr(y, scores) ;

