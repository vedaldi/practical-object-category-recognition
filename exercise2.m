% EXPERIMENT2: Effect of the training set size

% add required search paths
setup ;

% load a dataset to be used as positives
positives = load('data/face-histograms.mat') ;
negatives = load('data/background-histograms.mat') ;
names = {positives.names{:}, negatives.names{:}};
histograms = [positives.histograms, negatives.histograms] ;
labels = [ones(1,numel(positives.names)), - ones(1,numel(negatives.names))] ;

% L2 normalize the histograms before running the linear SVM
histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;

% split the data into train and test
selTrain = vl_colsubset(1:numel(labels), .5, 'uniform') ;
selTest = setdiff(1:numel(labels), selTrain) ;

% train and test with increasing number of positive examples
figure(1) ; clf ;
range = [1 2 5 10 100 500 +inf] ;
colors = jet(numel(range)) ;
for i = 1:length(range)
  selPosTrain = selTrain(labels(selTrain) > 0) ;
  selNegTrain = selTrain(labels(selTrain) < 0) ;
  reducedTrain = [vl_colsubset(selPosTrain, range(i), 'beginning') selNegTrain] ;

  x = histograms(:, reducedTrain) ;
  y = labels(reducedTrain) ;
  C = 10 ;
  [w,bias] = trainLinearSVM(x, y, C) ;

  x = histograms(:, selTest) ;
  y = labels(selTest) ;
  scores = w'*x + bias ;

  [rc,pr,info] = vl_pr(y, scores) ;

  hold on ;
  plot(rc,pr,'linewidth', 2, 'color', colors(i,:)) ;
  leg{i} = sprintf('num pos: %d, AP: %.2f', range(i), info.auc*100) ;
end
axis square ; grid on ;
xlabel('recall') ;
ylabel('precision') ;
legend(leg{:}, 'location','sw') ;
