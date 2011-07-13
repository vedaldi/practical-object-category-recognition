% EXPERIMENT4: Effect of the kernel

% add required search paths
setup ;

% load a dataset to be used as positives
positives = load('data/face-histograms.mat') ;
negatives = load('data/background-histograms.mat') ;
names = {positives.names{:}, negatives.names{:}};
histograms = [positives.histograms, negatives.histograms] ;
labels = [ones(1,numel(positives.names)), - ones(1,numel(negatives.names))] ;

% split the data into train and test
selTrain = vl_colsubset(1:numel(labels), 5/100, 'uniform') ;
selTest = setdiff(1:numel(labels), selTrain) ;

% train and test with increasing number of positive examples
figure(1) ; clf ;
clear leg ;
experiments = {'linear-kernel', 'non-linear-kernel'} ;
colors = jet(numel(experiments)) ;
for i = 1:length(experiments)
  switch experiments{i}
    case 'linear-kernel'
      temp = histograms ;
      temp = bsxfun(@times, temp, 1./sqrt(sum(temp.^2,1))) ;

    case 'non-linear-kernel'
      % remove spatial subdivisons by merging them
      temp = sqrt(histograms) ;
  end

  x = temp(:, selTrain) ;
  y = labels(selTrain) ;
  C = 100 ;
  [w,bias] = trainLinearSVM(x, y, C) ;

  x = temp(:, selTest) ;
  y = labels(selTest) ;
  scores = w'*x + bias ;

  [rc,pr,info] = vl_pr(y, scores) ;

  hold on ;
  plot(rc,pr,'linewidth', 2, 'color', colors(i,:)) ;
  leg{i} = sprintf('%s, AP: %.2f', experiments{i}, info.auc*100) ;
end
axis square ; grid on ;
xlabel('recall') ;
ylabel('precision') ;
legend(leg{:}, 'location','sw') ;
title('Note: using only 5/100 of the training data to illustrate') ;