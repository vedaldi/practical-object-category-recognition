% EXPERIMENT2: Effect of the spatial histogram

% add required search paths
setup ;

% load a dataset to be used as positives
positives = load('data/face-histograms.mat') ;
negatives = load('data/background-histograms.mat') ;
names = {positives.names{:}, negatives.names{:}};
histograms = [positives.histograms, negatives.histograms] ;
labels = [ones(1,numel(positives.names)), - ones(1,numel(negatives.names))] ;

% split the data into train and test
selTrain = vl_colsubset(1:numel(labels), .5, 'uniform') ;
selTest = setdiff(1:numel(labels), selTrain) ;

% train and test with increasing number of positive examples
figure(1) ; clf ;
clear leg ;
experiments = {'with-spatial', 'without-spatial', 'with-spatial-l1-norm', 'without-spatial-l1-norm'} ;
colors = jet(numel(experiments)) ;
for i = 1:length(experiments)
  switch experiments{i}
    case 'with-spatial'
      temp = histograms ;
      temp = bsxfun(@times, temp, 1./sqrt(sum(temp.^2,1))) ;

    case 'without-spatial'
      % remove spatial subdivisons by merging them
      temp = histograms(1:4:end,:) + ...
             histograms(2:4:end,:) + ...
             histograms(3:4:end,:) + ...
             histograms(4:4:end,:) / 4 ;
      temp = bsxfun(@times, temp, 1./sqrt(sum(temp.^2,1))) ;

    case 'with-spatial-l1-norm'
      temp = histograms ;

    case 'without-spatial-l1-norm'
      % remove spatial subdivisons by merging them
      temp = histograms(1:4:end,:) + ...
             histograms(2:4:end,:) + ...
             histograms(3:4:end,:) + ...
             histograms(4:4:end,:) / 4 ;
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
