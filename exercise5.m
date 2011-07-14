% EXPERIMENT 5: discriminative visual words

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
temp = histograms(1:4:end,:) + ...
       histograms(2:4:end,:) + ...
       histograms(3:4:end,:) + ...
       histograms(4:4:end,:) / 4 ;
temp = sqrt(temp) ;
histograms = temp ;

% split the data into train and test
selTrain = vl_colsubset(1:numel(labels), .5, 'uniform') ;
selTest = setdiff(1:numel(labels), selTrain) ;

% train the linear SVM
x = histograms(:, selTrain) ;
y = labels(selTrain) ;
C = 10 ;
[w,bias] = trainLinearSVM(x, y, C) ;


wordScores = bsxfun(@times, x, w) * y' ;
%wordScores = x * y' ;
wordScores = w ;
vocabulary = load('data/vocabulary.mat') ;

% collect visual words
for k = 1:length(wordScores)
  [drop, perm] = sort(-wordScores, 'descend') ;
  word0 = perm(k) ;
  patches = {} ;
  for i = 1:40
    if k == 1
      fprintf('Processing image %i of 40\n', i) ;
      if isempty(im{i})
        im{i} = imread(names{selTest(i)}) ;
        [keypoints{i},descriptors] = computeFeatures(im{i}) ;
        words{i} = quantizeDescriptors(vocabulary, descriptors) ;
      end
    end
    selection = find(words{i} == word0) ;
    selection = vl_colsubset(selection, 10, 'uniform') ;

    patches{end+1} = cell(1,numel(selection)) ;
    for j = 1:numel(selection)
      u0 = keypoints{i}(1,selection(j)) ;
      v0 = keypoints{i}(2,selection(j)) ;
      s0 = keypoints{i}(4,selection(j)) ;

      delta = round(s0*2) ;
      u1 = max(1,u0-delta) ;
      u2 = min(size(im{i},2),u0+delta) ;
      v1 = max(1,v0-delta) ;
      v2 = min(size(im{i},1),v0+delta) ;
      patches{i}{j} = imresize(im{i}(v1:v2,u1:u2,:),[32 32]) ;
    end
  end
  patches = [patches{:}] ;
  figure(1) ; clf ;
  vl_imarray(cat(4,patches{:})) ;
  set(gca,'xtick',[],'ytick',[]) ; axis image ;
  axis image ;
  title(sprintf('Visual word of rank %d (score %f)', k, wordScores(perm(k)))) ;
  drawnow ;
  pause ;
end

