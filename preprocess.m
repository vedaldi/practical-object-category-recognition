% PREPROCESS Download data and vlfeat, precompute histograms
%   PREPROCESS() download an image dataset into 'data/', VLFeat into
%   'vlfeat/', and precompute the histograms for the dataset.

% --------------------------------------------------------------------
%                                                    Download the data
% --------------------------------------------------------------------

if ~exist('data','dir'), mkdir('data') ; end

baseURL = 'http://www.robots.ox.ac.uk/~vgg/data' ;
pairs = {...
  {'motorbikes_side', 'motorbike'}, ...
  {'faces', 'face'}, ...
  {'airplanes_side', 'airplane'}, ...
  {'cars_brad', 'car'}, ...
  {'background', 'background'}} ;

for pair = pairs
  pair = pair{1} ;
  if exist(fullfile('data',pair{2}),'dir'), continue ; end
  mkdir(fullfile('data',pair{2})) ;
  from = sprintf('%s/%s/%s.tar', baseURL, pair{1}, pair{1}) ;
  to = fullfile('data', pair{2}) ;
  fprintf('Downloading %s to %s\n', from, to) ;
  untar(from, to) ;
end

if ~exist('vlfeat', 'dir')
  from = 'http://www.vlfeat.org/download/vlfeat-0.9.13-bin.tar.gz' ;
  fprintf('Downloading vlfeat from %s\n', from) ;
  untar(from, 'data') ;
  movefile('data/vlfeat-0.9.13', 'vlfeat') ;
end

% --------------------------------------------------------------------
%                                     Compute a visual word vocabulary
% --------------------------------------------------------------------

setup ;

% from 50 background images
names = getImageSet('data/background') ;
if ~exist('data/vocabulary.mat')
  vocabulary = computeVocabularyFromImageList(names(1:50)) ;
  save('data/vocabulary.mat', '-STRUCT', 'vocabulary') ;
else
  vocabulary = load('data/vocabulary.mat') ;
end

% --------------------------------------------------------------------
%                                                   Compute histograms
% --------------------------------------------------------------------

for subset = {'background', 'face', 'motorbike', 'car', 'airplane'}
  names = getImageSet(fullfile('data', char(subset))) ;
  histograms = computeHistogramsFromImageList(vocabulary, names) ;
  save(fullfile('data',[char(subset) '-histograms.mat']), 'names', 'histograms') ;
end
