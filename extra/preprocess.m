% PREPROCESS  Build vocabulary and compute histograms
%   PREPROCESS() download an image dataset into 'data/', VLFeat into
%   'vlfeat/', and precompute the histograms for the dataset.

% --------------------------------------------------------------------
%                                                      Download VLFeat
% --------------------------------------------------------------------

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
names{1} = textread('data/background_train.txt','%s') ;
names{2} = textread('data/aeroplane_train.txt','%s') ;
names{3} = textread('data/motorbike_train.txt','%s') ;
names{4} = textread('data/person_train.txt','%s') ;
names{5} = textread('data/car_train.txt','%s') ;
names{6} = textread('data/horse_train.txt','%s') ;
names = cat(1,names{:})' ;
if ~exist('data/vocabulary.mat')
  vocabulary = computeVocabularyFromImageList(vl_colsubset(names,200,'uniform')) ;
  save('data/vocabulary.mat', '-STRUCT', 'vocabulary') ;
else
  vocabulary = load('data/vocabulary.mat') ;
end

% --------------------------------------------------------------------
%                                                   Compute histograms
% --------------------------------------------------------------------

for subset = {'background_train', ...
              'background_val', ...
              'aeroplane_train', ...
              'aeroplane_val', ...
              'motorbike_train', ...
              'motorbike_val', ...
              'person_train', ...
              'person_val', ...
              'car_train', ...
              'car_val', ...
              'horse_train', ...
              'horse_val'}
  fprintf('Processing %s\n', char(subset)) ;
  names = textread(fullfile('data', [char(subset) '.txt']), '%s') ;
  histograms = computeHistogramsFromImageList(vocabulary, names) ;
  save(fullfile('data',[char(subset) '_hist.mat']), 'names', 'histograms') ;
end
