% PREPROCESS  Build vocabulary and compute histograms
%   PREPROCESS() download an image dataset into 'data/', VLFeat into
%   'vlfeat/', and precompute the histograms for the dataset.

% --------------------------------------------------------------------
%                                                      Download VLFeat
% --------------------------------------------------------------------

if ~exist('vlfeat', 'dir')
  from = 'http://www.vlfeat.org/sandbox/download/vlfeat-0.9.17-bin.tar.gz' ;
  fprintf('Downloading vlfeat from %s\n', from) ;
  untar(from, 'data') ;
  movefile('data/vlfeat-0.9.17', 'vlfeat') ;
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
names = vl_colsubset(names,2000,'uniform') ;

if ~exist('data/encoder_bovw.mat')
  encoder = trainEncoder(names, 'type', 'bovw', 'numWords', 512) ;
  save('data/encoder_bovw.mat', '-STRUCT', 'encoder') ;
end

if ~exist('data/encoder_vlad.mat')
  encoder = trainEncoder(names, 'type', 'vlad', 'numWords', 32, 'pcaDimension', 80) ;
  save('data/encoder_vlad.mat', '-STRUCT', 'encoder') ;
end

if ~exist('data/encoder_fv.mat')
  encoder = trainEncoder(names, 'type', 'fv', 'numWords', 32, 'pcaDimension', 80) ;
  save('data/encoder_fv.mat', '-STRUCT', 'encoder') ;
end

% --------------------------------------------------------------------
%                                                   Compute histograms
% --------------------------------------------------------------------

for encoderType = {'bovw', 'vlad', 'fv'}
  encoder = load(sprintf('data/encoder_%s.mat',char(encoderType)));
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
    outPath = fullfile('data',sprintf('%s_%s.mat', char(subset), char(encoderType))) ;
    fprintf('Processing %s\n', outPath) ;
    if exist(outPath); continue ; end
    names = textread(fullfile('data', [char(subset) '.txt']), '%s') ;
    histograms = encodeImage(encoder, names) ;
    save(outPath, 'names', 'histograms') ;
  end
end
