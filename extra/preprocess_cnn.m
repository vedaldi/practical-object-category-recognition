function preprocess_cnn()
% PREPROCESS_CNN  Prepare data for the practical
%   PREPROCESS_CNN() downloads an image dataset into 'data/'. Use th
%   `download.sh` script to dowlonad auxiliary software libraries.

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

subsets = {...
  'background_train', ...
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
  'horse_val'} ;

% --------------------------------------------------------------------
%                                                  Compute descriptors
% --------------------------------------------------------------------

encoders = {'vggm128-fc7', 'vggm128-conv5', 'vggm128-conv4', ...
            'vggm128-conv3', 'vggm128-conv2', 'vggm128-conv1'} ;
transforms = {'none', 'flip'} ;
[i,j,k] = ndgrid(1:numel(subsets),1:numel(encoders),1:numel(transforms)) ;

cases = vertcat(subsets(i(:)), encoders(j(:)), transforms(k(:)));

spmd
  addpath extra
end

parfor p = 1:size(cases,2)
  doOne(cases{1,p}, cases{2,p}, cases{3,p}) ;
end

% --------------------------------------------------------------------
function doOne(subset, encoderType, transform)
% --------------------------------------------------------------------
switch char(transform)
  case 'none', suffix = '' ;
  case 'zoom', suffix = '_zoom' ;
  case 'flip', suffix = '_flip' ;
end
if ~strcmp(suffix, '')
  if regexp(char(subset), '_val')
    %return ;
  end
end
outPath = fullfile('data',sprintf('%s_%s%s.mat', char(subset), char(encoderType), suffix)) ;
fprintf('Processing %s\n', outPath) ;
if exist(outPath) ; return ; end
encoder = loadEncoder(encoderType) ;
names = textread(fullfile('data', [char(subset) '.txt']), '%s') ;
names = strcat(names, suffix) ;
descriptors = encodeImage(encoder, names) ;
save(outPath, 'names', 'descriptors') ;

