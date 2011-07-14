function vocabulary = computeVocabularyFromImageList(names)
% COMPUTEVOCABULARYFROMIMAGELIST Compute a visual word vocabulary
%   VOCABULARY = COMPUTEVOCABULARYFROMIMAGELIST(NAMES) computes a
%   visual word vocabulary from a list of image names (paths)
%   NAMES.
%
%   VOCABULARY is a structure with fields:
%
%   WORDS:: 128 x K matrix of visual word centers.
%
%   KDTREE:: KD-tree indexing the visual word for fast quantization.

% Author: Andrea Vedaldi

numWords = 1000 ;
numFeatures = numWords * 50 ;

% This extracts a number of visual descriptors from the specified
% images. Only NUMFEATURES overall descriptors are retrieved as more
% do not really improve the estimation of the visual dictionary but
% slow down computation.

descriptors = cell(1,numel(names)) ;
for i = 1:numel(names)
  if exist(names{i}, 'file')
    fullPath = names{i} ;
  else
    fullPath = fullfile('data','images',[names{i} '.jpg']) ;
  end
  fprintf('Extracting features from %s\n', fullPath) ;
  im = imread(fullPath) ;
  [drop, d] = computeFeatures(im) ;
  descriptors{i} = vl_colsubset(d, round(numFeatures / numel(names)), 'uniform') ;
end

% This clusters the descriptors into NUMWORDS visual words by using
% KMEANS. It then compute a KDTREE to index them. The use of a KDTREE
% is optional, but speeds-up quantization significantly.

fprintf('Computing visual words and kdtree\n') ;
descriptors = single([descriptors{:}]) ;
vocabulary.words = vl_kmeans(descriptors, numWords, 'verbose', 'algorithm', 'elkan') ;
vocabulary.kdtree = vl_kdtreebuild(vocabulary.words) ;
