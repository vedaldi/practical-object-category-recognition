function vocabulary = computeVocabularyFromImageList(names)

numWords = 1000 ;
numFeatures = numWords * 10 ;

descriptors = cell(1,numel(names)) ;
for i = 1:numel(names)
  fprintf('Extracting features from %s\n', names{i}) ;
  im = imread(names{i}) ;
  [drop, d] = computeFeatures(im) ;
  descriptors{i} = vl_colsubset(d, round(numFeatures / numel(names)), 'uniform') ;
end

fprintf('Computing visual words and kdtree\n') ;
descriptors = single([descriptors{:}]) ;
vocabulary.words = vl_kmeans(descriptors, numWords, 'verbose', 'algorithm', 'elkan') ;
vocabulary.kdtree = vl_kdtreebuild(vocabulary.words) ;
