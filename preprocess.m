names = getImageSet('data/background') ;

if ~exist('data/vocabulary.mat')
  vocabulary = computeVocabularyFromImages(names(1:10)) ;
  save('data/vocabulary.mat', '-STRUCT', 'vocabulary') ;
else
  vocabulary = load('data/vocabulary.mat') ;
end

for subset = {'background', 'face', 'motorbike', 'car', 'airplane'}
  names = getImageSet(fullfile('data', char(subset))) ;
  histograms = computeHistogramsFromImageList(vocabulary, names) ;
  save(fullfile('data',[char(subset) '-histograms.mat']), 'names', 'histograms') ;
end
