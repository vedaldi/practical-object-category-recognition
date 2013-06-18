function descrs = sampleLocalFeatures(images, num, varargin)
% SAMPLELOCALFEATURES

randn('state',0) ;
numImages = numel(images) ;
numDescrsPerImage = ceil(num / numImages) ;
descrs = {} ;
parfor i = 1:numImages
  fprintf('%s: sampling features from image: %s\n', mfilename, images{i}) ;
  [~,descriptors] = computeFeatures(images{i}) ;
  randn('state',0) ;
  rand('state',0) ;
  descrs{i} = vl_colsubset(descriptors, single(numDescrsPerImage)) ;
end
descrs = cat(2, descrs{:}) ;
