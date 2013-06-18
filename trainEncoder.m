function encoder = trainEncoder(images, varargin)
% TRAINENCODER   Train image encoder: BoVW, VLAD, FV
%   ENCODER = TRAINENCOER(IMAGES) trains a BoVW encoder from the
%   specified list of images IMAGES.

opts.type = 'bovw' ;
opts.numWords = 64 ;
opts.seed = 1 ;
opts.pcaDimension = 0 ;
opts.subdivisions = [
   0  0 .5 .5 ;
  .5  0  1 .5 ;
   0 .5 .5  1 ;
  .5 .5  1  1]' ;
opts = vl_argparse(opts, varargin) ;

encoder.type = opts.type ;
encoder.subdivisions = opts.subdivisions ;

%% Step 1: obtain sample image descriptors
descrs = sampleLocalFeatures(images, opts.numWords * 400) ;

%% Step 1 (optional): learn PCA projection
if opts.pcaDimension > 0
  fprintf('%s: learning PCA rotation/projection\n', mfilename) ;
  x = bsxfun(@minus, descrs, mean(descrs,2)) ;
  X = x*x' / numDescriptors ;
  [V,D] = eig(X) ;
  d = diag(D) ;
  [d,perm] = sort(d,'descend') ;
  V = V(:,perm) ;
  endocer.projection = V(:,1:opts.pcaDimension)' ;
  clear X V D d ;
else
  encoder.projection = 1 ;
end
descrs = encoder.projection * descrs ;

%% Step 2: lear VQ or GMM vocabulary
dimension = size(descrs,1) ;
numDescriptors = size(descrs,2) ;

switch encoder.type
  case {'bovw', 'vlad'}
    vl_twister('state', opts.seed) ;
    encoder.numWords = opts.numWords ;
    encoder.words = vl_kmeans(descrs, opts.numWords, 'verbose', 'algorithm', 'ann') ;
    encoder.kdtree = vl_kdtreebuild(encoder.words) ;

  case {'fv'} ;
    vl_twister('state', opts.seed) ;
    [encoder.means, encoder.covariances, encoder.priors] = ...
        vl_gmm(descrs, opts.numWords, 'verbose') ;
end
