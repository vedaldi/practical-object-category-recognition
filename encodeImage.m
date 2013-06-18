function psi = encodeImage(encoder, im, cache)
% COMPUTEENCODING Compute a spatial encoding of visual words
%   PSI = ENCODEIMAGE(ENCODER, IM) applies the specified ENCODER
%   to image IM, reurning a corresponding code vector PSI.
%
%   IM can be an image, the path to an image, or a cell array of
%   the same, to operate on multiple images.
%
%   ENCODEIMAGE(ENCODER, IM, CACHE) utilizes the specified CACHE
%   directory to store encodings for the given images. The cache
%   is used only if the images are specified as file names.

% Author: Andrea Vedaldi

if ~iscell(im), im = {im} ; end
if nargin <= 2, cache = [] ; end

psi = cell(1,numel(im)) ;
if numel(im) > 1
  parfor i = 1:numel(im)
    psi{i} = processOne(encoder, im{i}, cache) ;
  end
else
  psi{1} = processOne(encoder, im{1}, cache) ;
end
psi = cat(2, psi{:}) ;

% --------------------------------------------------------------------
function psi = processOne(encoder, im, cache)
% --------------------------------------------------------------------
if isstr(im)
  if ~isempty(cache)
    psi = getFromCache(im, cache) ;
    if ~isempty(psi), return ; end
  end
  fprintf('encoding image %s\n', im) ;
end

psi = encodeOne(encoder, im) ;

if isstr(im) & ~isempty(cache)
  psi = storeToCache(im, cache, psi) ;
end

% --------------------------------------------------------------------
function psi = encodeOne(encoder, im)
% --------------------------------------------------------------------

im = standardizeImage(im) ;

[keypoints, descriptors] = computeFeatures(im) ;

imageSize = size(im) ;
psi = {} ;
for i = 1:size(encoder.subdivisions,2)
  minx = encoder.subdivisions(1,i) * imageSize(2) ;
  miny = encoder.subdivisions(2,i) * imageSize(1) ;
  maxx = encoder.subdivisions(3,i) * imageSize(2) ;
  maxy = encoder.subdivisions(4,i) * imageSize(1) ;

  ok = ...
    minx <= keypoints(1,:) & keypoints(1,:) < maxx  & ...
    miny <= keypoints(2,:) & keypoints(2,:) < maxy ;

  switch encoder.type
    case 'bovw'
      [words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
                                         descriptors, 'MaxComparisons', 15) ;
      z = vl_binsum(zeros(encoder.numWords,1), 1, double(words)) ;

    case 'fv'
      z = vl_fisher(encoder.projection * descriptors(:,ok), ...
                    encoder.means, ...
                    encoder.covariances, ...
                    encoder.priors) ;
    case 'vlad'
      [words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
                                         encoder.projection * descriptors, ...
                                         'MaxComparisons', 15) ;
      assign = zeros(encoder.numWords, numel(words), 'single') ;
      assign(sub2ind(size(assign), double(words), 1:numel(words))) = 1 ;
      z = vl_vlad(encoder.projection * descriptors(:,ok), ...
                  encoder.words, ...
                  assign) ;
  end
  psi{i} = z(:) ;
end
psi = cat(1, psi{:}) ;

% --------------------------------------------------------------------
function psi = getFromCache(name, cache)
% --------------------------------------------------------------------
[drop, name] = fileparts(name) ;
cachePath = fullfile(cache, [name '.mat']) ;
if exist(cachePath, 'file')
  data = load(cachePath) ;
  psi = data.psi ;
else
  psi = [] ;
end

% --------------------------------------------------------------------
function storeToCache(name, cache, psi)
% --------------------------------------------------------------------
[drop, name] = fileparts(name)
cachePath = fullfile(cache, [name '.mat']) ;
vl_xmkdir(cache) ;
data.psi = psi ;
save(cachePath, '-STRUCT', 'data') ;
