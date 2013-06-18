function histograms = computeHistogramsFromImageList(vocabulary, names, cache)
% COMPUTEHISGORAMSFROMIMAGELIST  Compute historams for multiple images
%   HISTOGRAMS = COMPUTEHISTOGRAMSFROMIMAGELIST(VOCABULARY, NAMES)
%   computes the histograms of visual words for the list of image
%   paths NAMES by applying iteratively
%   COMPUTEHISTOGRAMFROMIMAGE().
%
%   COMPUTEHISTOGRAMSFROMIMAGELIST(VOCABULARY, NAMES, CACHE) caches
%   the results to the CACHE directory.

% Author: Andrea Vedaldi

start = tic ;
histograms = cell(1,numel(names)) ;
if nargin <= 2
  cache = [] ;
end

parfor i = 1:length(names)
  histograms{i} = doOne(vocabulary, names{i}, cache) ;
end
histograms = [histograms{:}] ;

function histogram = doOne(vocabulary, name, cache)
if exist(name, 'file')
  fullPath = name ;
else
  fullPath = fullfile('data','images',[name '.jpg']) ;
end
if ~isempty(cache)
  % try to retrieve from cache
  histogram = getFromCache(fullPath, cache) ;
  if ~isempty(histograms), return ; end
end
fprintf('Extracting histogram from %s\n', fullPath) ;
histogram = computeHistogramFromImage(vocabulary, fullPath) ;
if ~isempty(cache)
  % save to cache
  storeToCache(fullPath, cache, histogram) ;
end

function histogram = getFromCache(fullPath, cache)
[drop, name] = fileparts(fullPath) ;
cachePath = fullfile(cache, [name '.mat']) ;
if exist(cachePath, 'file')
  data = load(cachePath) ;
  histogram = data.histogram ;
else
  histogram = [] ;
end

function storeToCache(fullPath, cache, histogram)
[drop, name] = fileparts(fullPath) ;
cachePath = fullfile(cache, [name '.mat']) ;
vl_xmkdir(cache) ;
data.histogram = histogram ;
save(cachePath, '-STRUCT', 'data') ;
