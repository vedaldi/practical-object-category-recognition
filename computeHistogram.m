function histogram = computeHistogram(width, height, keypoints, words)
% COMPUTEHISTOGRAM Compute a spatial histogram of visual words
%   HISTOGRAM = COMPUTEHISTOGRAM(WIDTH, HEIGHT, KEYPOINTS, WORDS)
%   computes a 2x2 spatial histogram of the N visual words
%   WORDS. KEYPOINTS is a 2 x N matrix of x,y coordinates of the
%   visual words and WIDTH and HEIGHT are the image dimensions; these
%   quantities are needed for the geometric tiling.

% Author: Andrea Vedaldi

numWords = 1000 ;
numSpatialX = 2 ;
numSpatialY = 2 ;

for i = 1:length(numSpatialX)
  binsx = vl_binsearch(linspace(1,width,numSpatialX(i)+1), keypoints(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,numSpatialY(i)+1), keypoints(2,:)) ;
  bins = sub2ind([numSpatialY(i), numSpatialX(i), numWords], ...
                 binsy,binsx,words) ;
  htile = zeros(numSpatialY(i) * numSpatialX(i) * numWords, 1) ;
  htile = vl_binsum(htile, ones(size(bins)), bins) ;
  htiles{i} = single(htile / sum(htile)) ;
end

histogram = cat(1,htiles{:}) ;
histogram = single(histogram / sum(histogram)) ;
