function histograms = removeSpatialInformation(histograms)
% REMOVESPATIALINFORMATION From spatial to simple histogram
%   HISTOGRAM = REMOVESPATIALINFORMATION(HISTOGRAM) removes the
%   spatial information from the 2 x 2 spatial histogram HISTOGRAM.

n = size(histograms,2) ;
histograms = squeeze(mean(reshape(histograms, [], 4, n), 2)) ;
