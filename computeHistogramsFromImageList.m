function histograms = computeHistogramsFromImageList(vocabulary, names)
% COMPUTEHISGORAMSFROMIMAGELIST  Compute historams for multiple images
%   HISTOGRAMS = COMPUTEHISTOGRAMSFROMIMAGELIST(VOCABULARY, NAMES)
%   computes the histogras of visual words for the list of image
%   paths NAMES.
%
%   The function calls COMPUTEFEATURES(), QUANTIZEFEATURES(), and
%   COMPUTEHISTOGRAM() for the specified images in sequence. It then
%   returns a matrix HISTOGRAMS with one histogram for each column.

% Author: Andrea Vedaldi

start = tic ;
histograms = cell(1,numel(names)) ;
for i = 1:length(names)
  fprintf('Extracting histogram from %s (time remaining %.2fs)\n', names{i}, ...
          (length(names)-i) * toc(start)/i) ;
  histograms{i} = computeHistogramFromImage(vocabulary, names{i}) ;
end
histograms = [histograms{:}] ;
