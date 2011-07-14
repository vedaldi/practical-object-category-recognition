function histograms = computeHistogramsFromImageList(vocabulary, names)
% COMPUTEHISGORAMSFROMIMAGELIST  Compute historams for multiple images
%   HISTOGRAMS = COMPUTEHISTOGRAMSFROMIMAGELIST(VOCABULARY, NAMES)
%   computes the histograms of visual words for the list of image
%   paths NAMES by applying iteratively COMPUTEHISTOGRAMFROMIMAGE().

% Author: Andrea Vedaldi

start = tic ;
histograms = cell(1,numel(names)) ;
parfor i = 1:length(names)
  if exist(names{i}, 'file')
    fullPath = names{i} ;
  else
    fullPath = fullfile('data','images',[names{i} '.jpg']) ;
  end
  fprintf('Extracting histogram from %s (time remaining %.2fs)\n', fullPath, ...
          (length(names)-i) * toc(start)/i) ;
  histograms{i} = computeHistogramFromImage(vocabulary, fullPath) ;
end
histograms = [histograms{:}] ;
