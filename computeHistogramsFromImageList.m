function histograms = computeHistogramsFromImageList(vocabulary, names)
start = tic ;
histograms = cell(1,numel(names)) ;
for i = 1:length(names)
  fprintf('Extracting histogram from %s (time remaining %.2fs)\n', names{i}, ...
          (length(names)-i) * toc(start)/i) ;
  histograms{i} = computeHistogramFromImage(vocabulary, names{i}) ;
end
histograms = [histograms{:}] ;
