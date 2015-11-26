function names = getImageSet(path, augment)
% GETIMAGESET  Scan a directory for images
%   NAMES = GETIMAGESET(PATH) scans PATH for JPG, PNG, JPEG, GIF,
%   BMP, and TIFF files and returns their path into NAMES.

% Author: Andrea Vedaldi

if nargin < 2, augment = false ; end

content = dir(path) ;
names = {content.name} ;
ok = regexpi(names, '.*\.(jpg|png|jpeg|gif|bmp|tiff)$', 'start') ;
names = names(~cellfun(@isempty,ok)) ;

for i = 1:length(names)
  names{i} = fullfile(path,names{i}) ;
end

if augment
  names = horzcat(names, strcat(names, '_flip')) ;
end