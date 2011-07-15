function names = getImageSet(path)
% GETIMAGESET  Scan a directory for images
%   NAMES = GETIMAGESET(PATH) scans PATH for JPG, PNG, JPEG, GIF,
%   BMP, and TIFF files and returns their path into NAMES.

% Author: Andrea Vedaldi

content = dir(path) ;
names = {content.name} ;
ok = regexpi(names, '.*\.(jpg|png|jpeg|gif|bmp|tiff)$', 'start') ;
names = names(~cellfun(@isempty,ok)) ;

for i = 1:length(names)
  names{i} = fullfile(path,names{i}) ;
end
