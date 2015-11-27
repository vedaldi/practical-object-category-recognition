function im = standardizeImage(im)
% STANDARDIZEIMAGE  Rescale an image to a standard size
%   IM = STANDARDIZEIMAGE(IM) rescale IM to have a minmimum dimension
%   equal to 256 pixels.

% Author: Andrea Vedaldi

zoom = false ;

if isstr(im)
  tokens = regexp(im, '^(.*?)((_flip|_zoom)?)$', 'tokens') ;
  im = tokens{1}{1} ;
  flip = strcmp(tokens{1}{2},'_flip') ;
  zoom = strcmp(tokens{1}{2},'_zoom') ;
  if exist(im, 'file')
    fullPath = im ;
  else
    fullPath = fullfile('data','images',[im '.jpg']) ;
  end
  im = imread(fullPath) ;
  if flip, im = fliplr(im) ; end 
end

im = im2single(im) ;
s = 256/min(size(im,1),size(im,2)) ;
if zoom, s = s * 1.5 ; end
im = imresize(im, round(s*[size(im,1) size(im,2)])) ;
