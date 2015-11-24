function im = standardizeImage(im)
% STANDARDIZEIMAGE  Rescale an image to a standard size
%   IM = STANDARDIZEIMAGE(IM) rescale IM to have a height of at most
%   480 pixels.

% Author: Andrea Vedaldi

if isstr(im)
  if exist(im, 'file')
    fullPath = im ;
  else
    fullPath = fullfile('data','images',[im '.jpg']) ;
  end
  im = imread(fullPath) ;
end

im = im2single(im) ;
s = 256/min(size(im,1),size(im,2)) ;
im = imresize(im, round(s*[size(im,1) size(im,2)])) ;
