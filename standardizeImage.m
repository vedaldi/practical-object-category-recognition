function im = standardizeImage(im)
im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end
