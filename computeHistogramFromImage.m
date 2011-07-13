function histogram = computeHistogramFromImage(vocabulary,im)
if isstr(im), im = imread(im) ; end
width = size(im,2) ;
height= size(im,1) ;
[keypoints, descriptors] = computeFeatures(im) ;
words = quantizeDescriptors(vocabulary, descriptors) ;
histogram = computeHistogram(width, height, keypoints, words) ;
