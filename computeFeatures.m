function [keypoints,descriptors] = computeFeatures(im)
im = standardizeImage(im) ;
[keypoints, descriptors] = vl_phow(im, 'step', 4, 'floatdescriptors', true) ;
