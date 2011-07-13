function [keypoints,descriptors] = computeFeatures(im)
% COMPUTEFEATURES Compute keypoints and descriptors for an image
%   [KEYPOINTS, DESCRIPTORS] = COMPUTEFEAUTRES(IM) computes the
%   keypoints and descriptors from the image IM. KEYPOINTS is a 4 x K
%   matrix with one column for keypoint, specifying the X,Y location,
%   the SCALE, and the CONTRAST of the keypoint.
%
%   DESCRIPTORS is a 128 x K matrix of SIFT descriptors of the
%   keypoints.

% Author: Andrea Vedaldi

im = standardizeImage(im) ;
[keypoints, descriptors] = vl_phow(im, 'step', 4, 'floatdescriptors', true) ;
