function saliency = displaySaliencyMap(testName, encoder, w)
% DISPLAYSALIENCYMAP  Display relevant visual words from an image
%   DISPLAYSALIENCYMAP(IM, W) displays in sequence the visual
%   words in the vector SELECTION. A visual word is displayed as a
%   sample of the patches in the image IM that match the most relevant
%   visual words according to the calssifier W.

% Author: Andrea Vedaldi

im = standardizeImage(testName) ;
im_ = bsxfun(@minus, 256*im, encoder.averageColor) ;

% The network is applied convolutionally to the image and the feature
% vector is averaged pooled across. To compute the derivative, we take
% the classifier vector w and distribute (copy) it at all such spatial
% locations.

% Get the size of the feature field computed by the network
info = vl_simplenn_display(encoder.net, 'inputSize', [size(im_), 1]) ;
fh = info.dataSize(1,end) ;
fw = info.dataSize(2,end) ;

% Copy w for all spatial locations
w = repmat(reshape(single(w),1,1,[]), fh, fw) ;

% Comptue network derivatives
res = vl_simplenn(encoder.net, im_, w)  ;

% Get saliency map
saliency = sum(abs(res(1).dzdx),3) ;
saliency = vl_imsmooth(saliency, 2) ;
saliency = saliency / max(saliency(:)) ;

% Visualize the map
figure(100) ; clf ;
subplot(3,1,1) ; imagesc(im) ; axis image off ; title('original image') 
subplot(3,1,2) ; imagesc(saliency) ; axis image off ; title('saliency') 
subplot(3,1,3) ; imagesc(bsxfun(@times, im, saliency)) ; axis image off ; title('product')
