function displayRankedImageList(names, scores, varargin)
% DISPLAYRANKEDIMAGELIST  Display a (subset of a) ranked list of images
%   DISPLAYRANKEDIMAGELIST(NAMES, SCORES) displays 100 images from
%   the list of image names NAMES sorted by decreasing scores
%   SCORES.
%
%   Use DISPLAYRANKEDIMAGELIST(..., 'numImages', N) to display N
%   images.

% Author: Andrea Vedaldi

opts.numImages = 81 ;
opts = vl_argparse(opts,varargin) ;

[drop, perm] = sort(scores, 'descend') ;
perm = vl_colsubset(perm, opts.numImages, 'uniform') ;
for i = 1:length(perm)
  vl_tightsubplot(length(perm),i,'box','outer') ;
  imagesc(imread(names{perm(i)})) ;
  title(sprintf('score: %.2f', scores(perm(i)))) ;
  set(gca,'xtick',[],'ytick',[]) ; axis image ;
end
colormap gray ;
