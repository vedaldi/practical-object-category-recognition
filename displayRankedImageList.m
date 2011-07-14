function displayRankedImageList(names, scores, varargin)
% DISPLAYRANKEDIMAGELIST  Display a (subset of a) ranked list of images
%   DISPLAYRANKEDIMAGELIST(NAMES, SCORES) displays 100 images from
%   the list of image names NAMES sorted by decreasing scores
%   SCORES.
%
%   Use DISPLAYRANKEDIMAGELIST(..., 'numImages', N) to display N
%   images.

% Author: Andrea Vedaldi

opts.numImages = 6*6 ;
opts = vl_argparse(opts,varargin) ;

[drop, perm] = sort(scores, 'descend') ;
perm = vl_colsubset(perm, opts.numImages, 'uniform') ;
for i = 1:length(perm)
  vl_tightsubplot(length(perm),i,'box','outer') ;
  if exist(names{perm(i)}, 'file')
    fullPath = names{perm(i)} ;
  else
    fullPath = fullfile('data','images',[names{perm(i)} '.jpg']) ;
  end
  imagesc(imread(fullPath)) ;
  title(sprintf('score: %.2f', scores(perm(i)))) ;
  set(gca,'xtick',[],'ytick',[]) ; axis image ;
end
colormap gray ;
