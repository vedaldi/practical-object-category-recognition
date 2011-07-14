function displayVisualWordsFromImageList(names, selection)
% DISPLAYVISUALWORDSFROMIMAGELIST

vocabulary = load('data/vocabulary.mat') ;
numWords = size(vocabulary.words,2) ;

% collect visual words
images = cell(1,numel(names)) ;
keypoints = cell(1,numel(names)) ;
words = cell(1,numel(names)) ;
distances = cell(1,numel(names)) ;

for k = 1:length(selection)

  patches = {} ;
  patchesScore = {} ;
  for i = 1:numel(names)
    if k == 1
      % for the first visual words load the images
      if exist(names{i}, 'file')
        fullPath = names{i} ;
      else
        fullPath = fullfile('data','images',[names{i} '.jpg']) ;
      end
      fprintf('Extracting visual words from %s\n', fullPath) ;
      images{i} = standardizeImage(imread(fullPath)) ;
      [keypoints{i},descriptors] = computeFeatures(images{i}) ;
      [words{i},distances{i}] = quantizeDescriptors(vocabulary, descriptors) ;
      clear descriptors ;
    end

    % wrap around to remove spatial index
    word0 = mod(selection(k)-1, numWords) + 1 ;

    % find the visual words equal to word0
    inds = find(words{i} == word0) ;
    if isempty(inds), continue ; end

    [~, perm] = sort(distances{i}(inds), 'descend') ;
    patches{end+1} = cell(1,numel(inds)) ;

    for j = 1:min(10,numel(inds))
      u0 = keypoints{i}(1,inds(perm(j))) ;
      v0 = keypoints{i}(2,inds(perm(j))) ;
      s0 = keypoints{i}(4,inds(perm(j))) ;

      delta = round(s0*2) ;
      u1 = max(1,u0-delta) ;
      u2 = min(size(images{i},2),u0+delta) ;
      v1 = max(1,v0-delta) ;
      v2 = min(size(images{i},1),v0+delta) ;
      patches{i}{j} = imresize(images{i}(v1:v2,u1:u2,:),[32 32]) ;
      patchesScore{i}(j) = distances{i}(inds(perm(j))) ;
    end
  end
  patches = [patches{:}] ;
  patchesScore = [patchesScore{:}] ;

  if isempty(patches)
    warning('Skipping visual word %d as no matches fonud', selection(k)) ;
    continue ;
  end

  figure(100) ; clf ;
  composite = cat(4,patches{:}) ;
  composite = max(0,min(1,composite)) ;
  %[drop, perm] = sort(patchesScore, 'descend') ;
  %composite = composite(:,:,:,perm) ;
  vl_imarray(composite) ;
  set(gca,'xtick',[],'ytick',[]) ; axis image ;
  axis image ;
  title(sprintf('Visual word %d (rank %d)', word0, k)) ;
  drawnow ;
  fprintf('Press any key to advance\n') ;
  pause ;
end
