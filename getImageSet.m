function names = getImageSet(path)
names = dir(fullfile(path, '*.jpg')) ;
names = {names.name} ;
for i = 1:length(names)
  names{i} = fullfile(path,names{i}) ;
end
