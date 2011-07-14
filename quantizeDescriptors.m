function [words,distances] = quantizeDescriptors(vocabulary, descriptors)
[words,distances] = vl_kdtreequery(vocabulary.kdtree, vocabulary.words, descriptors, 'MaxComparisons', 15) ;
words = double(words) ;
