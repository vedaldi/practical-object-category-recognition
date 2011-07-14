function [words,distances] = quantizeDescriptors(vocabulary, descriptors)
% QUANTIZEDESCRIPTOR  Quantize a visual descriptor to get a visual word
%   [WORDS, DISTANCES] = QUANTIZEDESCRIPTORS(VOCABULARY, DESCRIPTORS)
%   projects the D x N matrix of DESCRIPTORS to the visual word
%   VOCABULARY. WORDS is a 1 x N vector containing the indexes of the
%   nearest visual word to each descriptor and DISTANCES a 1 x N
%   vector containing the distances.

% Author: Andrea Vedaldi

[words,distances] = vl_kdtreequery(vocabulary.kdtree, vocabulary.words, ...
                                   descriptors, 'MaxComparisons', 15) ;
words = double(words) ;
