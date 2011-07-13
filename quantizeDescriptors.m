function words = quantizeDescriptors(vocabulary, descriptors)
words = vl_kdtreequery(vocabulary.kdtree, vocabulary.words, descriptors, 'MaxComparisons', 15) ;
words = double(words) ;


