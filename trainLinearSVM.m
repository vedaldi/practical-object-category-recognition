function w = trainLinearSVM(x,y,C)
lambda = 1 / (C * numel(y)) ;
w = vl_pegasos(single(x),int8(y),lambda,'NumIterations',numel(y)*50,'BiasMultiplier',1) ;
w = w(1:end-1) ;

