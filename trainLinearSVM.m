function [w,bias] = trainLinearSVM(x,y,C)
% TRAINLINEARSVM  Train a linear support vector machine
%   W = TRAINLINEARSVM(X,Y,C) learns an SVM from patterns X and labels
%   Y. X is a D x N matrix with N D-dimensiona patterns along the
%   columns. Y is a vector of labels +1 or -1 with N elements. C is
%   the regularization parameter of the SVM. The function returns the
%   vector W of weights of the linear SVM and the bias BIAS.
%
%   To evaluate the SVM there is no need of a special function. Simply
%   use SCORES = W' * X + BIAS.

% Auhtor: Andrea Vedaldi

lambda = 1 / (C * numel(y)) ;
vl_twister('state',0) ;
[w, bias] = vl_svmtrain(single(x), ...
                        y, ...
                        lambda, ...
                        'Solver', 'sdca', ...
                        'BiasMultiplier', 1) ;
