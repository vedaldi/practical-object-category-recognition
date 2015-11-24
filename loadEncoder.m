function encoder = loadEncoder(encoderType)

if nargin < 1
  encoderType = 'vggm128' ;
end

encoder.type = encoderType ;
encoder.net = load('matconvnet/data/models/imagenet-vgg-m-128.mat') ;
encoder.net.layers(end-2:end) = [] ;
encoder.averageColor = ...
  mean(mean(encoder.net.meta.normalization.averageImage,1),2) ;
