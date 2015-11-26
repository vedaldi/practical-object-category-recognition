function encoder = loadEncoder(encoderType)

if nargin < 1
  encoderType = 'vggm128-fc7' ;
end

encoder.type = encoderType ;
encoder.net = vl_simplenn_tidy(load('data/cnn/imagenet-vgg-m-128.mat')) ;
switch encoderType
  case 'vggm128-fc7'
    encoder.net.layers(end-2:end) = [] ;
  case 'vggm128-conv5'
    encoder.net.layers(16:end) = [] ;
  case 'vggm128-conv4'
    encoder.net.layers(13:end) = [] ;
end

encoder.averageColor = ...
  mean(mean(encoder.net.meta.normalization.averageImage,1),2) ;
