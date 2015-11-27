function encoder = loadEncoder(encoderType)

if nargin < 1
  encoderType = 'vggm128-fc7' ;
end

encoder.type = encoderType ;
encoder.net = vl_simplenn_tidy(load('data/cnn/imagenet-vgg-m-128.mat')) ;

switch encoderType
  case 'vggm128-conv1'
    encoder.net.layers(3:end) = [] ;
  case 'vggm128-conv2'
    encoder.net.layers(7:end) = [] ;
  case 'vggm128-conv3'
    encoder.net.layers(11:end) = [] ;
  case 'vggm128-conv4'
    encoder.net.layers(13:end) = [] ;
  case 'vggm128-conv5'
    encoder.net.layers(15:end) = [] ;
  case 'vggm128-fc7'
    encoder.net.layers(20:end) = [] ;
end

encoder.averageColor = ...
  mean(mean(encoder.net.meta.normalization.averageImage,1),2) ;
