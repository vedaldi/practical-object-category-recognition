% SETUP  Add the required search paths to MATLAB
if exist('vl_version') ~= 3, run('vlfeat/toolbox/vl_setup') ; end
if exist('vl_nnconv') ~= 3
  run('matconvnet/matlab/vl_setupnn') ;
  if exist('vl_nnconv') ~= 3
    vl_compilenn() ;
  end
end
