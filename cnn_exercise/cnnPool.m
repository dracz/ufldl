function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

resultDim = floor(convolvedDim / poolDim);
pooledFeatures = zeros(numFeatures, numImages, resultDim, resultDim);

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------


for featureNum=1:numFeatures
  for imgNum=1:numImages
    for poolRow=1:resultDim
      rowStart = 1 + (poolRow-1)*poolDim;
      rowEnd = rowStart + poolDim - 1;
      for poolCol=1:resultDim
	colStart = 1 + (poolCol-1)*poolDim;
	colEnd = colStart + poolDim - 1;
	pooled = convolvedFeatures(featureNum, imgNum, ...
				   rowStart:rowEnd, colStart:colEnd);
	pooledFeatures(featureNum, imgNum, poolRow, poolCol) = mean(pooled(:));
      end
    end
  end
end

