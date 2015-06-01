function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for l = 1:numel(stack)
    stackgrad{l}.w = zeros(size(stack{l}.w));
    stackgrad{l}.b = zeros(size(stack{l}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% stack{1}  w: [200x784 double], b: [200x1 double]
% stack{2}  w: [200x200 double], b: [200x1 double]

n = numel(stack); % number of hidden layers

z = cell(n+1, 1);
a = cell(n+1, 1);
a{1} = data;

% forward prop
for l=1:n
  z{l+1} = bsxfun(@plus, stack{l}.w * a{l}, stack{l}.b);
  a{l+1} = sigmoid(z{l+1});
end


% softmax output
S = softmaxTheta*a{n+1};
S = exp(bsxfun(@minus, S, max(S)));
p = bsxfun(@rdivide, S, sum(S));   

cost = -1/M * sum(sum((log(p).*groundTruth))) + sum(sum((softmaxTheta.^2))) * lambda / 2;

softmaxThetaGrad = -1/M * (groundTruth - p) * a{n+1}' + lambda*softmaxTheta;

% backprop error gradients
d = cell(n+1,1);
d{n+1} = -(softmaxTheta' * (groundTruth - p)) .* sigmPrime(a{n+1});

% compute the error deltas
for l=n:-1:2
  d{l} = stack{l}.w' * d{l+1} .* sigmPrime(a{l});
end

% compute the partial derivatives
for l=n:-1:1
  stackgrad{l}.w = d{l+1} * a{l}' / M;
  stackgrad{l}.b = sum(d{l+1}, 2) / M; 
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

% Return derivitive of sigm(z)
function sigmp = sigmPrime(z)
  sigmp = z.*(1-z);
end
