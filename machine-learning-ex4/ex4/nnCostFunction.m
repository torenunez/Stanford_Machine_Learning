function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% Convert the label numbers in 'y' into a matrix 'Y' with a boolean vector with 
% the corresponding label position for the class 'k' from 'K' for each  
% training example 'm'
K = num_labels %total number of classes
Y = eye(K)(y, :);

% layer 1
a1 = [ones(m, 1), X]; %add bias unit to input node

% layer 2
z2 = a1 * Theta1'; % z(j)=Θ(j−1)a(j−1)
a2 = sigmoid(z2); % a(j)=g(z(j))
a2 = [ones(size(a2, 1), 1), a2]; %add bias unit 

% layer 3
z3 = a2 * Theta2';
a3 = sigmoid(z3); %results in a 5000x10 matrix

% Cost (sum over all i in m and all k in K)
cost = sum((-Y .* log(a3)) - ((1 - Y) .* log(1 - a3)), 2); % all k in K
J = (1 / m) * sum(cost); % all i in m

% ===============
% Regularization: λ2m∑l=1L−1∑i=1sl∑j=1sl+1(Θ(l)j,i)2
% ===============
Theta1NoBias = Theta1(:, 2:end); %exclude the bias in Theta1
Theta2NoBias = Theta2(:, 2:end); %exclude the bais in Theta2

%  sumsq is equivalent to sum(term .^ 2);
% http://octave.sourceforge.net/octave/function/sumsq.html
Reg  = (lambda / (2 * m)) * (sum(sumsq(Theta1NoBias)) + sum(sumsq(Theta2NoBias)));
J = J + Reg;



%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


% Initialize accumulators
Delta1 = 0;
Delta2 = 0;

for i = 1:m
	% Loop forward propagation for i in m
	a1 = [1; X(i, :)']; %bias
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)]; %bias
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	% Calculate deltas
	delta3 = a3 - Y(i, :)'; %hypothesis minus label
	% https://share.coursera.org/wiki/index.php/ML:Neural_Networks:_Learning
	delta2 = (Theta2NoBias' * delta3) .* sigmoidGradient(z2); %δ(l)=((Θ(l))Tδ(l+1)) .∗ g′(z(l))

	% Accumulate deltas
	Delta2 = Delta2 + (delta3 * a2');
	Delta1 = Delta1 + (delta2 * a1');

endfor

% Calculate theta gradients
Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

% Roll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% derivative of the regulaization component in the cost function
% 2s cancel (squared and lambda/2m)
Theta1_grad(:, 2:end) += ((lambda / m) * Theta1NoBias);
Theta2_grad(:, 2:end) += ((lambda / m) * Theta2NoBias);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
