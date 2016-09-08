function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    x = X(:,2); %subset the second column for the data only

    % https://share.coursera.org/wiki/index.php/ML:Linear_Regression_with_One_Variable
    % yhat = h(x) = theta0 + theta1 * x (i.e. y = mx + b)
    h = (theta(2)*x) + theta(1);

    % replace partial derivatife function
    % https://math.stackexchange.com/questions/70728/partial-derivative-in-gradient-descent-for-two-variables/189792#189792
    
    % remember that theta0 is index #1
    theta0 = theta(1) - alpha * (1/m) * sum(h-y); %rename 
    % theta1 is index #2
    theta1  = theta(2) - alpha * (1/m) * sum((h - y) .* x); %rename

    % https://share.coursera.org/wiki/index.php/ML:Gradient_Descent
    % repeat until convergence
    theta = [theta0; theta1]; %update in one single step

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    end % loop end

end % function end
