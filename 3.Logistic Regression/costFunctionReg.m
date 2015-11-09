function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y); % number of training examples
    J = 0;
    grad = zeros(size(theta));
    theta1 = theta;
    theta1(1,:) = [];
    J = sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)))/m + lambda*sum(theta1.^2)/(2*m);
    for i=1:length(theta)
        if i==1
            grad(i) = sum((sigmoid(X*theta)-y).*X(:,i))/m;
        else
            grad(i) = sum((sigmoid(X*theta)-y).*X(:,i))/m + lambda*theta(i)/m;
        end
    end
end