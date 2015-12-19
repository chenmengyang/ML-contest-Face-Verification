function [ accuracy ] = train_validation( D,lambda,hSize,iterations)
    % TRAIN_VALIDATION Summary of this function goes here
    % Detailed explanation goes here
    % X,Y,V,L
    X = D{1};
    y = D{2};
    V = D{3};
    L = D{4};
    validation_set = V;
    input_layer_size  = size(X,2);  % 1015 Input Images of Digits
    hidden_layer_size = hSize;      % 100 hidden units
    num_labels = 271;               % 271 labels, from 1 to 271
    % Initializing Pameters
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    options = optimset('MaxIter', iterations);
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X, y, lambda);
    [nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    pred = predict(Theta1, Theta2, validation_set);
    accuracy = mean(double(pred == L))*100;
end

