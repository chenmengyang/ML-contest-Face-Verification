%% Neural Network Learning

%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading Data =============
load('train_data');
load('test_data');
load('pairs');

null_columns = [];
for i=1:4096
  if training_set(:,i)==zeros(1393,1)
     %fprintf('%s\n',num2str(i));
     null_columns = [null_columns,i];
  end
end

X = training_set;
X(:,null_columns) = [];
y = training_label;
validation_set(:,null_columns) = [];

%% Setup the parameters you will use for this exercise
input_layer_size  = 4063;  % 4063 Input Images of Digits
hidden_layer_size = 100;   % 100 hidden units
num_labels = 271;          % 271 labels, from 1 to 271   

%% ================ Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =============== Backpropagation ===============
fprintf('\nChecking Backpropagation... \n');
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =================== Training NN ===================
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 200);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Predict =================

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% ================= get final result =================
m = size(validation_set, 1);
num_labels = size(Theta2, 1);
h1 = sigmoid([ones(m, 1) validation_set] * Theta1');
% Here get the possibilities for each photo in testset to be 1-271 class
h2 = sigmoid([ones(m, 1) h1] * Theta2');
testSet_prossibility = h2;

%calculate the result by distance or possibility.
%[distances,result] = get_by_distance(pairs,testSet_prossibility);
[possible,~,result] = get_by_possibility(pairs,testSet_prossibility);

dlmwrite('result2.csv',result,'precision',15);