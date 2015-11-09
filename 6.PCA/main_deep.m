clear;
load('train_data');
load('test_data');

%  Before running PCA, first normalize
%  Before running normalization, eliminate all-zeros' features
null_columns = [];
for i=1:size(training_set,2)
  if training_set(:,i)==zeros(size(training_set,1),1)
     null_columns = [null_columns,i];
  end
end

X = training_set;
X(:,null_columns) = [];
Y = training_label;
validation_set(:,null_columns) = [];

[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

%  Find the suitable value of k
K = findK(S,0.99);
Z = projectData(X_norm, U, K);

%  Map validation_set to k dimensions
[V_norm, mu, sigma] = featureNormalize(validation_set);
V = projectData(V_norm, U, K);