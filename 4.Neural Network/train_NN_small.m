clear;
load('small_feature');
load('train_data');
load('testSet_small');
%load('pairs');
net = patternnet(20);
num_types = 271;
train_label = zeros(271,size(training_set,1));
for i=1:size(training_set,1)
    train_label(training_label(i),i) = 1;
end
net = train(net,small_feature',train_label);
iw = net.IW;
lw = net.LW;
bias = net.b;
% func = net.trainFcn;
% y1 = net(data')';
% y = vec2ind(net(data'))';
y1 = vec2ind(net(small_feature'))';
fprintf('\nTraining Set Accuracy: %f\n', mean(double(training_label == y1)) * 100);
