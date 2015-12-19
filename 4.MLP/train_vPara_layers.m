clear;
load('5Folds.mat');
statistics = [];
% t = [-4:0.2:1]';
% lambda = exp(t);
layers = [275,300,350];
for l=1:length(layers)
    fprintf('**********%d',l);
    for i=1:length(kFold)
        statistics(l,i)=train_validation(kFold{i},1,layers(l),100);
    end
end

% for i=1:length(kFold)
%     aaa1=train_validation(kFold{i},10,100,100);
% end