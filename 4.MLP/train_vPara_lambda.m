clear;
load('5Folds.mat');
statistics = [];
% t = [-4:0.2:1]';
% lambda = exp(t);
lambda = [2.5,2.75,3.25,3.5];
for l=1:length(lambda)
    fprintf('**********%d',l);
    for i=1:length(kFold)
        statistics(l,i)=train_validation(kFold{i},lambda(l),100,100);
    end
end

% for i=1:length(kFold)
%     aaa1=train_validation(kFold{i},10,100,100);
% end