%one vs one may be better? but really slow.
clear;
load('train_data');
load('trainSet_deep_PCA99');
load('testSet_deep_PCA99');
load('pairs');

% X = training_set;
% Y = training_label;

X = trainSet_deep_PCA99;
Y = training_label;
validation_set = testSet_deep_PCA99;


%t = templateSVM('Standardize',1);
%Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners','svm'); %100
%Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners',t); %84.278536
%Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners','knn');
%Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners','tree');

weight=[];
for i=1:271
    fprintf('\nTraining svm %d',i);
    %a = Mdl.BinaryLearners{i};
    %weight(i,:) = [a.Bias a.Beta'];
    sub_lable = zeros(size(training_label,1),1);
    sub_lable(find(training_label == i),1) = 1;
    cl=fitcsvm(X,sub_lable);
    weight=[weight;[cl.Bias;cl.Beta]'];
end

Mdl_predict = (weight*[ones(size(X,1),1),X]')';
Mdl_label = [];
for i=1:size(Mdl_predict,1)
    [~,Mdl_label(i,:)] = find(Mdl_predict(i,:)==max(Mdl_predict(i,:)));
end
fprintf('\nTraining Set Accuracy: %f\n', mean(double(Mdl_label == Y)) * 100);


test_set = [ones(1343,1) validation_set];
B = (weight*test_set')';
for i=1:size(B,1)
    m = min(B(i,:));
    if m < 0
        B(i,:) = B(i,:)-m;
    end
end

[possible,tmp,result] = get_by_possibility(pairs,B);

%dlmwrite('result2.csv',result,'precision',15);