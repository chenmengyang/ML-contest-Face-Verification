%one vs one may be better? but really slow.
clear;
load('train_data');
load('test_data');
load('pairs');

% X = training_set;
% Y = training_label;

null_columns = [];
for i=1:4096
  if training_set(:,i)==zeros(1393,1)
     %fprintf('%s\n',num2str(i));
     null_columns = [null_columns,i];
  end
end

X = training_set;
X(:,null_columns) = [];
Y = training_label;
validation_set(:,null_columns) = [];


t = templateSVM('Standardize',1);
Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners','svm'); %100
%Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners',t); %84.278536
%Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners','knn');
%Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners','tree');

weight=zeros(271,size(X,2)+1);
for i=1:271
    a = Mdl.BinaryLearners{i};
    weight(i,:) = [a.Bias a.Beta'];
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