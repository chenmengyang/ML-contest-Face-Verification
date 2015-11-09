%one vs one may be better? but really slow. 2.5 hours for this code.
clear;
load('train_data');
load('trainSet_deep_PCA99');
load('testSet_deep_PCA99');
load('pairs');

X = [ones(size(trainSet_deep_PCA99,1),1),trainSet_deep_PCA99];
Y = training_label;
test_set = [ones(size(testSet_deep_PCA99,1),1),testSet_deep_PCA99];

t = templateSVM('Standardize',1);

weight = [];
which_two = [];
for i=1:271
    for j=(i+1):271
        fprintf('\nTraining %d , %d\n',i,j);
        ind=[find(training_label==i);find(training_label==j)];
        cl=fitcsvm(trainSet_deep_PCA99(ind,:),training_label(ind,:));
        weight=[weight,[cl.Bias;cl.Beta]];
        which_two = [which_two;[i,j]];
    end
end

predict = X*weight;
votes = zeros(size(predict,1),271);
for i=1:size(predict,1)
    for j=1:size(which_two,1)
        if predict(i,j) >=0
            votes(i,which_two(j,2)) = votes(i,which_two(j,2)) + 1;
        else
            votes(i,which_two(j,1)) = votes(i,which_two(j,1)) + 1;
        end
    end
end

[~,vote_result]=max(votes,[],2);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(vote_result == Y)) * 100);

test_predict = test_set*weight;
test_votes = zeros(size(test_predict,1),271);
for i=1:size(test_predict,1)
    for j=1:size(which_two,1)
        if test_predict(i,j) >=0
            test_votes(i,which_two(j,2)) = test_votes(i,which_two(j,2)) + 1;
        else
            test_votes(i,which_two(j,1)) = test_votes(i,which_two(j,1)) + 1;
        end
    end
end

[possible,tmp,result] = get_by_possibility(pairs,test_votes);

dlmwrite('result4.csv',result,'precision',20);