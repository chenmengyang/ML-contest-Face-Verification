clear;
load('train_data');
load('small_feature');
load('testSet_small');
load('pairs');
k = 8;
label = knnsearch(small_feature,small_testSet,'dist','cityblock','k',k);
kClasses = [];
for i=1:size(label,1)
    kClasses(i,:) = training_label(label(i,:),:)';
end

match = [];
for i=1:size(pairs,1)
    fprintf('Calculating result possibility for pair%d\n',i);
    rate = 0;
    pic1 = kClasses(pairs(i,1),:);
    pic2 = kClasses(pairs(i,2),:);
    numbers = intersect(pic1,pic2);
    for j=1:length(numbers)
        rate = rate + (histc(pic1,numbers(j))/k) * (histc(pic2,numbers(j))/k);
    end
    match(i,:) = [i-1,rate];
end

%xlswrite('KNN_result.txt',match);
dlmwrite('KNN_result2.csv',match,'precision',10);
