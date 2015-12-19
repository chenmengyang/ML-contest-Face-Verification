function [ possible,tmp,result ] = get_by_possibility(pairs,testSet)
%GET_BY_POSSIBILITY Summary of this function goes here
%   Detailed explanation goes here
possible = testSet;
for i=1:size(possible,1)
    possible(i,:) = possible(i,:)/sum(possible(i,:));
end

tmp = [];
for i=1:size(pairs,1)
    %fprintf('Calculating for pair%d\n',i);
    photo1 = possible(pairs(i,1),:);
    photo2 = possible(pairs(i,2),:);
    tmp(i,:) = [i-1,photo1*photo2'];
end

%Find largest distance
result = tmp;
maxPossible = max(result(:,2));
result(:,2) = result(:,2)/maxPossible;

end

