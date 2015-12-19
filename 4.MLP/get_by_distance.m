function [ distances,result ] = get_by_distance(pairs,testSet)
%GET_BY_DISTANCE Summary of this function goes here
%   Detailed explanation goes here
distances = [];
for i=1:size(pairs,1)
    %fprintf('Calculating distance for pair%d\n',i);
    photo1 = testSet(pairs(i,1),:);
    photo2 = testSet(pairs(i,2),:);
    distances(i,:) = sum((photo1-photo2).^2).^0.5;
end

%Find largest distance
maxDistance = max(distances);
result = [];
for i=1:length(distances)
    %fprintf('Calculating result possibility for pair%d\n',i);
    result(i,:) = [i-1,(maxDistance-distances(i,:))/maxDistance];
end

end

