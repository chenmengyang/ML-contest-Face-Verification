function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

Z = zeros(size(X, 1), K);

for i=1:size(X, 1)
    x = X(i, :)';
    Z(i,:) = x' * U(:, 1:K);
end
% =============================================================

end