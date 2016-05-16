function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Create a "distance" matrix of size (m x K) and initialize it to all zeros/or create a null matrix.
% 'm' is the number of training examples, K is the number of centroids.

centroid_distance=[];

% Use a for-loop over the 1:K centroids.
% Inside this loop, create a column vector of the distance from each training example to that
% centroid, and store it as a column of the distance matrix. 

for i=1:K

    diffs=bsxfun(@minus,X,centroids(i,:)); %make sure use all columns of centroids(i)
    distance=sum(diffs.^2,2);
    centroid_distance=[centroid_distance,distance];
end

[min_value,idx]=min(centroid_distance,[],2);

% =============================================================

end
