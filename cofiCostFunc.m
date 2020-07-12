function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%



%summation = 0;
% Nested for loop accumulating cost on the condition that R(i,j) = 1 (unvectorized)
%for i = 1:num_movies
%    for j = 1:num_users
%        if R(i,j) == 1
%            summation = summation + (Theta(j,:) * X(i,:)' - Y(i,j)).^2 ;
%        end
%    end
%end

%J = 1/2 * summation;


% Vectorized implementation using R .* for element-wise multiplication to preserve 1's
J = (1/2) * sum(sum( R .* ((Theta * X')' - Y).^2 )) + lambda / 2 * (sum(sum(Theta.^2)) + sum(sum(X.^2)));

% X_grad is movies x features
% Theta_grad is users x features

% Compute X_grad by filtering
for i = 1:num_movies
    idx = find(R(i,:) == 1);        % Returns indices of values in R that are 1
    Theta_temp = Theta(idx,:);      % Chooses the first size(idx,2) rows of Theta of first idx users
    Y_temp = Y(i,idx);              % Chooses first idx elements of the ith row of Y
    X_grad(i,:) = (X(i,:) * Theta_temp' - Y_temp) * Theta_temp + lambda * X(i,:);  % Computes gradient
end     

% Compute Theta_grad by filtering
for j = 1:num_users
    idx = find(R(:,j) == 1);
    X_temp = X(idx,:);
    Y_temp = Y(idx,j);
    Theta_grad(j,:) = (X_temp * Theta(j,:)' - Y_temp)' * X_temp + lambda * Theta(j,:);
end




% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
