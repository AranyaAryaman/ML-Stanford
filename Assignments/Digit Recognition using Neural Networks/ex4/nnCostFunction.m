function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


      % forward propogation

      X = [ones(m,1) X];
      a1 = X;
      z2=Theta1 * X'
      a2 = sigmoid(z2);
      a2=a2';
      a2 = [ones(m,1) a2];
      z3 = Theta2 * a2';
      a3 = sigmoid(z3);
      a3 = a3';
      
      %recoding y's into vectors
      y_vec=zeros(m,num_labels);
      for i=1:m
        y_vec(i,y(i))=1;
      endfor
      
 %two time sum is not required, it is easy to realise that when a3*y_vec is 
 %done,the diagonal elements are the required terms for us. So, we can easily 
 %sum them
      J = -1/m *  (sum(diag(log(a3)*y_vec')) + sum(diag(log(1-a3)*(1-y_vec)')));
      
     %The above is a highly precise vectored approach to find the cost.
     
     % Regularisation of Cost
     sum1=0;
     for j=1:size(Theta1,1)
       for k=2:size(Theta1,2)     % since the program should work for any theta
         sum1+=Theta1(j,k)^2;
       endfor
     endfor
     
     sum2=0;
     for j=1:size(Theta2,1)     % 1 gives row size, 2 gives column size
       for k=2:size(Theta2,2)
         sum2+=Theta2(j,k)^2;
       endfor
     endfor

     J+= (lambda*(sum1+sum2))/(2*m);
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

    
  
    for t=1:m
      A1 = a1(t,:)';
      A3 = a3(t,:)';
      d3 = A3-y_vec(t,:)';
      A2 = (Theta2' * d3);
      A2 = A2(2:end);
      d2 = A2 .* sigmoidGradient(z2(:,t));
      Theta2_grad += d3 * a2(t,:);
      Theta1_grad += d2 * a1(t,:);
    endfor
    
      Theta1_grad = Theta1_grad/m;
      Theta2_grad = Theta2_grad/m;
      
      R1 = lambda * Theta1(:,2:end)/m;
      R2 = lambda * Theta2(:,2:end)/m;
      R1 = [zeros(size(Theta1,1),1) R1];
      R2 = [zeros(size(Theta2,1),1) R2]; 
      Theta1_grad += R1;
      Theta2_grad += R2;
      
      

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

      

    















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
