% The Problem

% Charlie wants to purchase office-space. He does a detailed survey of the offices and corporate complexes in the area, and tries to quantify a lot of factors, such as the distance of the offices from residential and other commercial areas, schools and workplaces; the reputation of the construction companies and builders involved in constructing the apartments; the distance of the offices from highways, freeways and important roads; the facilities around the office space and so on.

% Each of these factors are quantified, normalized and mapped to values on a scale of 0 to 1. Charlie then makes a table. Each row in the table corresponds to Charlie's observations for a particular house. If Charlie has observed and noted F features, the row contains F values separated by a single space, followed by the office-space price in dollars/square-foot. If Charlie makes observations for H houses, his observation table has (F+1) columns and H rows, and a total of (F+1) * H entries.

% Charlie does several such surveys and provides you with the tabulated data. At the end of these tables are some rows which have just F columns (the price per square foot is missing). Your task is to predict these prices. F can be any integer number between 1 and 5, both inclusive.

% There is one important observation which Charlie has made.

% The prices per square foot, are (approximately) a polynomial function of the features in the observation table. This polynomial always has an order less than 4
% Input Format

% The first line contains two space separated integers, F and N. Over here, F is the number of observed features. N is the number of rows for which features as well as price per square-foot have been noted. 
% This is followed by a table having F+1 columns and N rows with each row in a new line and each column separated by a single space. The last column is the price per square foot.

% The table is immediately followed by integer T followed by T rows containing F columns.


function  [X_norm, mu, sigma] = featureNormalize(X)
	% returns a normalized version of X where
	% the mean value of each feature is 0 and SD 1.
	% This is usually a good preprocessing step to
	% perform when working with learning algorithms
	% ------------------------------------------------
	% size() returns vector with dimensions of a matrix
	% repmat() extends a vector by repeating it a specified length
	
	% initialize variables
	X_norm = X;
	mu = zeros( 1, size(X, 2) );
	sigma = zeros( 1, size(X, 2) );
	% get mean and sd from X
	mu = mean(X);
	sigma = std(X);
	% basically computing Z-score for each cell on the X matrix 
	X_norm = (X - repmat(mu, length(X), 1) ) ./ repmat(sigma, length(X), 1);
end

function J = computeCost(X, y, theta)
	% compute cost for linear regression
	
	m = length(y); % number of training examples
	J = 0; % variable to return
	
	hyp = X * theta; % get hypothesis
	squaredErrors = (hyp - y).^2;
	%compute cost for a particular theta
	J = (1 / (2 * m) ) * sum(squaredErrors);
end

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
	% function learns the corresponding theta for the given training set
	% by taking num_iters gradient steps with learning rate alpha
	
	m = length(y); % number of training examples
	J_history = zeros(num_iters, 1);
	
	for iter = 1:num_iters
	
		hyp = X * theta; %get hypothesis from X
		theta -= alpha * (1/m) * (X' * (hyp-y) );
		
		% save the cost J in every iteration (shows how residual decreases)
		J_history(iter) = computeCost(X, y, theta);
	end
end

% ========================= GET INPUT VALUES =========================
% column by row, NOTE: octave does row by bolumn
function [input] = getInput(prompt, delim)
	% there is probably a better way to do this but:
	% get a single line of input and process it to numbers
	rawIn = input(prompt, 's');
	prosLine = strsplit(rawIn, delim);
	len = length(prosLine);
	
	input = zeros(1, len);
	for iter = 1:len
		input(iter) = str2num( cell2mat( prosLine(iter) ) );
	end
end

function [F, M, dat1, N, dat2] = doInput()
	fin = getInput('', ' ');
	F = fin(1); M = fin(2);

	% get sample training data
	dat1 = zeros(M, F+1);
	for iter = 1:M
		dat1(iter,:) = getInput('', ' ');
	end

	% get data to predict values for 
	N = getInput('', ' ');
	dat2 = zeros(N, F);
	for iter = 1:N
		dat2(iter,:) = getInput('', ' ');
	end
end
% ========================= PREDICTION CODE HERE =====================
function predict(F, M, dat1, N, dat2)
	X = dat1(:, 1:F); % get training features
	y = dat1(:, F+1); % get output for training features

	%[X mu sigma] = featureNormalize(X); % normalize features only if not already
	X = [ ones(M, 1) X ]; % add intercept term to X_norm


	alpha = 0.1; % choose some alhpa value
	num_iters = 400; % decide on steps for gradient descent 
	theta = zeros(F+1, 1); % start with zero theta values

	% get learned thetas
	[theta J_history] = gradientDescent(X, y, theta, alpha, num_iters);

	dat2 = [ones(N, 1) dat2];
	
	% predict prices
	_prices = (dat2 * theta);
	len = length(_prices);
	for iter = 1:len
		disp( _prices(iter) );
	end
end

[F, M, dat1, N, dat2] = doInput();
predict(F, M, dat1, N, dat2);