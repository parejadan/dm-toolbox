% Task 
% Charlie wants to buy a house. He does a detailed survey of the area where he wants to live, in which he quantifies, normalizes, and maps the desirable features of houses to values on a scale of 00 to 11 so the data can be assembled into a table. If Charlie noted FF features, each row contains FF space-separated values followed by the house price in dollars per square foot (making for a total of F+1F+1 columns). If Charlie makes observations about HH houses, his observation table has HH rows. This means that the table has a total of (F+1)×H(F+1)×H entries.

% Unfortunately, he was only able to get the price per square foot for certain houses and thus needs your help estimating the prices of the rest! Given the feature and pricing data for a set of houses, help Charlie estimate the price per square foot of the houses for which he has compiled feature data but no pricing.

% Important Observation: The prices per square foot form an approximately linear function for the features quantified in Charlie's table. For the purposes of prediction, you need to figure out this linear function.

% Recommended Technique: Use a regression-based technique. At this point, you are not expected to account for bias and variance trade-offs.

% Input Format

% The first line contains 22 space-separated integers, FF (the number of observed features) and NN (the number of rows/houses for which Charlie has noted both the features and price per square foot). 
% The NN subsequent lines each contain F+1F+1 space-separated integers describing a row in the table; the first FF elements are the noted features for a house, and the very last element is its price per square foot.

% The next line (following the table) contains a single integer, TT, denoting the number of houses for for which Charlie noted features but does not know the price per square foot. 
% The TT subsequent lines each contain FF space-separated integers describing the features of a house for which pricing is not known.


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