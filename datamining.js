
var merge = function(left, right) { //mergesort helper func; data passed by reference
	result = [];
	while (left.length && right.length) { //loop until left or right are empty
		if  (left[0] <= right[0]) {
			result.push( left.shift() ); //shift - pops left most element in array
		} else {
			result.push( right.shift() );
		}
	}
	//dump remaining elements if any onto sorted list
	while (left.length) result.push( left.shift() );
	while (right.length) result.push(right.shift() );

	return result;

}, mergeSort = function(arr) {
	if (arr.length < 2) return arr; //basecase, reached end of array;

	var mid = parseInt(arr.length/2); //get center index
	var lef = arr.slice(0, mid);  //subset array from 0 to mid-1 index
	var rig = arr.slice(mid, arr.length); //subset array's other half

	return merge( mergeSort(lef), mergeSort(rig) ); //recursively sort array 
}

//syntax wrappes for getting max and min
var getMX = function(arr) {
	return Math.max(..arr);

}, getMN = function(arr) {
	return Math.min(..arr);
}

//funcs for getting local min and max
var getlmx = function(arr, mx) {
	var lmx = {v: 0, i: 0};
	 //if less than limit, and largest, located local max
	arr.forEach(function(val, dex) {
		if (val > lmx.v && val < mx) {
			lmx = {v: val, i: dex};
		}
	});
	return lmx;

}, getlmn = function(arr, mn) {
	var lmn = {v: Infinity, i: 0};
	//if greater than limit, and smallest, located local min
	arr.forEach(function(val, dex) {
		if (val > lmn.v && val < mn) {
			lmn = {v: val, i: dex};
		}
	});
	return lmn;
}

//compute derivative at each point in arr
var compDY = function(arr) {
	for (i = 0; i < arr.length-i; i++) {
		arr[i].diff = d[i+1] - d[i];
	}
}

var mean = function(arr) {
	var sum = 0.0;
	arr.forEach(function(d) { sum += d} ); //sum all values
	return sum /arr.length; //compute and return average

}, variance = function(arr, meu) {
	var veri = 0;
	arr.forEach(function(d) { veri = Math.pow(d-meu, 2); });
	return veri/arr.length;
}
