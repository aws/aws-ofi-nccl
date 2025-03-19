//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#include "config.h"

#include <iostream>

#include "stats/histogram_binner.h"


#define CHECK_AND_EXIT(x)				      \
	if (!(x)) {					      \
		std::cerr << "Failure: " << #x << std::endl; \
		exit(1);				      \
	}


static void check_linear(void)
{
	histogram_linear_binner<std::size_t> linear(0, 10, 10);
	CHECK_AND_EXIT(linear.get_num_bins() == 10);
	CHECK_AND_EXIT(linear.get_bin(0) == 0);
	CHECK_AND_EXIT(linear.get_bin(1) == 0);
	CHECK_AND_EXIT(linear.get_bin(10) == 1);
	CHECK_AND_EXIT(linear.get_bin(99) == 9);
	CHECK_AND_EXIT(linear.get_bin(100) == 9);
	CHECK_AND_EXIT(linear.get_bin(1000) == 9);

	auto linear_labels = linear.get_bin_ranges();
	CHECK_AND_EXIT(linear_labels.size() == linear.get_num_bins());
	CHECK_AND_EXIT(linear_labels[0] == 0);
	CHECK_AND_EXIT(linear_labels[1] == 10);
	CHECK_AND_EXIT(linear_labels[9] == 90);
}


static void check_custom(void)
{
	std::vector<int> input_sizes(5);
	input_sizes[0] = -100;
	input_sizes[1] = 0;
	input_sizes[2] = 50;
	input_sizes[3] = 1000;
	input_sizes[4] = 10000;
	histogram_custom_binner<int> custom(input_sizes);
	CHECK_AND_EXIT(custom.get_num_bins() == 5);
	CHECK_AND_EXIT(custom.get_bin(-10) == 0);
	CHECK_AND_EXIT(custom.get_bin(-1) == 0);
	CHECK_AND_EXIT(custom.get_bin(0) == 1);
	CHECK_AND_EXIT(custom.get_bin(10) == 1);
	CHECK_AND_EXIT(custom.get_bin(49) == 1);
	CHECK_AND_EXIT(custom.get_bin(50) == 2);
	CHECK_AND_EXIT(custom.get_bin(51) == 2);
	CHECK_AND_EXIT(custom.get_bin(999) == 2);
	CHECK_AND_EXIT(custom.get_bin(1000) == 3);
	CHECK_AND_EXIT(custom.get_bin(1000000) == 4);
}


int
main(int argc, char *argv[])
{
	check_linear();
	check_custom();

	return 0;
}
