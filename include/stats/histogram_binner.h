//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#ifndef NCCL_OFI_STATS_HISTOGRAM_BINNER
#define NCCL_OFI_STATS_HISTOGRAM_BINNER

#include <cassert>
#include <vector>


//
// A linear binner creates `num_bins_arg` bins, each of size `bin_size_arg`
// where the first bin's range is `min_val_arg` to `min_val_arg + bin_size_arg -
// 1`.
//
template <typename T>
class histogram_linear_binner {
public:
	histogram_linear_binner(const T& min_val_arg, const T& bin_size_arg,
				std::size_t num_bins_arg)
		: min_val(min_val_arg),bin_size(bin_size_arg), num_bins(num_bins_arg)
	{
	}


	std::size_t get_bin(const T& input_val)
	{
		assert(input_val >= min_val);
		std::size_t bin =  (input_val - min_val) / bin_size;
		if (bin >= num_bins) {
			bin = num_bins - 1;
		}
		return bin;
	}


	std::size_t get_num_bins(void) const
	{
		return num_bins;
	}


	const std::vector<T> & get_bin_ranges(void)
	{
		if (range_labels.size() == 0) {
			for (std::size_t i = 0 ; i < num_bins ; ++i) {
				T val = min_val + (i * bin_size);
				range_labels.insert(range_labels.end(), val);
			}
		}

		return range_labels;
	}

protected:
	const T min_val;
	const T bin_size;

	const std::size_t num_bins;

	std::vector<T> range_labels;
};


//
// Flexible binner where the user provides the list of starting points of the
// bin.  Slowest binner, since a linear search is currently used to find the
// right bin.  This could be made log(n), but even that will be considerably
// slower than the direct computation used in the linear binner.
//
template <typename T>
class histogram_custom_binner {
public:
	histogram_custom_binner(const std::vector<T> &ranges_arg)
		: ranges(ranges_arg)
	{
	}


	std::size_t get_bin(const T& input_val)
	{
		std::size_t ret = 0;

		assert(input_val >= ranges[0]);
		for (auto i  = ranges.begin() + 1 ; i != ranges.end() ; ++i) {
			if (*i > input_val) {
				break;
			}
			ret++;
		}

		return ret;
	}


	std::size_t get_num_bins(void) const
	{
		return ranges.size();
	}


	const std::vector<std::size_t> & get_bin_ranges(void)
	{
		return ranges;
	}

protected:
	std::vector<T> ranges;
};

#endif
