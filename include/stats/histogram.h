//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#ifndef NCCL_OFI_STATS_HISTOGRAM
#define NCCL_OFI_STATS_HISTOGRAM

#include <cassert>
#include <chrono>
#include <cstddef>
#include <string>
#include <sstream>
#include <vector>

#include "nccl_ofi_log.h"
#include "histogram_binner.h"


//
// Base histogram class.  Histograms are a lightweight mechanism for tracking
// events occurances in code and are used for instrumenting the plugin code.
//
// T is the type of the data that will be inserted into the histogram.  Any POD
//  will work, and little effort has been put into making the interface safe for
//  non-Pods.
//
template <typename T, typename Binner>
class histogram {
public:
	histogram(const std::string& description_arg, Binner binner_arg)
		: description(description_arg), binner(binner_arg),
		bins(binner.get_num_bins()), num_samples(0), first_insert(true)
	{
	}

	void insert(const T& input_val)
	{
		if (OFI_UNLIKELY(first_insert)) {
			max_val = min_val = input_val;
			first_insert = false;
		}

		if (input_val > max_val) {
			max_val = input_val;
		} else if (input_val < min_val) {
			min_val = input_val;
		}

		bins[binner.get_bin(input_val)]++;
		num_samples++;
	}

	void print_stats(void) {
		auto range_labels = binner.get_bin_ranges();

		NCCL_OFI_INFO(NCCL_NET, "histogram %s", description.c_str());
		NCCL_OFI_INFO(NCCL_NET, "  min: %ld, max: %ld, num_samples: %lu",
					(long int)min_val, (long int)max_val, num_samples);
		for (size_t i = 0 ; i < bins.size() ; ++i) {
			std::stringstream ss;
			ss << "    " << range_labels[i] << " - ";
			if (i + 1 != bins.size()) {
				ss << range_labels[i + 1] - 1;
			} else {
				ss << "    ";
			}
			ss  << "    " << bins[i];
			NCCL_OFI_INFO(NCCL_NET, "%s", ss.str().c_str());
		}
	}

protected:
	std::string description;
	Binner binner;
	std::vector<std::size_t> bins;
	T max_val;
	T min_val;
	std::size_t num_samples;
	bool first_insert;
};


//
// Histogram class for tracking intervals.  A timer_histogram class can only
// track one interval at a time, and will auto-insert the result when
// stop_timer() is called.  Times are recorded in microseconds.
//
// T is the type of the data that will be inserted into the histogram.  Any POD
//  will work, and little effort has been put into making the interface safe for
//  non-Pods.
template <typename Binner, typename clock = std::chrono::steady_clock, typename T  = std::size_t>
class timer_histogram : public histogram<T, Binner> {
public:
	using rep = T;
	using histogram<T, Binner>::insert;

	timer_histogram(const std::string &description_arg, Binner binner_arg)
		: histogram<T, Binner>(description_arg, binner_arg)
	{
	}

	void start_timer(void)
	{
		start_time = clock::now();
		asm volatile ("" : : : "memory");
	}

	rep stop_timer(void)
	{
		asm volatile ("" : : : "memory");
		auto now = clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time);
		insert(duration.count());
		return duration.count();
	}


protected:
	typename clock::time_point start_time;
};

#endif
