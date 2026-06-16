//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#ifndef NCCL_OFI_STATS_HISTOGRAM_BIN
#define NCCL_OFI_STATS_HISTOGRAM_BIN

#include <cstdint>
#include <vector>

#include "nccl_ofi_config_bottom.h"

/**
 * @brief Helper for UDTs to get the "zero" element (so bin can have a default constructor required
 * for std::vector). UDTs can specialize as required. Default implementation converts zero to the
 * target type.
 */
template <typename T> inline const T &get_zero_elem()
{
	static T zero = 0;
	return zero;
}

/**
 * @brief A histogram bin that supports safe (without overflow) average.
 * @tparam The aggregated sample type. Normally is a primitive type. UDT with overloaded arithmetic
 * and integer conversion operators would work as well.
 */
template <typename T = uint64_t> class bin {
public:
	bin(const T &lower_bound_arg = get_zero_elem<T>(),
	    const T &upper_bound_arg = get_zero_elem<T>())
	    : lower_bound(lower_bound_arg), upper_bound(upper_bound_arg), sum(0), count(0),
	      min_sample(0), max_sample(0), first_insert(true)
	{
	}

	bin(const bin &) = default;
	~bin() = default;

	/** @brief Sets the bin's lower bound (inclusive). */
	inline void set_lower_bound(const T &lower_bound_arg)
	{
		lower_bound = lower_bound_arg;
	}

	/** @brief Sets the bin's upper bound (not inclusive). */
	inline void set_upper_bound(const T &upper_bound_arg)
	{
		upper_bound = upper_bound_arg;
	}

	/** @brief Adds a sample to the bin. */
	void add_sample(const T &sample_value)
	{
		// despite this slight optimization, there is no way avoiding two if-statements here
		if (OFI_UNLIKELY(first_insert)) {
			max_sample = min_sample = sample_value;
			first_insert = false;
		} else if (sample_value > max_sample) {
			max_sample = sample_value;
		} else if (sample_value < min_sample) {
			min_sample = sample_value;
		}

		// guard against overflow
		T new_sum = sum + sample_value;
		if (new_sum < sum || count == UINT64_MAX) {
			prev_sums.push_back({ sum, count });
			sum = sample_value;
			count = 1;
		} else {
			sum = new_sum;
			count++;
		}
	}

	/** @brief Aggregates another bin into this bin. */
	void aggregate(const bin<T> &b)
	{
		// return early if bin is empty
		if (b.get_sample_count() == 0) {
			return;
		}
		prev_sums.insert(prev_sums.end(), b.prev_sums.begin(), b.prev_sums.end());
		// be careful of overflow
		T new_sum = sum + b.sum;
		uint64_t new_count = count + b.count;
		if (new_sum < sum || new_count < count) {
			prev_sums.push_back({ sum, count });
			sum = b.sum;
			count = b.count;
		} else {
			sum = new_sum;
			count = new_count;
		}

		if (first_insert) {
			min_sample = b.min_sample;
			max_sample = b.max_sample;
			first_insert = false;
		} else {
			if (b.min_sample < min_sample) {
				min_sample = b.min_sample;
			}
			if (b.max_sample > max_sample) {
				max_sample = b.max_sample;
			}
		}
	}

	/** @brief Retrieves the bin's lower bound. */
	inline const T &get_lower_bound() const
	{
		return lower_bound;
	}

	/** @brief Retrieves the bin's upper bound. */
	inline const T &get_upper_bound() const
	{
		return upper_bound;
	}

	/** @brief Retrieves the number of samples in this bin. */
	uint64_t get_sample_count() const
	{
		uint64_t total_sample_count = count;
		for (auto itr : prev_sums) {
			uint64_t curr_count = itr.second;
			total_sample_count += curr_count;
		}
		return total_sample_count;
	}

	/** @brief Retrieves the averge of the samples in this bin. */
	double get_avg() const
	{
		// do this carefully, we compute the avg in each sum/count pair seprately, then do a
		// weighted avg so we don't overflow
		double avg = 0.0f;
		double total_sample_count = count;
		for (auto itr : prev_sums) {
			uint64_t curr_count = itr.second;
			total_sample_count += curr_count;
		}
		for (auto itr : prev_sums) {
			double curr_sum = (double)itr.first;
			avg += curr_sum / total_sample_count;
		}
		// last part
		if (count > 0) {
			avg += ((double)sum) / total_sample_count;
		}
		return avg;
	}

	/** @brief Retrieves the minimum sample value in this bin. */
	inline const T &get_min_sample() const
	{
		return min_sample;
	}

	/** @brief Retrieves the maximum sample value in this bin. */
	inline const T &get_max_sample() const
	{
		return max_sample;
	}

private:
	T lower_bound;
	T upper_bound;
	T sum;
	uint64_t count;
	T min_sample;
	T max_sample;
	std::vector<std::pair<T, uint64_t>> prev_sums;
	bool first_insert;
};

#endif