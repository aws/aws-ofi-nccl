//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#ifndef NCCL_OFI_STATS_HISTOGRAM
#define NCCL_OFI_STATS_HISTOGRAM

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <string>
#include <sstream>
#include <vector>
#include <mutex>
#include <unordered_set>
#include <unordered_map>

#include "histogram_binner.h"
#include "histogram_bin.h"
#include "histogram_def.h"
#include "histogram_formatter.h"

#include "nccl_ofi_config_bottom.h"
#include "nccl_ofi_log.h"

// entire operation of histograms is controlled by this global API
// by default it is set to true if plugin was built with --enable-histogram
// user can still set it to true if needed (as in unit tests)
extern bool histograms_enabled();
extern void enable_histograms(bool enable);

// forward declarations
template <typename T> class histogram_formatter;
template <typename T> class base_histogram;

/**
 * @brief A base interface for all histogram aggregator types (abtsract away the sample type).
 *
 * @note Histogram aggregation allows collecting the same histogram from multiple threads
 * concurrently. During plugin tear-down the data from all histograms is aggregated into one grand
 * total and printed.
 */
class base_histogram_aggregator {
public:
	base_histogram_aggregator() = default;

	/** @brief Prints all histograms aggregated by this aggregator. */
	virtual void print_histograms(PrintFormat print_format, bool skip_empty_bins) = 0;

protected:
	virtual ~base_histogram_aggregator() = default;
};

/** @brief A global manager of all histogram aggregators of all types. */
class global_histogram_aggregator {
public:
	static global_histogram_aggregator &get_instance();

	inline void register_histogram_aggregator(base_histogram_aggregator *aggregator)
	{
		std::lock_guard<std::mutex> guard(lock);
		aggregators.push_back(aggregator);
	}

	/** @brief Prints all histograms from all aggregators. */
	inline void print_all_histograms(PrintFormat print_format, bool skip_empty_bins)
	{
		load_env_override(print_format, skip_empty_bins);
		std::lock_guard<std::mutex> guard(lock);
		for (auto aggregator : aggregators) {
			aggregator->print_histograms(print_format, skip_empty_bins);
		}
	}

private:
	std::mutex lock;
	std::vector<base_histogram_aggregator *> aggregators;

	global_histogram_aggregator() = default;
	~global_histogram_aggregator() = default;

	void load_env_override(PrintFormat &print_format, bool &skip_empty_bins);
};

/**
 * @brief Aggregator of all histograms by name for a given sample type.
 *
 * @note Hsitograms with same name but different type will not be aggregated into the same
 * grand-total, but rather each will have its own separate summary print.
 */
template <typename T> class histogram_aggregator : public base_histogram_aggregator {
public:
	static histogram_aggregator<T> &get_instance()
	{
		static histogram_aggregator<T> instance;
		return instance;
	}

	/**
	 * @brief Register histogram for auto-aggregation during plugin tear-down.
	 *
	 * @note This is for supporting the use case of global-scope histograms that have not been
	 * aggregated by the time the plugin needs to shut down.
	 */
	void register_histogram(base_histogram<T> *hist);

	/**
	 * @brief Explicitly aggregates a histogram now.
	 *
	 * @note This is for supporting the case of local scope histograms that are about to be
	 * destroyed and need to be added to the total aggregation of histograms sharing the same
	 * name. This ensures it will be aggregaed now, and will not be aggregated again during
	 * plugin tear-down.
	 *
	 * @note Calling this method several times for the same histogram will have effect only for
	 * the first call, and the rest of the calls wil be ignored.
	 */
	void aggregate_histogram(base_histogram<T> *hist);

	/** @brief Prints all histograms aggregated by this aggregator. */
	void print_histograms(PrintFormat print_format, bool skip_empty_bins) override;

private:
	using histogram_set_type = std::unordered_set<base_histogram<T> *>;
	using histogram_aggregation_type = std::pair<base_histogram<T> *, histogram_set_type>;
	using histogram_map_type = std::unordered_map<std::string, histogram_aggregation_type>;
	histogram_map_type histogram_map;

	std::mutex histogram_map_lock;

	histogram_aggregator()
	{
		global_histogram_aggregator::get_instance().register_histogram_aggregator(this);
	}
	~histogram_aggregator() = default;

	bool aggregate_single_histogram(histogram_aggregation_type &histogram_agg,
					const base_histogram<T> *hist,
					bool print_warning = true);
};

/**
 * @brief Base histogram class, for common aggregation and printing.
 */
template <typename T> class base_histogram {
public:
	virtual ~base_histogram()
	{
		// aggregate this histogram
		// the following call has no effect if the histogram has already been finalized
		if (auto_aggregate) {
			histogram_aggregator<T>::get_instance().aggregate_histogram(this);
		}
	}

	inline const char *get_description() const
	{
		return description.c_str();
	}

	inline const std::string &get_units() const
	{
		return units;
	}

	inline uint64_t get_factor() const
	{
		return factor;
	}

	inline size_t get_num_bins() const
	{
		return bins.size();
	}

	inline const bin<T> &get_bin(size_t index) const
	{
		return bins[index];
	}

	/** @brief Retrieves the upper bound for the required bin (not inclusive). */
	inline const T &get_bin_upper_bound(size_t index) const
	{
		return bins[index].get_upper_bound();
	}

	/** @brief Retrieves the lower bound for the required bin. */
	inline const T &get_bin_lower_bound(size_t index) const
	{
		return bins[index].get_lower_bound();
	}

	/** @brief Aggregates another histogram into this one. */
	void aggregate(const base_histogram<T> &h)
	{
		assert(bins.size() == h.bins.size());
		for (size_t i = 0; i < bins.size(); ++i) {
			bins[i].aggregate(h.bins[i]);
		}
	}

	/**
	 * @brief Cancels auto aggregation.
	 *
	 * @note This is used by the histogram aggregator to designate that the histogram was
	 * already aggregated (and should avoid additional aggregation during destruction when
	 * aut-aggregation was scheduled).
	 */
	void cancel_auto_aggregate()
	{
		auto_aggregate = false;
	}

	/** @brief Clone this base histogram (required for aggregation). */
	virtual base_histogram<T> *clone() const = 0;

	/**
	 * @brief Prints the histogram.
	 * @param print_format The output print format.
	 * @param skip_empty_bins Optioanlly specify whether empty bins should be printed or not (by
	 * default yes).
	 * @param formatter Optional custom user-provided print formatter. Used when printFormat is
	 * set to PF_CUSTOM.
	 */
	void print_stats(PrintFormat print_format = PrintFormat::PF_TEXT,
			 bool skip_empty_bins = false,
			 histogram_formatter<T> *formatter = nullptr) const;

protected:
	std::string description;
	std::string units;
	uint64_t factor;
	std::vector<bin<T>> bins;
	bool enabled;
	bool auto_aggregate;

	base_histogram(const std::string &description_arg,
		       const char *units_arg = "",
		       uint64_t factor_arg = 1)
	    : description(description_arg), units(units_arg), factor(factor_arg), enabled(histograms_enabled()),
	      auto_aggregate(false)
	{
	}

	base_histogram(const base_histogram<T> &h)
	    : description(h.description), units(h.units), factor(h.factor), bins(h.bins),
	      enabled(h.enabled), auto_aggregate(false)
	{
		// NOTE: copy constructor sets auto-aggregate to false
	}

	/**
	 * @brief registers this histogram for auto-finalization/aggregation (in case it gets out of
	 * scope early).
	 *
	 * @note Derived classes should call this for enabling auto-aggregation, but be careful to
	 * make the call only from the bottom-most class in the class hierarchy.
	 *
	 * @see timer_histogram constructor for an example how this should be done.
	 */
	inline void schedule_auto_aggregate()
	{
		if (enabled) {
			histogram_aggregator<T>::get_instance().register_histogram(this);
			auto_aggregate = true;
		}
	}
};

/**
 * @brief A histogram for measuring data distribution.
 * @tparam BinGenerator The binner type.
 * @tparam ClockType The clock type to use (not restricted to std::chrono clocks,
 * so we can further optimize with direct TSC clock).
 *
 * @note Since we are measure only non-negative integer values, there is no motivation in using
 * a template data type, and it is fixed as uint64_t.
 */
template <typename T, typename Binner> class histogram : public base_histogram<T> {
public:
	histogram(const std::string &description_arg,
		  const Binner &binner_arg,
		  const char *units_arg = "",
		  uint64_t factor_arg = 1,
		  bool enable_auto_aggregate = true)
	    : base_histogram<T>(description_arg, units_arg, factor_arg), binner(binner_arg)
	{
		if (base_histogram<T>::enabled) {
			init_bins();
			if (enable_auto_aggregate) {
				base_histogram<T>::schedule_auto_aggregate();
			}
		}
	}

	histogram(const histogram<T, Binner> &h) : base_histogram<T>(h), binner(h.binner)
	{
		// NOTE: bins already initialized and copied data
		// NOTE: copy constructor does not schedule auto-aggregation
	}
	~histogram() override
	{
	}

	base_histogram<T> *clone() const override
	{
		return new (std::nothrow) histogram<T, Binner>(*this);
	}

	inline void insert(const T &input_val)
	{
		if (base_histogram<T>::enabled) {
			uint64_t bin_index = binner.get_bin(input_val);
			base_histogram<T>::bins[bin_index].add_sample(input_val);
		}
	}

protected:
	Binner binner;
	bool enabled;

	void init_bins()
	{
		// NOTE: bin ranges each denote the lower bound of each bin
		base_histogram<T>::bins.resize(binner.get_num_bins());
		for (size_t i = 0; i < base_histogram<T>::bins.size(); ++i) {
			base_histogram<T>::bins[i].set_lower_bound(binner.get_bin_ranges()[i]);
			if (i + 1 < binner.get_num_bins()) {
				base_histogram<T>::bins[i].set_upper_bound(
					binner.get_bin_ranges()[i + 1]);
			} else {
				base_histogram<T>::bins[i].set_upper_bound(
					binner.get_bin_ranges().back() + binner.get_bin_size());
			}
		}
	}
};

/**
 * @brief A histogram for measuring latency distribution.
 * @tparam BinGenerator The binner type.
 * @tparam ClockType The clock type to use (not restricted to std::chrono clocks,
 * so we can further optimize with direct TSC clock).
 *
 * @note Since we are measure only non-negative integer values, there is no motivation in using
 * a template data type, and it is fixed as uint64_t.
 */
template <typename Binner, typename clock = std::chrono::steady_clock, typename T = std::size_t>
class timer_histogram : public histogram<T, Binner> {
public:
	using rep = T;

	timer_histogram(const std::string &description_arg,
			const Binner &binner_arg,
			const char *units_arg = "",
			uint64_t factor_arg = 1,
			bool enable_auto_aggregate = true)
	    : histogram<T, Binner>(description_arg, binner_arg, units_arg, factor_arg, false)
	{
		// NOTE: this is a bit tricky, since auto-registration takes place at constructor,
		// but the histogram aggregator needs to call a virtual function, so only the
		// bottom-most clas in the class hierarchy can call the reigstration function (when
		// the virtual function table is ready), otherwise we get pure virtual function call
		// exception.
		// For this reason, false was passed to parent class for enable_auto_aggregate, to
		// avoid registration before the virtual function table is ready.
		if (enable_auto_aggregate) {
			base_histogram<T>::schedule_auto_aggregate();
		}
	}

	timer_histogram(const timer_histogram<Binner, clock, T> &h) : histogram<T, Binner>(h)
	{
	}
	~timer_histogram() override
	{
	}

	base_histogram<T> *clone() const override
	{
		return new (std::nothrow) timer_histogram<Binner, clock, T>(*this);
	}

	/** @brief Marks start of latency measurement. */
	inline void start_timer()
	{
		if (base_histogram<T>::enabled) {
			start_time = clock::now();
			// add compiler fence to ensure start timestamp is taken before measured
			// activity
			std::atomic_signal_fence(std::memory_order_seq_cst);
		}
	}

	/** @brief Marks end of latency measurement. */
	inline uint64_t stop_timer()
	{
		if (base_histogram<T>::enabled) {
			// add compiler fence to ensure end timestamp is taken after measured
			// activity
			std::atomic_signal_fence(std::memory_order_seq_cst);
			auto now = clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
				now - start_time);
			histogram<T, Binner>::insert(duration.count());

			// NOTE: this is returned only for sake of unit tests, so we return it in
			// microseconds as the unit tests expect, but the sample was added as nanoseconds
			return std::chrono::duration_cast<std::chrono::microseconds>(
				now - start_time).count();
		}
		return 0;
	}

protected:
	typename clock::time_point start_time;
};

/**
 * @brief Prints all histograms. Should be normally called during shutdown.
 * @param print_format The histogram print format.
 * @param skip_empty_bins Specifies whether empty bins should be printed or not.
 *
 * @note These values are overridable by the following environment variables:
 * - OFI_HISTOGRAM_PRINT_FORMAT (string): text, table, csv, json
 * - OFI_HISTOGRAM_SKIP_EMPTY_BINS (boolean)
 */
inline void print_all_histograms(PrintFormat print_format = PrintFormat::PF_TABLE,
				 bool skip_empty_bins = false)
{
	if (histograms_enabled()) {
		global_histogram_aggregator::get_instance().print_all_histograms(print_format,
										skip_empty_bins);
	}
}

/**
 * @brief A scoped histogram object RAII wrapper.
 * @tparam A pointer type to the histogram.
 */
template <typename T> class scoped_histogram {
public:
	scoped_histogram(T &histogram) : m_histogram(histogram)
	{
		m_histogram.start_timer();
	}
	~scoped_histogram()
	{
		m_histogram.stop_timer();
	}

private:
	T &m_histogram;
};

#endif

// include inline histogram implementation
#include "histogram_formatter_impl.h"
#include "histogram_impl.h"
