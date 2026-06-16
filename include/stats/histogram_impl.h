//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#ifndef NCCL_OFI_STATS_HISTOGRAM_IMPL
#define NCCL_OFI_STATS_HISTOGRAM_IMPL

#include "histogram.h"

template <typename T> void histogram_aggregator<T>::register_histogram(base_histogram<T> *hist)
{
	std::lock_guard<std::mutex> guard(histogram_map_lock);

	auto pairib = histogram_map.insert(
		{ hist->get_description(), { nullptr, histogram_set_type() } });
	auto &histogram_agg = pairib.first->second;
	auto &histogram_set = histogram_agg.second;
	if (!histogram_set.insert(hist).second) {
		NCCL_OFI_WARN("Cannot register histogram %s: already registered",
			      hist->get_description());
	}
	if (histogram_agg.first == nullptr) {
		// NOTE: at time of clone the histogram has no sample data
		histogram_agg.first = hist->clone();
		if (histogram_agg.first == nullptr) {
			NCCL_OFI_WARN("Cannot register histogram %s: failed to clone histogram for "
				      "aggregation",
				      hist->get_description());
		}
	}
}

template <typename T> void histogram_aggregator<T>::aggregate_histogram(base_histogram<T> *hist)
{
	std::lock_guard<std::mutex> guard(histogram_map_lock);

	// get the histogram set
	auto itr = histogram_map.find(hist->get_description());
	if (itr == histogram_map.end()) {
		NCCL_OFI_WARN("Cannot aggregate histogram %s: name group unrecognized",
			      hist->get_description());
		return;
	}
	auto &histogram_agg = itr->second;

	// make sure it was registered
	auto itr2 = histogram_agg.second.find(hist);
	if (itr2 == histogram_agg.second.end()) {
		NCCL_OFI_WARN("Cannot aggregate histogram %s: not registered in set",
			      hist->get_description());
		return;
	}

	// now aggregate
	aggregate_single_histogram(histogram_agg, hist);

	// in any case it must be removed from the set now, since it might go out of scope (e.g.
	// local variable)
	histogram_agg.second.erase(itr2);

	// NOTE: for global scope objects (whose destructor will be called later, during CRT
	// shutdown sequence), we would like to avoid a second call to aggregate the object (from
	// within its destructor)
	hist->cancel_auto_aggregate();
}

template <typename T>
void histogram_aggregator<T>::print_histograms(PrintFormat print_format, bool skip_empty_bins)
{
	NCCL_OFI_TRACE(NCCL_ALL, "Aggregating all histograms");
	// go over all groups, and for each group aggregate remaining histogram, and finally print
	for (auto &kv : histogram_map) {
		auto &histogram_agg = kv.second;
		auto &histogram_set = histogram_agg.second;
		bool warn_printed = false; // print warning only once per histogram set
		while (!histogram_set.empty()) {
			auto itr = histogram_set.begin();
			auto hist = *itr;
			bool print_warning = !warn_printed;
			bool res = aggregate_single_histogram(histogram_agg, hist, print_warning);
			if (!res && !warn_printed) {
				warn_printed = true;
			}
			// remove histogram from set after being aggregated into grand-total
			// NOTE: caller is responsible for managing histogram memory/life-cycle
			histogram_set.erase(itr);

			// NOTE: for global scope objects (whose destructor will be called later,
			// during CRT shutdown sequence), we would like to avoid a second call to
			// aggregate the object (from within its destructor)
			hist->cancel_auto_aggregate();
		}
	}

	// now print aggregated histograms and release memory of aggregate histogram
	NCCL_OFI_TRACE(NCCL_ALL, "Printing all histograms");
	for (auto &kv : histogram_map) {
		auto &histogram_agg = kv.second;
		if (histogram_agg.first != nullptr) {
			histogram_agg.first->print_stats(print_format, skip_empty_bins);
			delete histogram_agg.first;
			histogram_agg.first = nullptr;
		}
	}
	histogram_map.clear();
}


template <typename T>
bool histogram_aggregator<T>::aggregate_single_histogram(histogram_aggregation_type &histogram_agg,
							 const base_histogram<T> *hist,
							 bool print_warning /* = true */)
{
	bool res = true;
	if (histogram_agg.first == nullptr) {
		// allocate aggregate histogram on-demand
		histogram_agg.first = hist->clone();
		if (histogram_agg.first == nullptr) {
			if (print_warning) {
				NCCL_OFI_WARN("Cannot aggregate histogram %s: failed to clone "
					      "first histogram for aggregation",
					      hist->get_description());
			}
			res = false;
		}
	} else {
		histogram_agg.first->aggregate(*hist);
	}
	return res;
}

/**
 * @brief Prints the histogram.
 * @param print_format The output print format.
 * @param skip_empty_bins Optioanlly specify whether empty bins should be printed or not (by
 * default yes).
 */
template <typename T>
void base_histogram<T>::print_stats(PrintFormat print_format /* = PrintFormat::TEXT */,
				    bool skip_empty_bins /* = false */,
				    histogram_formatter<T> *formatter /* = nullptr */) const
{
	// however we format the output, we must avoid printing to stdout line by line,
	// since in multi-rank use cases (as in SLURM), the output of multiple histograms
	// from different ranks will be printed together, all mixed up
	std::ostringstream oss;
	if (print_format != PrintFormat::PF_CUSTOM) {
		formatter = get_predefined_formatter<T>(print_format);
	}
	if (formatter == nullptr) {
		// deny request
		NCCL_OFI_WARN("Cannot print histograms, invalid print format or internal error");
		return;
	}
	formatter->format_histogram(*this, skip_empty_bins, oss);

	// now print histogram
	NCCL_OFI_INFO(NCCL_ALL, "Histogram [%s] data: \n%s", get_description(), oss.str().c_str());
}

#endif
