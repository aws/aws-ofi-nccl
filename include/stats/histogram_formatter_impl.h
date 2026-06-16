//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#ifndef NCCL_OFI_STATS_HISTOGRAM_FORMATTER_IMPL
#define NCCL_OFI_STATS_HISTOGRAM_FORMATTER_IMPL

#include <iomanip>

template <typename T>
void text_formatter<T>::format_histogram(const base_histogram<T> &h,
					 bool skip_empty_bins,
					 std::ostream &os)
{
	for (size_t i = 0; i < h.get_num_bins(); ++i) {
		const bin<T> &b = h.get_bin(i);
		if (skip_empty_bins && b.get_sample_count() == 0) {
			continue;
		}
		os << "Bin " << i << " [" << h.get_bin_lower_bound(i) << " - "
		   << h.get_bin_upper_bound(i);
		if (!h.get_units().empty()) {
			os << " " << h.get_units();
		}
		os << "]: ";

		if (b.get_sample_count() == 0) {
			os << "no data" << std::endl;
			continue;
		}

		double avg = b.get_avg() / h.get_factor();
		os << "avg: " << std::setprecision(2) << std::fixed << avg
		   << ", min: " << b.get_min_sample() / h.get_factor()
		   << ", max: " << b.get_max_sample() / h.get_factor()
		   << ", samples: " << b.get_sample_count() << std::endl;
	}
}

template <typename T>
void table_formatter<T>::format_histogram(const base_histogram<T> &h,
					  bool skip_empty_bins,
					  std::ostream &os)
{
	// prepare table data
	std::string title = std::string(h.get_description()) + " histogram";
	std::vector<std::vector<std::string>> bins;
	bins.push_back({ "Range", "Average", "Min", "Max", "Sample Count" });
	for (size_t i = 0; i < h.get_num_bins(); ++i) {
		const bin<T> &b = h.get_bin(i);
		if (skip_empty_bins && b.get_sample_count() == 0) {
			continue;
		}
		std::stringstream ss;
		if (i + 1 < h.get_num_bins()) {
			ss << h.get_bin_lower_bound(i) << " - " << h.get_bin_upper_bound(i) << " "
			   << h.get_units();
		} else {
			ss << h.get_bin_lower_bound(i) << " - inf " << h.get_units();
		}
		std::string time_range = ss.str();

		ss.str(""); // clear stream
		ss << std::setprecision(2) << std::fixed << (b.get_avg() / h.get_factor());
		std::string avg =
			b.get_sample_count() > 0 ? (ss.str() + " " + h.get_units()) : "N/A";
		std::string min_val =
			b.get_sample_count() > 0
				? (std::to_string(b.get_min_sample() / h.get_factor()) + " " +
				   h.get_units())
				: "N/A";
		std::string max_val =
			b.get_sample_count() > 0
				? (std::to_string(b.get_max_sample() / h.get_factor()) + " " +
				   h.get_units())
				: "N/A";
		bins.push_back({ time_range,
				 avg,
				 min_val,
				 max_val,
				 std::to_string(b.get_sample_count()) });
	}

	// now print table to output stream
	std::vector<Justify> justify = { Justify::JUSTIFY_CENTER,
					 Justify::JUSTIFY_RIGHT,
					 Justify::JUSTIFY_RIGHT,
					 Justify::JUSTIFY_RIGHT,
					 Justify::JUSTIFY_RIGHT };
	print_table(title, bins, justify, os);
}

template <typename T>
void csv_formatter<T>::format_histogram(const base_histogram<T> &h,
					bool skip_empty_bins,
					std::ostream &os)
{
	os << "bin-index,low-bound,high-bound,avg,min,max,count" << std::endl;
	for (size_t i = 0; i < h.get_num_bins(); ++i) {
		const bin<T> &b = h.get_bin(i);
		if (skip_empty_bins && b.get_sample_count() == 0) {
			continue;
		}
		os << i << "," << h.get_bin_lower_bound(i) << "," << h.get_bin_upper_bound(i)
		   << ",";

		if (b.get_sample_count() == 0) {
			os << "0,0,0,0" << std::endl;
			continue;
		}

		double avg = b.get_avg() / h.get_factor();
		os << std::setprecision(2) << std::fixed << avg << ","
		   << b.get_min_sample() / h.get_factor() << ","
		   << b.get_max_sample() / h.get_factor() << "," << b.get_sample_count()
		   << std::endl;
	}
}

template <typename T>
void json_formatter<T>::format_histogram(const base_histogram<T> &h,
					 bool skip_empty_bins,
					 std::ostream &os)
{
	os << "{" << std::endl;
	os << "  \"name\": \"" << h.get_description() << "\"," << std::endl;
	os << "  \"bins\": [" << std::endl;
	for (size_t i = 0; i < h.get_num_bins(); ++i) {
		const bin<T> &b = h.get_bin(i);
		if (skip_empty_bins && b.get_sample_count() == 0) {
			continue;
		}

		os << "    { \"start\": " << h.get_bin_lower_bound(i)
		   << ", \"end\": " << h.get_bin_upper_bound(i);

		if (b.get_sample_count() == 0) {
			os << ", \"avg\": \"N/A\", \"min\": \"N/A\", \"max\": \"N/A\", \"count\": "
			      "0 }";
		} else {
			double avg = b.get_avg() / h.get_factor();
			os << ", \"avg\": " << std::setprecision(2) << std::fixed << avg
			   << ", \"min\": " << b.get_min_sample() / h.get_factor()
			   << ", \"max\": " << b.get_max_sample() / h.get_factor()
			   << ", \"count\": " << b.get_sample_count() << " }";
		}

		if (i + 1 < h.get_num_bins()) {
			os << ",";
		}
		os << std::endl;
	}
	os << "  ]" << std::endl;
	os << "}";
}

#endif