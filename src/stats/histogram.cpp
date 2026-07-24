/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "stats/histogram.h"

#include <cstring>

#include "nccl_ofi_log.h"

#if ENABLE_HISTOGRAM == 1
static bool histograms_enabled_cfg = true;
#else
static bool histograms_enabled_cfg = true;
#endif

bool histograms_enabled()
{
	return histograms_enabled_cfg;
}

void enable_histograms(bool enable) {
	histograms_enabled_cfg = enable;
}

global_histogram_aggregator &global_histogram_aggregator::get_instance()
{
	static global_histogram_aggregator instance;
	return instance;
}

static void parse_print_format(const char *print_format_str, PrintFormat &print_format)
{
	if (strcasecmp(print_format_str, "TEXT") == 0) {
		print_format = PrintFormat::PF_TEXT;
	} else if (strcasecmp(print_format_str, "TABLE") == 0) {
		print_format = PrintFormat::PF_TABLE;
	} else if (strcasecmp(print_format_str, "CSV") == 0) {
		print_format = PrintFormat::PF_CSV;
	} else if (strcasecmp(print_format_str, "JSON") == 0) {
		print_format = PrintFormat::PF_JSON;
	} else {
		NCCL_OFI_WARN("Unrecognized histogram print format: %s, ignoring",
			      print_format_str);
	}
}

static void parse_bool(const char *bool_str, bool &bool_value, const char *var_name)
{
	if (strcasecmp(bool_str, "true") == 0 || strcasecmp(bool_str, "yes") == 0 ||
	    strcasecmp(bool_str, "on") == 0) {
		bool_value = true;
	} else if (strcasecmp(bool_str, "false") == 0 || strcasecmp(bool_str, "no") == 0 ||
		   strcasecmp(bool_str, "off") == 0) {
		bool_value = false;
	} else {
		NCCL_OFI_WARN("Unrecognized histogram %s flag: %s, ignoring", var_name, bool_str);
	}
}

static void parse_skip_empty_bins(char *skip_empty_binsStr, bool &skip_empty_bins)
{
	parse_bool(skip_empty_binsStr, skip_empty_bins, "skip-empty-bins");
}

void global_histogram_aggregator::load_env_override(PrintFormat &print_format,
						    bool &skip_empty_bins)
{
	char *print_format_str = getenv("OFI_HISTOGRAM_PRINT_FORMAT");
	if (print_format_str != nullptr) {
		parse_print_format(print_format_str, print_format);
	}
	char *skip_empty_binsStr = getenv("OFI_HISTOGRAM_SKIP_EMPTY_BINS");
	if (skip_empty_binsStr != nullptr) {
		parse_skip_empty_bins(skip_empty_binsStr, skip_empty_bins);
	}
}

static size_t get_total_column_width(const std::vector<size_t> &col_width)
{
	size_t total = 0;
	for (size_t w : col_width) total += w;
	total += col_width.size() - 1;
	return total;
}

static void print_table_border(std::ostream &os, const std::vector<size_t> &col_width)
{
	os << "+" << std::string(get_total_column_width(col_width), '-') << "+" << std::endl;
}

static void print_row_border(std::ostream &os, const std::vector<size_t> &col_width)
{
	os << "+";
	for (size_t w : col_width) {
		os << std::string(w, '-') << "+";
	}
	os << std::endl;
}

static void
print_column(std::ostream &os, const std::string &value, size_t col_width, Justify justify)
{
	size_t padding = col_width - value.length();
	size_t left = 0;
	if (justify == Justify::JUSTIFY_LEFT)
		left = 1;
	else if (justify == Justify::JUSTIFY_CENTER)
		left = padding / 2;
	else if (justify == Justify::JUSTIFY_RIGHT)
		left = padding - 1;
	size_t right = padding - left;
	os << std::string(left, ' ') << value << std::string(right, ' ');
}

static void
print_title(std::ostream &os, const std::string &title, const std::vector<size_t> &col_width)
{
	os << "|";
	print_column(os, title, get_total_column_width(col_width), Justify::JUSTIFY_CENTER);
	os << "|" << std::endl;
}

static void print_row(std::ostream &os,
		      const std::vector<std::string> &row,
		      const std::vector<size_t> &col_width,
		      const std::vector<Justify> &col_justify)
{
	os << "|";
	for (size_t i = 0; i < row.size(); ++i) {
		print_column(os, row[i], col_width[i], col_justify[i]);
		os << "|";
	}
	os << std::endl;
}

void print_table(const std::string &title,
		 const std::vector<std::vector<std::string>> &rows,
		 const std::vector<Justify> &col_justify,
		 std::ostream &os)
{
	std::vector<size_t> col_width(rows[0].size(), 0);
	for (auto &row : rows) {
		for (size_t i = 0; i < row.size(); ++i) {
			// reserve space on both sides
			col_width[i] = std::max(col_width[i], row[i].length() + 2);
		}
	}
	// if the title width is larger than table width, expand each column width evenly, if not
	// divisible, then last column gets excess space
	if (title.length() + 2 > get_total_column_width(col_width)) {
		size_t extra = title.length() + 2 - get_total_column_width(col_width);
		size_t inc = extra / col_width.size();
		size_t remainder = extra % col_width.size();
		for (size_t i = 0; i < col_width.size(); ++i)
			col_width[i] += inc + (i == col_width.size() - 1 ? remainder : 0);
	}

	if (!title.empty()) {
		print_table_border(os, col_width);
		print_title(os, title, col_width);
	}
	// first row should have all center justify
	std::vector<Justify> colTitleJustify(col_width.size(), Justify::JUSTIFY_CENTER);
	for (size_t i = 0; i < rows.size(); ++i) {
		print_row_border(os, col_width);
		print_row(os, rows[i], col_width, i == 0 ? colTitleJustify : col_justify);
	}
	print_table_border(os, col_width);
}