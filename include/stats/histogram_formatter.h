//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#ifndef NCCL_OFI_STATS_HISTOGRAM_FORMATTER
#define NCCL_OFI_STATS_HISTOGRAM_FORMATTER

#include <iostream>

#include "histogram_def.h"

// forward declaration
template <typename T> class base_histogram;

/**
 * @brief Helper for printing data into an output stream in a tabular form.
 * @param title Table title.
 * @param rows Data rows.
 * @param col_justify Justification for each column.
 * @param os The target output stream.
 */
extern void print_table(const std::string &title,
			const std::vector<std::vector<std::string>> &rows,
			const std::vector<Justify> &col_justify,
			std::ostream &os);

/** @brief Allow user to customize how histogram are formatted for printing. */
template <typename T> class histogram_formatter {
public:
	virtual ~histogram_formatter() = default;

	/**
	 * @brief Format and print a histogram to the given output stream.
	 * @param h The histogram to format.
	 * @param skip_empty_bins Specifies whether empty bins should be printed or not.
	 * @param os The output stream used to print the histogram.
	 */
	virtual void
	format_histogram(const base_histogram<T> &h, bool skip_empty_bins, std::ostream &os) = 0;

protected:
	histogram_formatter() = default;
};

/** @brief Format histogram data as plain text. */
template <typename T> class text_formatter : public histogram_formatter<T> {
public:
	static text_formatter<T> *get_instance()
	{
		static text_formatter<T> instance;
		return &instance;
	}

	void format_histogram(const base_histogram<T> &h,
			      bool skip_empty_bins,
			      std::ostream &os) override;

private:
	text_formatter() = default;
	~text_formatter() override = default;
};

/** @brief Format histogram data as a table. */
template <typename T> class table_formatter : public histogram_formatter<T> {
public:
	static table_formatter<T> *get_instance()
	{
		static table_formatter<T> instance;
		return &instance;
	}

	void format_histogram(const base_histogram<T> &h,
			      bool skip_empty_bins,
			      std::ostream &os) override;

private:
	table_formatter() = default;
	~table_formatter() override = default;
};

/** @brief Format histogram data as CSV text. */
template <typename T> class csv_formatter : public histogram_formatter<T> {
public:
	static csv_formatter<T> *get_instance()
	{
		static csv_formatter<T> instance;
		return &instance;
	}

	void format_histogram(const base_histogram<T> &h,
			      bool skip_empty_bins,
			      std::ostream &os) override;

private:
	csv_formatter() = default;
	~csv_formatter() override = default;
};

/** @brief Format histogram data as Json text. */
template <typename T> class json_formatter : public histogram_formatter<T> {
public:
	static json_formatter<T> *get_instance()
	{
		static json_formatter<T> instance;
		return &instance;
	}

	void format_histogram(const base_histogram<T> &h,
			      bool skip_empty_bins,
			      std::ostream &os) override;

private:
	json_formatter() = default;
	~json_formatter() override = default;
};

template <typename T> inline histogram_formatter<T> *get_predefined_formatter(PrintFormat format)
{
	switch (format) {
	case PrintFormat::PF_TEXT:
		return text_formatter<T>::get_instance();

	case PrintFormat::PF_TABLE:
		return table_formatter<T>::get_instance();

	case PrintFormat::PF_CSV:
		return csv_formatter<T>::get_instance();

	case PrintFormat::PF_JSON:
		return json_formatter<T>::get_instance();

	default:
		return nullptr;
	}
}

#endif