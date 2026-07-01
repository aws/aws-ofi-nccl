/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "stats/histogram.h"

#include <mutex>
#include <unordered_map>
#include <atomic>
#include <cassert>
#include <sstream>
#include <cstring>

#include "nccl_ofi_log.h"

using HistogramMap = std::unordered_map<std::string, BaseHistogram *>;
using GlobalHistogramMap = std::unordered_map<uint64_t, HistogramMap>;
using ThreadIdMap = std::unordered_map<pthread_t, uint64_t>;

// manage all histograms on all threads
static std::mutex sHistogramMapLock;
static GlobalHistogramMap sGlobalHistogramMap;
static bool sHistogramsPrinted = false;

// simple thread id management
static std::atomic<uint64_t> sNextThreadId = 0;
static const uint64_t INVALID_THREAD_ID = (uint64_t)-1;
static uint64_t sCurrentThreadId = INVALID_THREAD_ID;

inline uint64_t getCurrentThreadId()
{
	if (sCurrentThreadId == INVALID_THREAD_ID) {
		sCurrentThreadId = sNextThreadId.fetch_add(1, std::memory_order_relaxed);
	}
	return sCurrentThreadId;
}

// get a predefined print formatter
static PrintFormatter *getPredefinedFormatter(PrintFormat format);

// table printing
enum class Justify { JUSTIFY_LEFT, JUSTIFY_CENTER, JUSTIFY_RIGHT };
void printTable(const std::string &title,
		const std::vector<std::vector<std::string>> &rows,
		const std::vector<Justify> &colJustify,
		std::ostream &os);
std::string fmtVal(double v, size_t precision = 2, const char *units = " ns");
std::string fmtVal(uint64_t v, const char *units = " ns");

// in order to avoid explicit initialization/termination code we use atexit
// this is a compromise between avoiding a memory leak or adding termination code to
// nccl_net_ofi_fini().
class AtExitRegisterHelper {
public:
	AtExitRegisterHelper()
	{
		atexit(BaseHistogram::cleanupHistograms);
	}
};
static AtExitRegisterHelper _atexitHelper;

class TextFormatter : public PrintFormatter {
public:
	static TextFormatter *getInstance();
	void formatHistogram(const BaseHistogram &h, bool skipEmptyBins, std::ostream &os) override;
};

class TableFormatter : public PrintFormatter {
public:
	static TableFormatter *getInstance();
	void formatHistogram(const BaseHistogram &h, bool skipEmptyBins, std::ostream &os) override;
};

class CSVFormatter : public PrintFormatter {
public:
	static CSVFormatter *getInstance();
	void formatHistogram(const BaseHistogram &h, bool skipEmptyBins, std::ostream &os) override;
};

class JsonFormatter : public PrintFormatter {
public:
	static JsonFormatter *getInstance();
	void formatHistogram(const BaseHistogram &h, bool skipEmptyBins, std::ostream &os) override;
};

LinearBinGenerator::LinearBinGenerator(uint64_t binSize /* = OFI_DEFAULT_BIN_SIZE */,
				       uint64_t binCount /* = OFI_DEFAULT_BIN_COUNT */,
				       uint64_t binOffset /* = 0 */)
    : m_binSize(binSize), m_binCount(binCount), m_binOffset(binOffset)
{
}

uint64_t LinearBinGenerator::getBinCount() const
{
	// add extra bin for large values, and if we have offset for first bin then add
	// another bin for all small values
	return (m_binOffset > 0) ? m_binCount + 2 : m_binCount + 1;
}

uint64_t LinearBinGenerator::selectBin(uint64_t sample) const
{
	uint64_t bin = 0;
	if (m_binOffset == 0) {
		bin = sample / m_binSize;
	} else {
		if (sample < m_binOffset) {
			bin = 0;
		} else {
			bin = (sample - m_binOffset) / m_binSize + 1;
		}
	}
	uint64_t binCount = getBinCount();
	if (bin >= binCount) {
		bin = binCount - 1;
	}
	return bin;
}

uint64_t LinearBinGenerator::getBinBound(uint64_t binIndex) const
{
	if (binIndex == getBinCount() - 1) {
		// last bin is unbounded in any case
		return UINT64_MAX;
	}
	// if no offset, then do linera computation
	if (m_binOffset == 0) {
		return (binIndex + 1) * m_binSize - 1;
	}
	// with offset, first bin has different bound, others shifted by offset
	if (binIndex == 0) {
		return m_binOffset - 1;
	} else {
		return m_binOffset + binIndex * m_binSize - 1;
	}
}


Pow2LinearBinGenerator::Pow2LinearBinGenerator(
	uint64_t binSizeFactor /* = OFI_DEFAULT_BIN_SIZE_FACTOR */,
	uint64_t binCount /* = OFI_DEFAULT_BIN_COUNT */,
	uint64_t binOffset /* = 0 */)
    : m_binSizeFactor(binSizeFactor), m_binCount(binCount), m_binOffset(binOffset)
{
}

uint64_t Pow2LinearBinGenerator::getBinCount() const
{
	// add extra bin for large values, and if we have offset for first bin then add
	// another bin for all small values
	return (m_binOffset > 0) ? m_binCount + 2 : m_binCount + 1;
}

uint64_t Pow2LinearBinGenerator::selectBin(uint64_t sample) const
{
	uint64_t bin = 0;
	if (m_binOffset == 0) {
		bin = sample >> m_binSizeFactor;
	} else {
		if (sample < m_binOffset) {
			bin = 0;
		} else {
			bin = ((sample - m_binOffset) >> m_binSizeFactor) + 1;
		}
	}
	uint64_t binCount = getBinCount();
	if (bin >= binCount) {
		bin = binCount - 1;
	}
	return bin;
}

uint64_t Pow2LinearBinGenerator::getBinBound(uint64_t binIndex) const
{
	if (binIndex == getBinCount() - 1) {
		// last bin is unbounded in any case
		return UINT64_MAX;
	}
	// if no offset, then do linera computation
	if (m_binOffset == 0) {
		return ((binIndex + 1) << m_binSizeFactor) - 1;
	}
	// with offset, first bin has different bound, others shifted by offset
	if (binIndex == 0) {
		return m_binOffset;
	} else {
		return m_binOffset + (binIndex << m_binSizeFactor) - 1;
	}
}


uint64_t Log2BinGenerator::getBinCount() const
{
	return sizeof(uint64_t) * 8;
}

uint64_t Log2BinGenerator::getBinBound(uint64_t binIndex) const
{
	if (binIndex == 63) {
		return UINT64_MAX;
	}
	return (1ull << (binIndex + 1)) - 1;
}

Bin::Bin(uint64_t bound) : m_bound(bound), m_sum(0), m_count(0), m_min(0), m_max(0)
{
}

void Bin::aggregate(const Bin &b)
{
	m_sum += b.m_sum;
	m_count += b.m_count;
	m_min = std::min(m_min, b.m_min);
	m_max = std::max(m_max, b.m_max);
}

void BaseHistogram::aggregate(const BaseHistogram &h)
{
	for (size_t i = 0; i < m_bins.size(); ++i) {
		m_bins[i].aggregate(h.m_bins[i]);
	}
}

void BaseHistogram::print(PrintFormat printFormat,
			  bool skipEmptyBins /* = false */,
			  PrintFormatter *formatter /* = nullptr */) const
{
	// however we format the output, we must avoid printing to stdout line by line, since in
	// multi-rank use cases (as in SLURM), the output of multiple histograms from different
	// ranks will be printed together, all mixed up
	std::ostringstream oss;
	if (printFormat != PrintFormat::PF_CUSTOM) {
		formatter = getPredefinedFormatter(printFormat);
	}
	if (formatter == nullptr) {
		// deny request
		NCCL_OFI_WARN("Cannot print histograms, invalid print format or internal error");
		return;
	}
	formatter->formatHistogram(*this, skipEmptyBins, oss);

	// now print histogram
	NCCL_OFI_INFO(NCCL_ALL, "Histogram [%s] data: \n%s", getName(), oss.str().c_str());
}

void BaseHistogram::cleanupHistograms()
{
	// if histograms have not been printed at all we should print them now with default settings
	if (!sHistogramsPrinted) {
		// CSV, no aggregation
		printAllHistograms(PrintFormat::PF_CSV);
	}

	// cleanup
	for (auto &kv : sGlobalHistogramMap) {
		HistogramMap &histogramMap = kv.second;
		for (auto &kv2 : histogramMap) {
			BaseHistogram *histogram = kv2.second;
			delete histogram;
		}
		histogramMap.clear();
	}
	sGlobalHistogramMap.clear();
}

BaseHistogram *getOrAllocHistogram(const char *name, const BaseHistogramAllocator &allocator)
{
	// search in current thread's map
	// NOTE: we could do better here with read-write lock, but this is really not in the
	// critical path, so the effort is not worth the added complexity
	std::unique_lock<std::mutex> lock(sHistogramMapLock);
	HistogramMap &histogramMap = sGlobalHistogramMap[getCurrentThreadId()];
	auto itr = histogramMap.find(name);
	if (itr != histogramMap.end()) {
		return itr->second;
	}
	BaseHistogram *histogram = allocator.allocHistogram();
	if (histogram != nullptr) {
		bool res = histogramMap.insert({ name, histogram }).second;
		assert(res);
	}
	return histogram;
}

static void aggregateHistograms(HistogramMap &aggMap)
{
	// aggregate by name over all threads
	for (const auto &kv : sGlobalHistogramMap) {
		const HistogramMap &histogramMap = kv.second;
		for (const auto &kv2 : histogramMap) {
			const std::string &name = kv2.first;
			BaseHistogram *histogram = kv2.second;
			aggMap[name]->aggregate(*histogram);
		}
	}
}

static void parsePrintFormat(const char *printFormatStr, PrintFormat &printFormat)
{
	if (strcasecmp(printFormatStr, "TEXT") == 0) {
		printFormat = PrintFormat::PF_TEXT;
	} else if (strcasecmp(printFormatStr, "TABLE") == 0) {
		printFormat = PrintFormat::PF_TABLE;
	} else if (strcasecmp(printFormatStr, "CSV") == 0) {
		printFormat = PrintFormat::PF_CSV;
	} else if (strcasecmp(printFormatStr, "JSON") == 0) {
		printFormat = PrintFormat::PF_JSON;
	} else {
		NCCL_OFI_WARN("Unrecognized histogram print format: %s, ignoring", printFormatStr);
	}
}

static void parseBool(const char *boolStr, bool &boolValue, const char *varName)
{
	if (strcasecmp(boolStr, "true") == 0 || strcasecmp(boolStr, "yes") == 0 ||
	    strcasecmp(boolStr, "on") == 0) {
		boolValue = true;
	} else if (strcasecmp(boolStr, "false") == 0 || strcasecmp(boolStr, "no") == 0 ||
		   strcasecmp(boolStr, "off") == 0) {
		boolValue = false;
	} else {
		NCCL_OFI_WARN("Unrecognized histogram %s flag: %s, ignoring", varName, boolStr);
	}
}

static void parseAggregate(const char *aggregateStr, bool &aggregate)
{
	parseBool(aggregateStr, aggregate, "aggregate");
}

static void parseSkipEmptyBins(char *skipEmptyBinsStr, bool &skipEmptyBins)
{
	parseBool(skipEmptyBinsStr, skipEmptyBins, "skip-empty-bins");
}

void printAllHistograms(PrintFormat printFormat /* = PrintFormat::PF_TEXT */,
			PrintFormatter *formatter /* = nullptr */,
			bool skipEmptyBins /* = false */,
			bool aggregate /* = false */)
{
	sHistogramsPrinted = true;

	// check for env override
	char *printFormatStr = getenv("OFI_HISTOGRAM_PRINT_FORMAT");
	if (printFormatStr != nullptr) {
		parsePrintFormat(printFormatStr, printFormat);
	}
	char *aggStr = getenv("OFI_HISTOGRAM_AGGREGATE");
	if (aggStr != nullptr) {
		parseAggregate(aggStr, aggregate);
	}
	char *skipEmptyBinsStr = getenv("OFI_HISTOGRAM_SKIP_EMPTY_BINS");
	if (skipEmptyBinsStr != nullptr) {
		parseSkipEmptyBins(skipEmptyBinsStr, skipEmptyBins);
	}

	if (aggregate) {
		HistogramMap aggMap;
		aggregateHistograms(aggMap);
		for (const auto &kv2 : aggMap) {
			BaseHistogram *histogram = kv2.second;
			histogram->print(printFormat, skipEmptyBins, formatter);
		}
	} else {
		// without aggregation we just print each histogram
		for (const auto &kv : sGlobalHistogramMap) {
			const HistogramMap &histogramMap = kv.second;
			for (const auto &kv2 : histogramMap) {
				BaseHistogram *histogram = kv2.second;
				histogram->print(printFormat, skipEmptyBins, formatter);
			}
		}
	}
}

PrintFormatter *getPredefinedFormatter(PrintFormat format)
{
	switch (format) {
	case PrintFormat::PF_TEXT:
		return TextFormatter::getInstance();

	case PrintFormat::PF_TABLE:
		return TableFormatter::getInstance();

	case PrintFormat::PF_CSV:
		return CSVFormatter::getInstance();

	case PrintFormat::PF_JSON:
		return JsonFormatter::getInstance();

	default:
		return nullptr;
	}
}

TextFormatter *TextFormatter::getInstance()
{
	static TextFormatter sInstance;
	return &sInstance;
}

void TextFormatter::formatHistogram(const BaseHistogram &h, bool skipEmptyBins, std::ostream &os)
{
	uint64_t binLowerBound = 0;
	for (size_t i = 0; i < h.getBinCount(); ++i) {
		const Bin &bin = h.getBinAt(i);
		if (skipEmptyBins && bin.getCount() == 0) {
			continue;
		}
		uint64_t binUpperBound = h.getBinBound(i);
		os << "Bin " << i << " [" << binLowerBound << " - " << binUpperBound;
		if (!h.getUnits().empty()) {
			os << " " << h.getUnits();
		}
		os << "]: ";
		binLowerBound = binUpperBound + 1;

		if (bin.getCount() == 0) {
			os << "no data" << std::endl;
			continue;
		}

		uint64_t avg = bin.getSum() / bin.getCount() / h.getFactor();
		os << "avg: " << avg << ", min: " << bin.getMin() / h.getFactor()
		   << ", max: " << bin.getMax() / h.getFactor() << ", samples: " << bin.getCount()
		   << std::endl;
	}
}

TableFormatter *TableFormatter::getInstance()
{
	static TableFormatter sInstance;
	return &sInstance;
}

void TableFormatter::formatHistogram(const BaseHistogram &h, bool skipEmptyBins, std::ostream &os)
{
	// prepare table data
	std::string title = std::string(h.getName()) + " histogram";
	std::vector<std::vector<std::string>> bins;
	bins.push_back({ "Range", "Average", "Min", "Max", "Sample Count" });
	uint64_t binLowerBound = 0;
	for (size_t i = 0; i < h.getBinCount(); ++i) {
		const Bin &bin = h.getBinAt(i);
		if (skipEmptyBins && bin.getCount() == 0) {
			continue;
		}
		uint64_t binUpperBound = h.getBinBound(i);
		std::stringstream ss;
		if (i + 1 < h.getBinCount()) {
			ss << binLowerBound << " - " << binUpperBound << " " << h.getUnits();
		} else {
			ss << binLowerBound << " - inf " << h.getUnits();
		}
		std::string timeRange = ss.str();
		binLowerBound = binUpperBound + 1;

		std::string avg =
			bin.getCount() > 0
				? (std::to_string(bin.getSum() / bin.getCount() / h.getFactor()) +
				   " " + h.getUnits())
				: "N/A";
		std::string minVal = bin.getCount() > 0
					     ? (std::to_string(bin.getMin() / h.getFactor()) + " " +
						h.getUnits())
					     : "N/A";
		std::string maxVal = bin.getCount() > 0
					     ? (std::to_string(bin.getMax() / h.getFactor()) + " " +
						h.getUnits())
					     : "N/A";
		bins.push_back({ timeRange, avg, minVal, maxVal, std::to_string(bin.getCount()) });
	}

	// now print table to output stream
	std::vector<Justify> justify = { Justify::JUSTIFY_CENTER,
					 Justify::JUSTIFY_RIGHT,
					 Justify::JUSTIFY_RIGHT,
					 Justify::JUSTIFY_RIGHT,
					 Justify::JUSTIFY_RIGHT };
	printTable(title, bins, justify, os);
}

CSVFormatter *CSVFormatter::getInstance()
{
	static CSVFormatter sInstance;
	return &sInstance;
}

void CSVFormatter::formatHistogram(const BaseHistogram &h, bool skipEmptyBins, std::ostream &os)
{
	os << "bin-index,low-bound,high-bound,avg,min,max,count" << std::endl;
	uint64_t binLowerBound = 0;
	for (size_t i = 0; i < h.getBinCount(); ++i) {
		const Bin &bin = h.getBinAt(i);
		if (skipEmptyBins && bin.getCount() == 0) {
			continue;
		}
		uint64_t binUpperBound = h.getBinBound(i);
		os << i << "," << binLowerBound << "," << binUpperBound << ",";
		binLowerBound = binUpperBound + 1;

		if (bin.getCount() == 0) {
			os << "0,0,0,0" << std::endl;
			continue;
		}

		uint64_t avg = bin.getSum() / bin.getCount() / h.getFactor();
		os << avg << "," << bin.getMin() / h.getFactor() << ","
		   << bin.getMax() / h.getFactor() << "," << bin.getCount() << std::endl;
	}
}

JsonFormatter *JsonFormatter::getInstance()
{
	static JsonFormatter sInstance;
	return &sInstance;
}

void JsonFormatter::formatHistogram(const BaseHistogram &h, bool skipEmptyBins, std::ostream &os)
{
	os << "{" << std::endl;
	os << "  \"name\": \"" << h.getName() << "\"," << std::endl;
	os << "  \"bins\": [" << std::endl;
	uint64_t binLowerBound = 0;
	for (size_t i = 0; i < h.getBinCount(); ++i) {
		const Bin &bin = h.getBinAt(i);
		if (skipEmptyBins && bin.getCount() == 0) {
			continue;
		}

		uint64_t binUpperBound = h.getBinBound(i);
		os << "    { \"start\": " << binLowerBound << ", \"end\": " << binUpperBound;
		binLowerBound = binUpperBound + 1;

		if (bin.getCount() == 0) {
			os << ", \"avg\": \"N/A\", \"min\": \"N/A\", \"max\": \"N/A\", \"count\": "
			      "0 }";
		} else {
			uint64_t avg = bin.getSum() / bin.getCount() / h.getFactor();
			os << ", \"avg\": " << avg << ", \"min\": " << bin.getMin() / h.getFactor()
			   << ", \"max\": " << bin.getMax() / h.getFactor()
			   << ", \"count\": " << bin.getCount() << " }";
		}

		if (i + 1 < h.getBinCount()) {
			os << ",";
		}
		os << std::endl;
	}
	os << "  ]" << std::endl;
	os << "}";
}

static size_t getTotalColumnWidth(const std::vector<size_t> &colWidth)
{
	size_t total = 0;
	for (size_t w : colWidth) total += w;
	total += colWidth.size() - 1;
	return total;
}

static void printTableBorder(std::ostream &os, const std::vector<size_t> &colWidth)
{
	os << "+" << std::string(getTotalColumnWidth(colWidth), '-') << "+" << std::endl;
}

static void printRowBorder(std::ostream &os, const std::vector<size_t> &colWidth)
{
	os << "+";
	for (size_t w : colWidth) {
		os << std::string(w, '-') << "+";
	}
	os << std::endl;
}

static void
printColumn(std::ostream &os, const std::string &value, size_t colWidth, Justify justify)
{
	size_t padding = colWidth - value.length();
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
printTitle(std::ostream &os, const std::string &title, const std::vector<size_t> &colWidth)
{
	os << "|";
	printColumn(os, title, getTotalColumnWidth(colWidth), Justify::JUSTIFY_CENTER);
	os << "|" << std::endl;
}

static void printRow(std::ostream &os,
		     const std::vector<std::string> &row,
		     const std::vector<size_t> &colWidth,
		     const std::vector<Justify> &colJustify)
{
	os << "|";
	for (size_t i = 0; i < row.size(); ++i) {
		printColumn(os, row[i], colWidth[i], colJustify[i]);
		os << "|";
	}
	os << std::endl;
}

void printTable(const std::string &title,
		const std::vector<std::vector<std::string>> &rows,
		const std::vector<Justify> &colJustify,
		std::ostream &os)
{
	std::vector<size_t> colWidth(rows[0].size(), 0);
	for (auto &row : rows) {
		for (size_t i = 0; i < row.size(); ++i) {
			// reserve space on both sides
			colWidth[i] = std::max(colWidth[i], row[i].length() + 2);
		}
	}
	// if the title width is larger than table width, expand each column width evenly, if not
	// divisible, then last column gets excess space
	if (title.length() + 2 > getTotalColumnWidth(colWidth)) {
		size_t extra = title.length() + 2 - getTotalColumnWidth(colWidth);
		size_t inc = extra / colWidth.size();
		size_t remainder = extra % colWidth.size();
		for (size_t i = 0; i < colWidth.size(); ++i)
			colWidth[i] += inc + (i == colWidth.size() - 1 ? remainder : 0);
	}

	if (!title.empty()) {
		printTableBorder(os, colWidth);
		printTitle(os, title, colWidth);
	}
	// first row should have all center justify
	std::vector<Justify> colTitleJustify(colWidth.size(), Justify::JUSTIFY_CENTER);
	for (size_t i = 0; i < rows.size(); ++i) {
		printRowBorder(os, colWidth);
		printRow(os, rows[i], colWidth, i == 0 ? colTitleJustify : colJustify);
	}
	printTableBorder(os, colWidth);
}

std::string fmtVal(double v, size_t precision /* = 2 */, const char *units /* = " ns" */)
{
	std::ostringstream ss;
	ss.precision(precision);
	ss << std::fixed << v << units;
	return ss.str();
}

std::string fmtVal(uint64_t v, const char *units /* = " ns" */)
{
	std::ostringstream ss;
	ss << std::fixed << v << units;
	return ss.str();
}