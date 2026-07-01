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

#include "nccl_ofi_config_bottom.h"

/*
 * @file	Provide user-level profiling primitives (currently only histograms).
 *
 * Usage examples:
 *
 * 1. Simplae latency histogram to measure whole function time:
 *
 * ```c
 * void someFunction() {
 *   OFI_DECLARE_SCOPED_LATENCY_HISTOGRAM(someFunction);
 * }
 * ```
 *
 * 2. Simple data histogram to mesaure data distribution:
 *
 * ```c
 * uint64_t someDataSample = ...;
 * OFI_DECLARE_HISTOGRAM_ADD_SAMPLE(bench_data_histogram, someDataSample);
 * ```
 *
 * 3. Latency histogram with bin size 1000 nanos, 10 bins, and first bin offset of 500 nanoseconds.
 * Units is microseconds:
 *
 * ```c
 * OFI_DECLARE_LATENCY_HISTOGRAM_EX(someName,
 * 	 OFI_LINEAR_BIN_GENERATOR(1000, 10, 500), OF_DEFAULT_CLOCK, "us", 1000);
 * ...
 * OFI_HISTOGRAM_START(someName);
 * ...
 * OFI_HISTOGRAM_END(someName);
 * ```
 */

/** @def Default bin size. Matching 200 nanoseconds for latency histogram. */
#define OFI_DEFAULT_BIN_SIZE 200

/**
 * @def Default bin count. Matching total of 2 microseconds for linear bin generator with latency
 * histogram, and 2.048 microseconds for pow2-linear bin generator.
 */
#define OFI_DEFAULT_BIN_COUNT 10

/**
 * @def Default bin size factor. Matching 256 nanoseconds for pow2-linear bin generator with
 * latency histogram.
 */
#define OFI_DEFAULT_BIN_SIZE_FACTOR 8

/** @def Compiler barrier/fence to enfoce no instruction reordering during compile-time. */
#define OFI_COMPILER_BARRIER asm volatile("" : : : "memory");

/** @def Default clock to use when taking time stamps. */
#define OFI_DEFAULT_CLOCK SteadyClock

/**
 * @brief Steady clock adapter.
 *
 * @note Since we are measuring time interval in nanoseconds,
 * there is no motivation in adding template parameter for time representation data time.
 */
class SteadyClock {
public:
	/** @brief Get current clock timestamp in nanoseconds. */
	static uint64_t getSysTimeNS()
	{
		return std::chrono::steady_clock::now().time_since_epoch().count();
	}
};

/** @brief High-resolution clock adapetr. */
class HighResClock {
public:
	static uint64_t getSysTimeNS()
	{
		return std::chrono::high_resolution_clock::now().time_since_epoch().count();
	}
};

/**
 * @brief A linear bin generator. Each bin has the same size. Offset (lower-bound) for first bin
 * may be specified.
 */
class LinearBinGenerator {
public:
	/**
	 * @brief Constructs a linear bin generator.
	 * @param binSize The size of each bin.
	 * @param binCount The bin count (will add extra bin for large values).
	 * @param binOffset The initial offset for all bins. If larger than zero, then an initial
	 * bin with range [0, binOffset] (not including right end point) will be added.
	 */
	LinearBinGenerator(uint64_t binSize = OFI_DEFAULT_BIN_SIZE,
			   uint64_t binCount = OFI_DEFAULT_BIN_COUNT,
			   uint64_t binOffset = 0);

	LinearBinGenerator(const LinearBinGenerator &) = default;
	~LinearBinGenerator() = default;

	/**
	 * @brief Retrieves the number of bins (including extra bin for large values and initial
	 * bin if bin offset is not zero).
	 */
	uint64_t getBinCount() const;

	/** @brief Retreives the bin index for a sample value. */
	uint64_t selectBin(uint64_t sample) const;

	/* @brief Retrieves the upper bound (inclusive) for a bin by its index. */
	uint64_t getBinBound(uint64_t binIndex) const;

private:
	uint64_t m_binSize;
	uint64_t m_binCount;
	uint64_t m_binOffset;
};

/**
 * @brief Optimized linear bin generator. Bin size is specified as a power of 2, so bin selection
 * is faster (using shift instead of heavy integer division).
 */
class Pow2LinearBinGenerator {
public:
	/**
	 * @brief Constructs a linear bin generator, with bin size specifies as power of 2.
	 * @param binSizeFactor The size of each bin, specified as a power of 2 (e.g. 8 means bin
	 * size 256).
	 * @param binCount The bin count (will add extra bin for large values).
	 * @param binOffset The initial offset for all bins. If larger than zero, then an initial
	 * bin with range [0, binOffset] (not including right end point) will be added.
	 */
	Pow2LinearBinGenerator(uint64_t binSizeFactor = OFI_DEFAULT_BIN_SIZE_FACTOR,
			       uint64_t binCount = OFI_DEFAULT_BIN_COUNT,
			       uint64_t binOffset = 0);

	Pow2LinearBinGenerator(const Pow2LinearBinGenerator &) = default;
	~Pow2LinearBinGenerator() = default;

	/**
	 * @brief Retrieves the number of bins (including extra bin for large values and initial
	 * bin if bin offset is not zero).
	 */
	uint64_t getBinCount() const;

	/** @brief Retreives the bin index for a sample value. */
	uint64_t selectBin(uint64_t sample) const;

	/** @brief Retrieves the upper bound (inclusive) for a bin by its index. */
	uint64_t getBinBound(uint64_t binIndex) const;

private:
	uint64_t m_binSizeFactor;
	uint64_t m_binCount;
	uint64_t m_binOffset;
};

/**
 * @brief Logarithmic bin generator. Bin size is an increasing power of 2. There is a total of 64
 * bins. This is probably the fastest implementation.
 */
class Log2BinGenerator {
public:
	/**
	 * @brief Constructs a logarithmic bin generator, where each bin size is an increasing power
	 * of 2 (i.e. 2, 4, 8, etc.).
	 * @param binCount The bin count (will add extra bin for large values).
	 * @param binOffset The initial offset for all bins. If larger than zero, then an initial
	 * bin with range [0, binOffset] (not including right end point) will be added.
	 */
	Log2BinGenerator() = default;

	Log2BinGenerator(const Log2BinGenerator &) = default;
	~Log2BinGenerator() = default;

	/**
	 * @brief Retrieves the number of bins which is always 64.
	 */
	uint64_t getBinCount() const;

	/** @brief Retreives the bin index for a sample value. */
	inline uint64_t selectBin(uint64_t sample) const
	{
		// we have exactly 64 bins
		// bin 0 for 0-1
		// bin 1 for 2-3
		// bin 2 for 4-7
		// bin 3 for 8-15 [2^3, 2^4)
		// bin N for 2^N - 2^(N+1)-1 [2^N, 2^(N+1))
		// bin 63 for 2^63 - 2^64-1

		// count leading zeros (on the left), to get the highest raised bit (inverse of 64)
		// this will isolate the bin index
		if (sample <= 1) {
			return 0;
		}
		uint64_t bit = __builtin_clzll(sample);
		return 63 - bit;
	}

	/** @brief Retrieves the upper bound (inclusive) for a bin by its index. */
	uint64_t getBinBound(uint64_t binIndex) const;
};

/** @brief A histogram bin. */
class Bin {
public:
	Bin(uint64_t bound = 0);
	Bin(const Bin &) = default;
	~Bin() = default;

	/** @brief Sets the bin's upper bound. */
	inline void setBound(uint64_t bound)
	{
		m_bound = bound;
	}

	/** @brief Adds a sample to the bin. */
	inline void addSample(uint64_t sampleValue)
	{
		// despite this slight optimization, there is no way avoidingtwo if-statements here
		if (OFI_UNLIKELY(m_firstInsert)) {
			m_max = m_min = sampleValue;
			m_firstInsert = false;
		}

		if (sampleValue > m_max) {
			m_max = sampleValue;
		} else if (sampleValue < m_min) {
			m_min = sampleValue;
		}

		m_sum += sampleValue;
		m_count++;
	}

	void aggregate(const Bin &b);

	inline uint64_t getBound() const
	{
		return m_bound;
	}

	inline uint64_t getSum() const
	{
		return m_sum;
	}

	inline uint64_t getCount() const
	{
		return m_count;
	}

	inline uint64_t getMin() const
	{
		return m_min;
	}

	inline uint64_t getMax() const
	{
		return m_max;
	}

private:
	uint64_t m_bound;
	uint64_t m_sum;
	uint64_t m_count;
	uint64_t m_min;
	uint64_t m_max;
	bool m_firstInsert;
};

/** @enum Histogram print format */
enum class PrintFormat {
	/** @brief Textual format (each bin in a line). */
	PF_TEXT,

	/** @brief Tabular format. */
	PF_TABLE,

	/** @brief CSV format (header row and data rows) */
	PF_CSV,

	/** @brief Json format. */
	PF_JSON,

	/** @brief Custom user format. */
	PF_CUSTOM
};

// forward declaration
class PrintFormatter;

/**
 * @brief Base histogram class (extracted to avoid having all implementation in header file due to
 * templates).
 */
class BaseHistogram {
public:
	virtual ~BaseHistogram() = default;

	inline const char *getName() const
	{
		return m_name.c_str();
	}

	inline const std::string &getUnits() const
	{
		return m_units;
	}

	inline uint64_t getFactor() const
	{
		return m_factor;
	}

	inline size_t getBinCount() const
	{
		return m_bins.size();
	}

	inline const Bin &getBinAt(size_t index) const
	{
		return m_bins[index];
	}

	inline uint64_t getBinBound(size_t index) const
	{
		return m_bins[index].getBound();
	}

	/** @brief Aggregates histograms. */
	void aggregate(const BaseHistogram &h);

	/**
	 * @brief Prints the histogram.
	 * @param printFormat The output print format.
	 * @param skipEmptyBins Optioanlly specify whether empty bins should be printed or not (by
	 * default yes).
	 * @param formatter Optional custom user-provided print formatter. Used when printFormat is
	 * set to PF_CUSTOM.
	 */
	void print(PrintFormat printFormat,
		   bool skipEmptyBins = false,
		   PrintFormatter *formatter = nullptr) const;

protected:
	std::string m_name;
	std::string m_units;
	uint64_t m_factor;
	std::vector<Bin> m_bins;

	BaseHistogram(const char *name, const char *units = "", uint64_t factor = 1)
	    : m_name(name), m_units(units), m_factor(factor)
	{
	}

	// the only external function that can destroy a histogram
	static void cleanupHistograms();

	// allow atexit helper to access cleanup function
	friend class AtExitRegisterHelper;
};

/** @brief Allow user to customize how histogram are printed. */
class PrintFormatter {
public:
	virtual ~PrintFormatter() = default;

	/**
	 * @brief Format and print a histogram to the given output stream.
	 */
	virtual void
	formatHistogram(const BaseHistogram &h, bool skipEmptyBins, std::ostream &os) = 0;

protected:
	PrintFormatter() = default;
};

/** @brief Allocator helper class to encapsulate histogram allocation details. */
class BaseHistogramAllocator {
public:
	virtual ~BaseHistogramAllocator() = default;
	virtual BaseHistogram *allocHistogram() const = 0;

protected:
	BaseHistogramAllocator() = default;
};

/**
 * @brief Retrieves a histogram by name for the current thread, or allocates a new one.
 *
 * @param name The histogram's name.
 * @param allocator The allocator used to allocate the histogram if it is not allocated already for
 * the current thread.
 *
 * @return BaseHistogram* The resulting histogram.
 *
 * @note Only one histogram by the same name exists per thread.
 */
extern BaseHistogram *getOrAllocHistogram(const char *name,
					  const BaseHistogramAllocator &allocator);

/**
 * @brief Specific histogram allocator by bin generator type.
 * @tparam BinGenerator The bin generator type.
 * @tparam ClockType The clock type to use (not restricted to std::chrono clocks,
 * so we can further optimize with direct TSC clock, and reduce histogram footprint).
 */
template <typename BinGenerator, typename ClockType = OFI_DEFAULT_CLOCK>
class HistogramAllocator : public BaseHistogramAllocator {
public:
	HistogramAllocator(const char *name,
			   const BinGenerator &binGenerator,
			   const char *units = "",
			   uint64_t factor = 1)
	    : m_name(name), m_binGenerator(binGenerator), m_units(units), m_factor(factor)
	{
	}
	~HistogramAllocator() override
	{
	}

	BaseHistogram *allocHistogram() const override;

private:
	std::string m_name;
	BinGenerator m_binGenerator;
	std::string m_units;
	uint64_t m_factor;
};

/**
 * @brief A histogram for measuring latency or general data distribution.
 * @tparam BinGenerator The bin generator type.
 * @tparam ClockType The clock type to use (not restricted to std::chrono clocks,
 * so we can further optimize with direct TSC clock).
 *
 * @note Since we are measure only non-negative integer values, there is no motivation in using
 * a template data type, and it is fixed as uint64_t.
 */
template <typename BinGenerator, typename ClockType = OFI_DEFAULT_CLOCK>
class Histogram : public BaseHistogram {
public:
	/**
	 * @brief Retrieves or creates a histogram (thread-safe). A static allocator (coupled
	 * with private constructor) is used to control the life-cycle of a single histogram.
	 */
	static Histogram<BinGenerator, ClockType> *getHistogram(const char *name,
								const BinGenerator &binGenerator,
								const char *units = "",
								uint64_t factor = 1)
	{
		HistogramAllocator<BinGenerator, ClockType> allocator(
			name, binGenerator, units, factor);
		return (Histogram<BinGenerator, ClockType> *)getOrAllocHistogram(name, allocator);
	}

	inline void addSample(uint64_t sampleValue)
	{
		uint64_t binIndex = m_binGenerator.selectBin(sampleValue);
		m_bins[binIndex].addSample(sampleValue);
	}

	/** @brief Marks start of latency measurement. */
	inline void addSampleStart()
	{
		m_startTime = ClockType::getSysTimeNS();
		// add compiler fence to ensure start timestamp is taken before measured activity
		OFI_COMPILER_BARRIER;
	}

	/** @brief Marks end of latency measurement. */
	inline uint64_t addSampleEnd()
	{
		// add compiler fence to ensure end timestamp is taken after measured activity
		OFI_COMPILER_BARRIER;
		uint64_t endTime = ClockType::getSysTimeNS();
		uint64_t sampleValue = endTime - m_startTime;
		addSample(sampleValue);
		return sampleValue; // this is really needed only for unit testing
	}

private:
	BinGenerator m_binGenerator;
	uint64_t m_startTime;

	Histogram(const char *name,
		  const BinGenerator &binGenerator,
		  const char *units = "",
		  uint64_t factor = 1)
	    : BaseHistogram(name, units, factor), m_binGenerator(binGenerator), m_startTime(0)
	{
		initBins();
	}

	Histogram(const Histogram &h)
	    : BaseHistogram(h), m_binGenerator(h.m_binGenerator), m_startTime(0)
	{
		initBins();
	}
	~Histogram()
	{
	}

	void initBins()
	{
		m_bins.resize(m_binGenerator.getBinCount());
		for (size_t i = 0; i < m_bins.size(); ++i) {
			m_bins[i].setBound(m_binGenerator.getBinBound(i));
		}
	}

	// let allocator have special access to private constructor
	friend class HistogramAllocator<BinGenerator, ClockType>;
};


// allocator implementation
template <typename BinGenerator, typename ClockType>
BaseHistogram *HistogramAllocator<BinGenerator, ClockType>::allocHistogram() const
{
	return new (std::nothrow) Histogram<BinGenerator, ClockType>(
		m_name.c_str(), m_binGenerator, m_units.c_str(), m_factor);
}

/**
 * @brief Prints all histograms.
 * @param printFormat The output print format.
 * @param formatter Optional custom user-provided print formatter. Used when printFormat is
 * set to PF_CUSTOM.
 * @param skipEmptyBins Specifies whether to avoid printing empty bins (by default they are
 * printed).
 * @param aggregate Aggregates all histograms from all threads that share the same name.
 */
extern void printAllHistograms(PrintFormat printFormat = PrintFormat::PF_TEXT,
			       PrintFormatter *formatter = nullptr,
			       bool skipEmptyBin = false,
			       bool aggrgate = false);

/**
 * @brief A scoped histogram object RAII wrapper.
 * @tparam A pointer type to the histogram.
 */
template <typename T> class ScopedHistogram {
public:
	ScopedHistogram(T histogram) : m_histogram(histogram)
	{
		if (m_histogram != nullptr) {
			m_histogram->addSampleStart();
		}
	}
	~ScopedHistogram()
	{
		if (m_histogram != nullptr) {
			m_histogram->addSampleEnd();
		}
	}

private:
	T m_histogram;
};

// following macros are enabled only when "--enable-histogram" is configured
#if defined(ENABLE_HISTOGRAM) && (ENABLE_HISTOGRAM == 1)

/**
 * Bin Generator macros.
 */

/**
 * @def Linear bin generator.
 * @param size The size of each histogram bin.
 * @param count The number of histogram bins.
 * @param offset The offset of the first bin. Samples with lower values will be put in an extra bin.
 * */
#define OFI_LINEAR_BIN_GENERATOR(size, count, offset) LinearBinGenerator(size, count, offset)

/**
 * @def Power-of-2 linear bin generator. Slightly better performance then linear bin generator. The
 * size of each bin is a power of 2 (still linear, all bins have the same size).
 * @param size_factor The power-of-2 size factor of each histogram bin.
 * @param count The number of histogram bins.
 * @param offset The offset of the first bin. Samples with lower values will be put in an extra bin.
 * */
#define OFI_POW2_BIN_GENERATOR(size_factor, count, offset)                                         \
	Pow2LinearBinGenerator(size_factor, count, offset)

/**
 * @def Log2 bin generator. The size of each bin is a power of 2, such that
 * the histogram is logarithmic.
 */
#define OFI_LOG2_BIN_GENERATOR() Log2BinGenerator()

/**
 * @def Default bin generator. Sutable for measuring latency under 1 microsecond, having 200
 * nanosecond bin resilution.
 */
#define OFI_DEFAULT_BIN_GENERATOR OFI_LINEAR_BIN_GENERATOR(200, 5, 0)

/**
 * Histogram declaration macros.
 */

/**
 * @brief Declares and defines a latency histogram.
 *
 * @param name The histogram name. If aggregation is used all histograms from all threads sharing
 * the same name will be aggregated before report.
 * @param generator The bin generator.
 * @param clock_type The clock type (default is SteadyClock).
 * @param units The histogram's units (e.g. "bytes", "pages", "cycles"). Pass empty string if not
 * used.
 * @param factor Factor to be applied on the raw histogram data (e.g. 1024 for converting bytes to
 * KB). Pass 1 if not converting.
 */
#define OFI_DECLARE_LATENCY_HISTOGRAM_EX(name, generator, clock_type, units, factor)               \
	static thread_local Histogram<decltype(generator), clock_type> *__histogram_##name =       \
		Histogram<decltype(generator), clock_type>::getHistogram(                          \
			#name, generator, units, factor);

/** @def Declares and defines a named latency histogram with default settings. */
#define OFI_DECLARE_LATENCY_HISTOGRAM(name)                                                        \
	OFI_DECLARE_LATENCY_HISTOGRAM_EX(                                                          \
		name, OFI_DEFAULT_BIN_GENERATOR, OFI_DEFAULT_CLOCK, "ns", 1)

/**
 * @brief Declares and defines a named data histogram.
 *
 * @param name The histogram name. If aggregation is used all histograms from all threads sharing
 * the same name will be aggregated before report.
 * @param generator The bin generator.
 * @param units The histogram's units (e.g. "bytes", "pages", "cycles"). Pass empty string if not
 * used.
 * @param factor Factor to be applied on the raw histogram data (e.g. 1024 for converting bytes to
 * KB). Pass 1 if not converting.
 */
#define OFI_DECLARE_DATA_HISTOGRAM_EX(name, generator, units, factor)                              \
	static thread_local Histogram<decltype(generator)> *__histogram_##name =                   \
		Histogram<decltype(generator)>::getHistogram(#name, generator, units, factor);

/** @def Declares and defines a named data histogram with default settings. */
#define OFI_DECLARE_DATA_HISTOGRAM(name)                                                           \
	OFI_DECLARE_DATA_HISTOGRAM_EX(name, OFI_DEFAULT_BIN_GENERATOR, "", 1)

/** @brief Get a reference to the current histogram. */
#define OFI_HISTOGRAM_REF(name) __histogram_##name

/**
 * Histogram data collection macros.
 */

/**
 * @def Defines a scoped latency histogram, using the given named histogram, to measure latency
 * until end of current block.
 *
 * @param name The histogram name. Should be already defined. If aggregation is used all histograms
 * from all threads sharing the same name will be aggregated before report.
 *
 * @note The histogram will be created only once by the first time a request is made to retrieve the
 * histogram by name.
 */
#define OFI_SCOPED_LATENCY_HISTOGRAM(name)                                                         \
	ScopedHistogram<decltype(__histogram_##name)> __scoped_histogram_##name(__histogram_##name);

/**
 * @def Adds a latency histogram start point, using the given named histogram.
 * @param name The name of the histogram. Should be already defined.
 */
#define OFI_HISTORGRAM_START(name)                                                                 \
	if (__histogram_##name != nullptr) {                                                       \
		__histogram_##name->addSampleStart();                                              \
	}

/**
 * @def Adds a latency histogram end point, using the given named histogram.
 * @param name The name of the histogram. Should be already defined.
 *
 * @note Matching start/end points of a latency histogram must be added in the same thread.
 */
#define OFI_HISTORGRAM_END(name)                                                                   \
	if (__histogram_##name != nullptr) {                                                       \
		__histogram_##name->addSampleEnd();                                                \
	}

/**
 * @def Adds a sample to a named data histogram.
 *
 * @param name The name of the histogram.
 * @param sample The sample value.
 */
#define OFI_HISTOGRAM_ADD_SAMPLE(name, sample)                                                     \
	if (__histogram_##name != nullptr) {                                                       \
		__histogram_##name->addSample(sample);                                             \
	}

/**
 * Macros for defining histogram and adding sample in one go (no need for seprate definition).
 * Use these macros unless you need non-default bin/clock settings and or units/factor settings.
 */

/** @def Declares and defines a named histogram and a scoped latency measurement in one go. */
#define OFI_DECLARE_SCOPED_LATENCY_HISTOGRAM(name)                                                 \
	OFI_DECLARE_LATENCY_HISTOGRAM(name);                                                       \
	OFI_SCOPED_LATENCY_HISTOGRAM(name);

/**
 * @def Declares and defines a named latency histogram, and adds a start point to it in one go.
 * @param name The name of the histogram.
 */
#define OFI_DECLARE_HISTOGRAM_START(name)                                                          \
	{                                                                                          \
		OFI_DECLARE_LATENCY_HISTOGRAM(name);                                               \
		OFI_HISTORGRAM_START(name);                                                        \
	}

/**
 * @def Declares and defines a named latency histogram, and adds an end point to it in one go.
 * @param name The name of the histogram.
 */
#define OFI_DECLARE_HISTOGRAM_END(name)                                                            \
	{                                                                                          \
		OFI_DECLARE_LATENCY_HISTOGRAM(name);                                               \
		OFI_HISTORGRAM_END(name);                                                          \
	}

/**
 * @def Declares and defines a named data histogram, and adds a sample to it in one go.
 *
 * @param name The name of the histogram.
 * @param sample The sample value.
 */
#define OFI_DECLARE_HISTOGRAM_ADD_SAMPLE(name, sample)                                             \
	{                                                                                          \
		OFI_DECLARE_DATA_HISTOGRAM(name);                                                  \
		if (__histogram_##name != nullptr) {                                               \
			__histogram_##name->addSample(sample);                                     \
		}                                                                                  \
	}
#else

// define all histogram macros to be empty
#define OFI_DECLARE_LATENCY_HISTOGRAM_EX(name, generator, clock_type, untis, factor)
#define OFI_DECLARE_LATENCY_HISTOGRAM(name)
#define OFI_DECLARE_DATA_HISTOGRAM(name, generator, untis, factor)

#define OFI_SCOPED_LATENCY_HISTOGRAM(name)
#define OFI_HISTOGRAM_START(name)
#define OFI_HISTOGRAM_END(name)
#define OFI_HISTOGRAM_ADD_SAMPLE(name, sample)

#define OFI_DECLARE_SCOPED_LATENCY_HISTOGRAM(name)
#define OFI_DECLARE_HISTOGRAM_START(name)
#define OFI_DECLARE_HISTOGRAM_END(name)
#define OFI_DECLARE_HISTOGRAM_ADD_SAMPLE(name, sample)

#endif // End ENABLE_HISTOGRAM == 1

#endif
