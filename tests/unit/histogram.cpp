//
// Copyright (c) 2025-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#include "config.h"

#include <iostream>

#include "unit_test.h"
#include "nccl_ofi.h"
#include "stats/histogram.h"


#define CHECK_AND_EXIT(x)				      \
	if (!(x)) {					      \
		std::cerr << "Failure: " << #x << std::endl; \
		exit(1);				      \
	}


static void check_histogram(void)
{
	OFI_DECLARE_DATA_HISTOGRAM_EX(test_data_histogram,
		OFI_LINEAR_BIN_GENERATOR(2, 5, 0), "", 1);

	for (uint64_t i = 0; i <= 10; ++i) {
		OFI_HISTOGRAM_ADD_SAMPLE(test_data_histogram, i);
	}
	OFI_HISTOGRAM_ADD_SAMPLE(test_data_histogram, 0);

	auto h = OFI_HISTOGRAM_REF(test_data_histogram);
	CHECK_AND_EXIT(h->getBinCount() == 5);
	CHECK_AND_EXIT(h->getBinAt(0).getCount() == 3);
	CHECK_AND_EXIT(h->getBinAt(1).getCount() == 2);
	CHECK_AND_EXIT(h->getBinAt(2).getCount() == 2);
	CHECK_AND_EXIT(h->getBinAt(3).getCount() == 2);
	CHECK_AND_EXIT(h->getBinAt(4).getCount() == 3);

	h->print(PrintFormat::PF_TABLE);

	// no need to dispose of histogram, it will be deleted during app tear down
}

class TestClock {
public:
	static uint64_t getSysTimeNS()
	{
		return sClockValue;
	}

	static void advanceClock(uint64_t value)
	{
		sClockValue = value;
	}

private:
	static uint64_t sClockValue;
};

uint64_t TestClock::sClockValue = 0;


static void check_timer_histogram(void)
{
	OFI_DECLARE_LATENCY_HISTOGRAM_EX(test_latency_histogram,
		OFI_LINEAR_BIN_GENERATOR(10, 5, 0), TestClock, "ns", 1);

	// NOTE: we are testing that the histogram actually stores the time diff (rather than testing 
	// different clock time representation, as everything is in nanoseconds now)
	OFI_HISTORGRAM_START(test_latency_histogram);
	TestClock::advanceClock(1);
	uint64_t timePassedNanos = OFI_HISTOGRAM_REF(test_latency_histogram)->addSampleEnd();
	CHECK_AND_EXIT(timePassedNanos == 1);

	OFI_HISTORGRAM_START(test_latency_histogram);
	TestClock::advanceClock(1000);
	timePassedNanos = OFI_HISTOGRAM_REF(test_latency_histogram)->addSampleEnd();
	CHECK_AND_EXIT(timePassedNanos == 1000);

	OFI_HISTORGRAM_START(test_latency_histogram);
	TestClock::advanceClock(5000);
	timePassedNanos = OFI_HISTOGRAM_REF(test_latency_histogram)->addSampleEnd();
	CHECK_AND_EXIT(timePassedNanos == 5000);

	OFI_HISTORGRAM_START(test_latency_histogram);
	TestClock::advanceClock(10000);
	timePassedNanos = OFI_HISTOGRAM_REF(test_latency_histogram)->addSampleEnd();
	CHECK_AND_EXIT(timePassedNanos == 10000);

	OFI_HISTORGRAM_START(test_latency_histogram);
	TestClock::advanceClock(15000);
	timePassedNanos = OFI_HISTOGRAM_REF(test_latency_histogram)->addSampleEnd();
	CHECK_AND_EXIT(timePassedNanos == 15000);

	OFI_HISTORGRAM_START(test_latency_histogram);
	TestClock::advanceClock(100000);
	timePassedNanos = OFI_HISTOGRAM_REF(test_latency_histogram)->addSampleEnd();
	CHECK_AND_EXIT(timePassedNanos == 100000);

	OFI_HISTOGRAM_REF(test_latency_histogram)->print(PrintFormat::PF_JSON);

	// no need to dispose of histogram, it will be deleted during app tear down
}


int
main(int argc, char *argv[])
{
	unit_test_init();

	check_histogram();
	check_timer_histogram();

        return 0;
}
