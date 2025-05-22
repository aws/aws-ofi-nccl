//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#include "config.h"

#include <iostream>

#include "nccl_ofi.h"
#include "test-logger.h"
#include "stats/histogram.h"


#define CHECK_AND_EXIT(x)				      \
	if (!(x)) {					      \
		std::cerr << "Failure: " << #x << std::endl; \
		exit(1);				      \
	}

// wrapper around histogram to get access to the results to verify
template <typename T, class Binner>
class test_histogram : public histogram<T, Binner> {
public:
	test_histogram(const std::string& description_arg, Binner binner_arg) :
		histogram<T, Binner>(description_arg, binner_arg)
	{
	}

	const std::vector<std::size_t> & get_results(void)
	{
		return this->bins;
	}
};


static void check_histogram(void)
{
	using Binner = histogram_linear_binner<int>;

	test_histogram<int, Binner> histogram("testing!",
						   Binner(0, 2, 5));;

	histogram.insert(0);
	histogram.insert(1);
	histogram.insert(2);
	histogram.insert(3);
	histogram.insert(4);
	histogram.insert(5);
	histogram.insert(6);
	histogram.insert(7);
	histogram.insert(8);
	histogram.insert(9);
	histogram.insert(10);
	histogram.insert(0);

	auto results = histogram.get_results();
	CHECK_AND_EXIT(results.size() == 5);
	CHECK_AND_EXIT(results[0] == 3);
	CHECK_AND_EXIT(results[1] == 2);
	CHECK_AND_EXIT(results[2] == 2);
	CHECK_AND_EXIT(results[3] == 2);
	CHECK_AND_EXIT(results[4] == 3);

	histogram.print_stats();
}


class test_clock {
public:
	using rep = std::size_t;
	using period = std::nano;
	using duration = std::chrono::duration<rep, period>;
	using time_point = std::chrono::time_point<test_clock>;
	static constexpr bool is_steady = true;

	static time_point now() noexcept
	{
		return time_point(duration(std::chrono::nanoseconds(clock)));
	}

	static void advance_time(rep inc_val)
	{
		clock += inc_val;
	}

private:
	static rep clock;
};
test_clock::rep test_clock::clock = 0;


static void check_timer_histogram(void)
{
	using test_histogram = timer_histogram<histogram_linear_binner<std::size_t>,
						test_clock>;
	using Binner = histogram_linear_binner<test_histogram::rep>;

	test_histogram::rep time;

	test_histogram timer_histogram("timers!", Binner(0, 10, 5));

	timer_histogram.start_timer();
	test_clock::advance_time(1);
	time = timer_histogram.stop_timer();
	CHECK_AND_EXIT(time == 0);

	timer_histogram.start_timer();
	test_clock::advance_time(1000);
	time = timer_histogram.stop_timer();
	CHECK_AND_EXIT(time == 1);

	timer_histogram.start_timer();
	test_clock::advance_time(5000);
	time = timer_histogram.stop_timer();
	CHECK_AND_EXIT(time == 5);

	timer_histogram.start_timer();
	test_clock::advance_time(10000);
	time = timer_histogram.stop_timer();
	CHECK_AND_EXIT(time == 10);

	timer_histogram.start_timer();
	test_clock::advance_time(15000);
	time = timer_histogram.stop_timer();
	CHECK_AND_EXIT(time == 15);

	timer_histogram.start_timer();
	test_clock::advance_time(100000);
	time = timer_histogram.stop_timer();
	CHECK_AND_EXIT(time == 100);

	timer_histogram.print_stats();
}


int
main(int argc, char *argv[])
{
	ofi_log_function = logger;

	check_histogram();
	check_timer_histogram();

        return 0;
}
