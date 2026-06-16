//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#ifndef NCCL_OFI_STATS_HISTOGRAM_DEF
#define NCCL_OFI_STATS_HISTOGRAM_DEF

// common definitions for histogram support

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

// column justification constants for table printing
enum class Justify { JUSTIFY_LEFT, JUSTIFY_CENTER, JUSTIFY_RIGHT };

#endif // NCCL_OFI_STATS_HISTOGRAM_DEF