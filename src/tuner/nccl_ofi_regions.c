/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <math.h>

#include "tuner/nccl_ofi_tuner_region.h"
#include "nccl_ofi_param.h"


typedef struct nccl_ofi_tuner_region_dims {
	/* communicator size */
	size_t num_ranks;
	size_t num_nodes;
} nccl_ofi_tuner_region_dims_t;

typedef struct nccl_ofi_tuner_region_context {
	enum nccl_ofi_tuner_platform platform;
	struct nccl_ofi_tuner_region_dims dims;
	size_t num_regions[NCCL_NUM_FUNCTIONS];
	nccl_ofi_tuner_region_t *regions[NCCL_NUM_FUNCTIONS];
} nccl_ofi_tuner_region_context_t;

/* Vector subtraction */
static inline nccl_ofi_tuner_point_t vsub(nccl_ofi_tuner_point_t a, nccl_ofi_tuner_point_t b)
{
	return (nccl_ofi_tuner_point_t){.x = a.x - b.x, .y = a.y - b.y};
}

/* Vector dot product */
static inline double vdot(nccl_ofi_tuner_point_t a, nccl_ofi_tuner_point_t b)
{
	return a.x * b.x + a.y * b.y;
}

/* Magnitude of the cross product */
static inline double vcross(nccl_ofi_tuner_point_t a, nccl_ofi_tuner_point_t b)
{
	return a.x * b.y - a.y * b.x;
}

/* Returns a + s * b */
static inline nccl_ofi_tuner_point_t vmadd(nccl_ofi_tuner_point_t a, long double s, nccl_ofi_tuner_point_t b)
{
	nccl_ofi_tuner_point_t c;
	c.x = a.x + s * b.x;
	c.y = a.y + s * b.y;
	return c;
}

/*
 * @brief	Check if the x0->x1 segment crosses the y0->y1 segment.
 *
 * @param	x0
 *		point at the beginning of the first segment.
 * @param	x1
 *		point at the end of the first segment.
 * @param	y0
 *		point at the beginning of the second segment.
 * @param	y1
 *		point at the end of the second segment.
 * @param	eps
 *		small value used for comparisons.
 * @param	sect
 *		If it is a valid pointer, it will be set to the intesection point.
 *
 * @return	1 for intersect, -1 for not, 0 if the intersect point
 *		is too close to y0 or y1.
 */
static int intersect(nccl_ofi_tuner_point_t x0,
		     nccl_ofi_tuner_point_t x1,
		     nccl_ofi_tuner_point_t y0,
		     nccl_ofi_tuner_point_t y1,
		     double eps,
		     nccl_ofi_tuner_point_t *sect)
{
	nccl_ofi_tuner_point_t dx = vsub(x1, x0);
	nccl_ofi_tuner_point_t dy = vsub(y1, y0);
	double d = vcross(dy, dx);
	if (fabs(d) < eps) {
		return 0; /* Edges are parallel */
	}

	long double a = (long double) (vcross(x0, dx) - vcross(y0, dx))/d;
	if (sect) {
		*sect = vmadd(y0, a, dy);
	}

	if (a < -eps || a > 1 + eps) {
		return -1;
	}
	if (a < eps || a > 1 - eps) {
		return 0;
	}

	a = (vcross(x0, dy) - vcross(y0, dy)) / d;
	if (a < 0 || a > 1) {
		return -1;
	}

	return 1;
}

/*
 * @brief	Distance between x and nearest point on y0->y1 segment.
 *
 * @param	x
 *		point.
 * @param	y0
 *		point at the beginning of the segment.
 * @param	y1
 *		point at the end of the segment.
 * @param	eps
 *		small value used for comparisons.
 *
 * @return	distance or infinity if the point lies outside the segment.
 */
static inline double distance(nccl_ofi_tuner_point_t x,
			      nccl_ofi_tuner_point_t y0,
			      nccl_ofi_tuner_point_t y1,
			      double eps)
{
	nccl_ofi_tuner_point_t dy = vsub(y1, y0);
	nccl_ofi_tuner_point_t x1, s;
	int r;

	x1.x = x.x + dy.y;
	x1.y = x.y - dy.x;
	r = intersect(x, x1, y0, y1, eps, &s);
	if (r == -1) {
		return HUGE_VAL;
	}
	s = vsub(s, x);
	return sqrt(vdot(s, s));
}

/*
 * @brief	Ray-casting algorithm to check if a pointer is inside, outside or
 * 		on the edge of a region.
 * 		For a detailed explanation, check:
 * 		https://rosettacode.org/wiki/Ray-casting_algorithm
 *
 * @param	point
 *		(nBytes, num_ranks) coordinates.
 * @param	region
 *		Region, which is a list of vertices.
 *
 * @return	1 for inside,
 * 		-1 for outside
 * 		0 for on edge.
 */
int is_inside_region(nccl_ofi_tuner_point_t point, nccl_ofi_tuner_region_t *region)
{
	assert(region->num_vertices > 1);

	size_t i, k;
	nccl_ofi_tuner_point_t *pv;
	double min_x, max_x, min_y, max_y;
	const double eps = 1e-10;

	for (i = 0; i < region->num_vertices; i++) {
		k = (i + 1) % region->num_vertices;
		min_x = distance(point, region->vertices[i], region->vertices[k], eps);
		if (min_x < eps) {
			/* Point on the edge */
			return 0;
		}
	}

	min_x = max_x = region->vertices[0].x;
	min_y = max_y = region->vertices[1].y;

	/*
	 * This is a pre-screening step to quickly mark as external all points
	 * outside the bounding box (a rectangle drawn around the polygon).
	 */
	for (i = 0, pv = region->vertices; i < region->num_vertices; i++, pv++) {
		if (pv->x > max_x) {
			max_x = pv->x;
		}
		if (pv->x < min_x) {
			min_x = pv->x;
		}
		if (pv->y > max_y) {
			max_y = pv->y;
		}
		if (pv->y < min_y) {
			min_y = pv->y;
		}
	}
	if (point.x < min_x || point.x > max_x || point.y < min_y || point.y > max_y) {
		/* Point outside the bounding box */
		return -1;
	}

	/* Pick a point far enough to be outside any polygon */
	nccl_ofi_tuner_point_t e = {.x = 2.0 * TUNER_MAX_SIZE, .y = 2.0 * TUNER_MAX_RANKS};

	int crosses = 0;
	int intersectResult = -1;
	for (i = 0; i < region->num_vertices; i++) {
		k = (i + 1) % region->num_vertices;
		intersectResult = intersect(point, e, region->vertices[i], region->vertices[k], eps, 0);

		assert(intersectResult == 1 || intersectResult == -1);

		if (intersectResult == 1) {
			crosses++;
		}
	}

	return (crosses & 1) ? 1 : -1;
}

/* Allocate and copy regions */
static ncclResult_t set_regions(nccl_ofi_tuner_region_context_t *region_ctx,
				ncclFunc_t collType,
				size_t num_regions,
				const nccl_ofi_tuner_region_t regions[])
{
	assert(collType < NCCL_NUM_FUNCTIONS);
	region_ctx->num_regions[collType] = num_regions;
	region_ctx->regions[collType] = (nccl_ofi_tuner_region_t *)calloc(num_regions, sizeof(nccl_ofi_tuner_region_t));
	if (region_ctx->regions[collType] == NULL) {
		NCCL_OFI_WARN("Context regions allocation failed.");
		return ncclInternalError;
	}

	memcpy(region_ctx->regions[collType], &regions[0], num_regions * sizeof(nccl_ofi_tuner_region_t));
	return ncclSuccess;
}

/*
 * Given 2 points a and b, find the line connecting them.
 * Then find the farthest point on that line with either the same x or same y coordinate
 * of the third point z.
 */
nccl_ofi_tuner_point_t extend_region(nccl_ofi_tuner_point_t a, nccl_ofi_tuner_point_t b, nccl_ofi_tuner_point_t z)
{
	nccl_ofi_tuner_point_t ret;

	if (a.x == b.x) {
		/* a and b are on the same vertical line */
		ret = (nccl_ofi_tuner_point_t){.x = a.x, .y = z.y};
		return ret;
	}

	if (a.y == b.y) {
		/* a and b are on the same horizontal line */
		ret = (nccl_ofi_tuner_point_t){.x = z.x, .y = a.y};
		return ret;
	}

	double m = (a.y - b.y) / (a.x - b.x);
	double c = b.y - m * b.x;
	double projected_zy = m * z.x + c;

	if (projected_zy < z.y) {
		ret = (nccl_ofi_tuner_point_t){.x = z.x, .y = projected_zy};
	} else {
		ret = (nccl_ofi_tuner_point_t){.x = (z.y - c) / m, .y = z.y};
	}

	return ret;
}


/**
 * P5en platform specific Regions
 */
static ncclResult_t region_init_internal_p5en(nccl_ofi_tuner_region_context_t *region_ctx)
{
	ncclResult_t ret = ncclSuccess;
	ncclFunc_t collType;
	size_t nRanks = region_ctx->dims.num_ranks;
	size_t nNodes = region_ctx->dims.num_nodes;

	if (nRanks == 8 * nNodes) {
		{
			collType = ncclFuncAllReduce;
			nccl_ofi_tuner_point_t extended_tree_ll =
				extend_region((nccl_ofi_tuner_point_t){262144, 192},
							  (nccl_ofi_tuner_point_t){262144, 1024},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			nccl_ofi_tuner_point_t extended_tree_ll128 =
				extend_region((nccl_ofi_tuner_point_t){150994944, 128},
							  (nccl_ofi_tuner_point_t){251658240, 1024},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			nccl_ofi_tuner_point_t extended_nvlstree_simple =
				extend_region((nccl_ofi_tuner_point_t){6442450944, 256},
							  (nccl_ofi_tuner_point_t){17179869184, 768},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_LL,
				 .num_vertices = 4,
				 .vertices = {{0, 16}, {262144, 16}, {262144, 1024}, extended_tree_ll}},
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 7,
				 .vertices = {extended_tree_ll,
							  {262144, 1024},
							  {262144, 16},
							  {14680064, 16},
							  {150994944, 128},
							  {251658240, 1024},
							  extended_tree_ll128}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 5,
				 .vertices = {{150994944, 128},
							  {14680064, 16},
							  {33554432, 16},
							  {536870912, 32},
							  {536870912, 128}}},
				{.algorithm = NCCL_ALGO_NVLS_TREE,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 8,
				 .vertices = {extended_tree_ll128,
							  {251658240, 1024},
							  {150994944, 128},
							  {536870912, 128},
							  {536870912, 32},
							  {6442450944, 256},
							  {17179869184, 768},
							  extended_nvlstree_simple}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 7,
				 .vertices = {extended_nvlstree_simple,
							  {17179869184, 768},
							  {6442450944, 256},
							  {536870912, 32},
							  {33554432, 16},
							  {1073741824, 16},
							  {TUNER_MAX_SIZE, 16}}}};
			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
		{
			collType = ncclFuncAllGather;
			nccl_ofi_tuner_point_t extended_ring_ll =
				extend_region((nccl_ofi_tuner_point_t){8388608, 256},
							  (nccl_ofi_tuner_point_t){33554432, 1024},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			nccl_ofi_tuner_point_t extended_ring_ll128 =
				extend_region((nccl_ofi_tuner_point_t){8589934592, 512},
							  (nccl_ofi_tuner_point_t){17179869184, 1024},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL,
				 .num_vertices = 7,
				 .vertices = {{0, 16}, {131072, 16}, {262144, 32}, {8388608, 256}, {33554432, 1024}, extended_ring_ll, {0, TUNER_MAX_RANKS}}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 10,
				 .vertices = {extended_ring_ll,
							  {33554432, 1024},
							  {8388608, 256},
							  {262144, 32},
							  {131072, 16},
							  {268435456, 16},
							  {2147483648, 128},
							  {8589934592, 512},
							  {17179869184, 1024},
							  extended_ring_ll128}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 6,
				 .vertices = {extended_ring_ll128,
							  {17179869184, 1024},
							  {8589934592, 512},
							  {268435456, 16},
							  {17179869184, 16},
							  {TUNER_MAX_SIZE, 16}}}};
			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
		{
			collType = ncclFuncReduceScatter;
			nccl_ofi_tuner_point_t extended_ring_ll =
				extend_region((nccl_ofi_tuner_point_t){8388608, 256},
							  (nccl_ofi_tuner_point_t){33554432, 1024},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			nccl_ofi_tuner_point_t extended_ring_ll128 =
				extend_region((nccl_ofi_tuner_point_t){8589934592, 512},
							  (nccl_ofi_tuner_point_t){17179869184, 1024},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL,
				 .num_vertices = 7,
				 .vertices = {{0, 16}, {131072, 16}, {262144, 32}, {8388608, 256}, {33554432, 1024}, extended_ring_ll, {0, TUNER_MAX_RANKS}}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 10,
				 .vertices = {extended_ring_ll,
							  {33554432, 1024},
							  {8388608, 256},
							  {262144, 32},
							  {131072, 16},
							  {268435456, 16},
							  {2147483648, 128},
							  {8589934592, 512},
							  {17179869184, 1024},
							  extended_ring_ll128}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 6,
				 .vertices = {extended_ring_ll128,
							  {17179869184, 1024},
							  {8589934592, 512},
							  {268435456, 16},
							  {17179869184, 16},
							  {TUNER_MAX_SIZE, 16}}}};
			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
	} else if (nRanks == nNodes) {
		{
			collType = ncclFuncAllReduce;
			nccl_ofi_tuner_point_t extended_tree_ll128 =
				extend_region((nccl_ofi_tuner_point_t){524288, 8},
							  (nccl_ofi_tuner_point_t){1048576, 96},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			nccl_ofi_tuner_point_t extended_tree_simple =
				extend_region((nccl_ofi_tuner_point_t){8388608, 32},
							  (nccl_ofi_tuner_point_t){33554432, 128},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			nccl_ofi_tuner_point_t extended_ring_ll128 =
				extend_region((nccl_ofi_tuner_point_t){50331648, 16},
							  (nccl_ofi_tuner_point_t){301989888, 128},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_LL,
				 .num_vertices = 4,
				 .vertices = {{0, 2}, {65536, 2}, {65536, 64}, {65536, TUNER_MAX_RANKS}}},
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 7,
				 .vertices = {{65536, TUNER_MAX_RANKS},
							  {65536, 64},
							  {65536, 2},
							  {262144, 2},
							  {524288, 8},
							  {1048576, 96},
							  extended_tree_ll128}},
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 7,
				 .vertices = {extended_tree_ll128,
							  {1048576, 768},
							  {524288, 8},
							  {262144, 2},
							  {8388608, 32},
							  {33554432, 128},
							  extended_tree_simple}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 8,
				 .vertices = {extended_tree_simple,
							  {33554432, 128},
							  {8388608, 32},
							  {262144, 2},
							  {6291456, 2},
							  {50331648, 16},
							  {301989888, 128},
							  extended_ring_ll128}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 5,
				 .vertices = {extended_ring_ll128,
							  {301989888, 128},
							  {50331648, 16},
							  {6291456, 2},
							  {TUNER_MAX_SIZE, 2}}}};
			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess){
				goto exit;
			}
		}
		{
			collType = ncclFuncAllGather;
			nccl_ofi_tuner_point_t extended_pat_simple =
				extend_region((nccl_ofi_tuner_point_t){50331648, 64},
							  (nccl_ofi_tuner_point_t){117440512, 128},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			nccl_ofi_tuner_point_t extended_ring_ll128 =
				extend_region((nccl_ofi_tuner_point_t){50331648, 16},
							  (nccl_ofi_tuner_point_t){301989888, 128},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {
				{.algorithm = NCCL_ALGO_PAT,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 10,
				 .vertices = {{0, 2},
							  {65536, 2},
							  {1048576, 2},
							  {16777216, 32},
							  {50331648, 64},
							  {117440512, 128},
							  extended_pat_simple,
							  {TUNER_MAX_SIZE, TUNER_MAX_RANKS},
							  {65536, TUNER_MAX_RANKS},
							  {0, TUNER_MAX_RANKS}}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 9,
				 .vertices = {extended_pat_simple,
							  {117440512, 128},
							  {50331648, 64},
							  {16777216, 32},
							  {1048576, 2},
							  {4194304, 2},
							  {50331648, 16},
							  {301989888, 128},
							  extended_ring_ll128}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 5,
				 .vertices = {extended_ring_ll128,
							  {301989888, 128},
							  {50331648, 16},
							  {4194304, 2},
							  {TUNER_MAX_SIZE, 2}}}};
			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
		{
			collType = ncclFuncReduceScatter;
			nccl_ofi_tuner_point_t extended_pat_simple =
				extend_region((nccl_ofi_tuner_point_t){50331648, 64},
							  (nccl_ofi_tuner_point_t){117440512, 128},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			nccl_ofi_tuner_point_t extended_ring_ll128 =
				extend_region((nccl_ofi_tuner_point_t){50331648, 16},
							  (nccl_ofi_tuner_point_t){301989888, 128},
							  (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			const nccl_ofi_tuner_region_t regions[] = {
				{.algorithm = NCCL_ALGO_PAT,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 10,
				 .vertices = {{0, 2},
							  {65536, 2},
							  {1048576, 2},
							  {16777216, 32},
							  {50331648, 64},
							  {117440512, 128},
							  extended_pat_simple,
							  {TUNER_MAX_SIZE, TUNER_MAX_RANKS},
							  {65536, TUNER_MAX_RANKS},
							  {0, TUNER_MAX_RANKS}}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 9,
				 .vertices = {extended_pat_simple,
							  {117440512, 128},
							  {50331648, 64},
							  {16777216, 32},
							  {1048576, 2},
							  {4194304, 2},
							  {50331648, 16},
							  {301989888, 128},
							  extended_ring_ll128}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 5,
				 .vertices = {extended_ring_ll128,
							  {301989888, 128},
							  {50331648, 16},
							  {4194304, 2},
							  {TUNER_MAX_SIZE, 2}}}};
			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
	}
exit:
	return ret;
}


/**
 * P5 and P5e platform specific Regions
 */
static ncclResult_t region_init_internal_p5_p5e(nccl_ofi_tuner_region_context_t *region_ctx)
{
	ncclResult_t ret = ncclSuccess;
	ncclFunc_t collType;
	size_t nRanks = region_ctx->dims.num_ranks;
	size_t nNodes = region_ctx->dims.num_nodes;

	if (nRanks == 8 * nNodes) {
		{
			collType = ncclFuncAllReduce;
			nccl_ofi_tuner_point_t extended_tree_ll128 =
				extend_region((nccl_ofi_tuner_point_t){402653184, 2048},
					      (nccl_ofi_tuner_point_t){402653184, 4096},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			nccl_ofi_tuner_point_t extended_nvlstree_simple_1 =
				extend_region((nccl_ofi_tuner_point_t){8053063680, 160},
					      (nccl_ofi_tuner_point_t){9663676416, 192},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			nccl_ofi_tuner_point_t extended_nvlstree_simple_2 =
				extend_region((nccl_ofi_tuner_point_t){402653184, 2048},
					      (nccl_ofi_tuner_point_t){402653184, 4096},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			nccl_ofi_tuner_point_t extended_ring_simple =
				extend_region((nccl_ofi_tuner_point_t){8053063680, 160},
					      (nccl_ofi_tuner_point_t){9663676416, 192},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 12,
				 .vertices = {{0, 16},
					      {31457280, 16},
					      {37748736, 32},
					      {117440512, 64},
					      {301989888, 128},
					      {301989888, 256},
					      {335544320, 512},
					      {536870912, 1024},
					      {402653184, 2048},
					      {402653184, 4096},
					      extended_tree_ll128,
					      {0, extended_tree_ll128.y}}},
				{.algorithm = NCCL_ALGO_NVLS_TREE,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 3,
				 .vertices = {{31457281, 16}, {TUNER_MAX_SIZE, 16}, {31457281, 16}}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 11,
				 .vertices = {{31457280, 17},
					      {1073741824, 17},
					      {2147483648, 64},
					      {2147483648, 128},
					      {1342177280, 160},
					      {2147483648, 256},
					      {1074790400, 256},
					      {444596224, 160},
					      {301989888, 128},
					      {117440512, 64},
					      {37748736, 32}}},
				{.algorithm = NCCL_ALGO_NVLS_TREE,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 17,
				 .vertices = {{2147483648, 128},
					      {6442450944, 128},
					      {8053063680, 160},
					      {9663676416, 192},
					      extended_nvlstree_simple_1,
					      extended_nvlstree_simple_2,
					      {402653184, 4096},
					      {402653184, 2048},
					      {536870912, 1024},
					      {335544320, 512},
					      {301989888, 256},
					      {310378496, 160},
					      {444596224, 160},
					      {1074790400, 256},
					      {2684354560, 256},
					      {2147483648, 224},
					      {1342177280, 160}}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 7,
				 .vertices = {{1073741824, 17},
					      {extended_ring_simple.x, 17},
					      extended_ring_simple,
					      {9663676416, 192},
					      {8053063680, 160},
					      {2684354560, 64},
					      {1610612736, 32}}}};

			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
	} else if (nRanks == 2 * nNodes) {
		{
			collType = ncclFuncAllReduce;
			nccl_ofi_tuner_point_t extended_tree_ll128 =
				extend_region((nccl_ofi_tuner_point_t){88160256, 128},
					      (nccl_ofi_tuner_point_t){178163712, 256},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			nccl_ofi_tuner_point_t extended_tree_simple_1 =
				extend_region((nccl_ofi_tuner_point_t){787480576, 128},
					      (nccl_ofi_tuner_point_t){1073741824, 256},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			nccl_ofi_tuner_point_t extended_tree_simple_2 =
				extend_region((nccl_ofi_tuner_point_t){257114112, 128},
					      (nccl_ofi_tuner_point_t){269484032, 256},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			nccl_ofi_tuner_point_t extended_nvlstree_simple =
				extend_region((nccl_ofi_tuner_point_t){787480576, 128},
					      (nccl_ofi_tuner_point_t){1073741824, 256},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 11,
				 .vertices = {{0, 4},
					      {1314816, 4},
					      {1051648, 8},
					      {1051648, 12},
					      {2367488, 16},
					      {5525504, 32},
					      {9473024, 64},
					      {88160256, 128},
					      {178163712, 256},
					      extended_tree_ll128,
					      {0, extended_tree_ll128.y}}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 14,
				 .vertices = {{1314816, 4},
					      {19736576, 4},
					      {41842688, 8},
					      {296747008, 64},
					      {257114112, 128},
					      {269484032, 256},
					      {178163712, 256},
					      {88160256, 128},
					      {9473024, 64},
					      {5525504, 32},
					      {2367488, 16},
					      {1051648, 12},
					      {1051648, 8},
					      {1314816, 4}}},
				{.algorithm = NCCL_ALGO_NVLS_TREE,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 6,
				 .vertices = {{19736576, 4},
					      {81844224, 4},
					      {275775488, 8},
					      {275775488, 48},
					      {296747008, 64},
					      {41842688, 8}}},
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 3,
				 .vertices = {{81844224, 4}, {269484032, 4}, {81844224, 4}}},
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 3,
				 .vertices = {{269484032, 4}, {TUNER_MAX_SIZE, 4}, {269484032, 4}}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 10,
				 .vertices = {{81844224, 5},
					      {TUNER_MAX_SIZE, 5},
					      {TUNER_MAX_SIZE, 32},
					      {1073741824, 40},
					      {1073741824, 128},
					      {787480576, 128},
					      {296747008, 64},
					      {275775488, 48},
					      {275775488, 8},
					      {81844224, 5}}},
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 7,
				 .vertices = {{296747008, 64},
					      {787480576, 128},
					      {1073741824, 256},
					      extended_tree_simple_1,
					      extended_tree_simple_2,
					      {269484032, 256},
					      {257114112, 128}}},
				{.algorithm = NCCL_ALGO_NVLS_TREE,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 6,
				 .vertices = {extended_nvlstree_simple,
					      {1073741824, 256},
					      {787480576, 128},
					      {1073741824, 128},
					      {1073741824, 40},
					      {TUNER_MAX_SIZE, 32}}}};

			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
	} else if (nRanks == nNodes) {
		{
			collType = ncclFuncAllReduce;
			nccl_ofi_tuner_point_t extended_tree_ll128 =
				extend_region((nccl_ofi_tuner_point_t){9999360, 64},
					      (nccl_ofi_tuner_point_t){119477248, 128},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
			nccl_ofi_tuner_point_t extended_ring_ll128 =
				extend_region((nccl_ofi_tuner_point_t){4736000, 2},
					      (nccl_ofi_tuner_point_t){269484032, 128},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {
				{.algorithm = NCCL_ALGO_TREE,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 5,
				 .vertices = {{0, 16}, {2367488, 16}, {9999360, 64}, {119477248, 128}, extended_tree_ll128}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_LL128,
				 .num_vertices = 9,
				 .vertices = {{0, 2},
					      {4736000, 2},
					      {269484032, 128},
					      extended_ring_ll128,
					      extended_tree_ll128,
					      {119477248, 128},
					      {9999360, 64},
					      {2367488, 16},
					      {0, 16}}},
				{.algorithm = NCCL_ALGO_RING,
				 .protocol = NCCL_PROTO_SIMPLE,
				 .num_vertices = 4,
				 .vertices = {{4736000, 2}, {TUNER_MAX_SIZE, 2}, extended_ring_ll128, {269484032, 128}}}};

			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
		{
			collType = ncclFuncAllGather;
			nccl_ofi_tuner_point_t extended_ring_simple =
				extend_region((nccl_ofi_tuner_point_t){4194304, 2},
					      (nccl_ofi_tuner_point_t){8589934592, 2048},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {{.algorithm = NCCL_ALGO_RING,
								    .protocol = NCCL_PROTO_SIMPLE,
								    .num_vertices = 4,
								    .vertices = {{4194304, 2},
										 {TUNER_MAX_SIZE, 2},
										 extended_ring_simple,
										 {8589934592, 2048}}}};

			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
		{
			collType = ncclFuncReduceScatter;
			nccl_ofi_tuner_point_t extended_ring_simple =
				extend_region((nccl_ofi_tuner_point_t){8388608, 2},
					      (nccl_ofi_tuner_point_t){4294967296, 1024},
					      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

			const nccl_ofi_tuner_region_t regions[] = {{.algorithm = NCCL_ALGO_RING,
								    .protocol = NCCL_PROTO_SIMPLE,
								    .num_vertices = 4,
								    .vertices = {{8388608, 2},
										 {TUNER_MAX_SIZE, 2},
										 extended_ring_simple,
										 {4294967296, 1024}}}};

			ret = set_regions(region_ctx, collType, sizeof(regions) / sizeof(regions[0]), regions);
			if (ret != ncclSuccess) {
				goto exit;
			}
		}
	} else {
		/* Fall back to NCCL's tuner, so no regions */
	}

exit:
	return ret;
}


/*****************************************************************************
 *****************************************************************************
 *        functions that are called by common tuner code start here
 *****************************************************************************
 *****************************************************************************/

bool is_region_supported(enum nccl_ofi_tuner_platform platform, size_t nRanks, size_t nNodes)
{
	if (platform == NCCL_OFI_TUNER_P5_P5E || platform == NCCL_OFI_TUNER_P5EN) {
		return true;
	}

	return false;
}

ncclResult_t region_get_coll_info_internal_v2(nccl_ofi_tuner_context_t *ctx,
					      ncclFunc_t collType,
					      size_t nBytes,
					      int collNetSupport,
					      int nvlsSupport,
					      int numPipeOps,
					      int *algorithm,
					      int *protocol,
					      int *nChannels)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_tuner_region_context_t *region_ctx = (nccl_ofi_tuner_region_context_t *)ctx->type_ctx;
	int in_out = -1;
	nccl_ofi_tuner_point_t p;

	if (region_ctx == NULL || region_ctx->regions[collType] == NULL) {
		/* we do not update cost table. Fall back to NCCL's tuner */
		NCCL_OFI_INFO(NCCL_TUNING, "Region Context is not ready. Fall back to NCCL's tuner.");
		ret = ncclSuccess;
		goto exit;
	}

	p.x = (double)nBytes;
	p.y = (double)region_ctx->dims.num_ranks;

	/* Check all regions */
	for (size_t i = 0; i < region_ctx->num_regions[collType] && in_out < 0; i++) {
		/* PAT is not supported in V2 tuner, in this case revert to nccl internal tuner */
		if (region_ctx->regions[collType][i].algorithm == NCCL_ALGO_PAT) {
			continue;
		}
		if (region_ctx->regions[collType][i].algorithm == NCCL_ALGO_NVLS_TREE && nvlsSupport == 0) {
			continue;
		}

		in_out = is_inside_region(p, &region_ctx->regions[collType][i]);
		if (in_out >= 0) {
			*algorithm = region_ctx->regions[collType][i].algorithm;
			*protocol = region_ctx->regions[collType][i].protocol;

			NCCL_OFI_INFO(NCCL_TUNING,
					"Region TUner choosing algo %d proto %d with cost %.8f µsecs for coll %d size %ld.",
					*algorithm,
					*protocol,
					0.0,
					collType,
					nBytes);
		}
	}

	if (in_out < 0) {
		NCCL_OFI_INFO(NCCL_TUNING, "Falling back to NCCL's tuner for coll %d size %ld.", collType, nBytes);
	}

exit:
	return ret;
}

ncclResult_t region_get_coll_info_internal_v3(nccl_ofi_tuner_context_t *ctx,
					   ncclFunc_t collType,
					   size_t nBytes,
					   int numPipeOps,
					   float **collCostTable,
					   int numAlgo,
					   int numProto,
					   int *nChannels)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_tuner_region_context_t *region_ctx = (nccl_ofi_tuner_region_context_t *)ctx->type_ctx;
	float(*table)[NCCL_NUM_PROTOCOLS] = (float(*)[NCCL_NUM_PROTOCOLS])collCostTable;
	int in_out = -1;
	int algorithm = NCCL_ALGO_UNDEF;
	int protocol = NCCL_PROTO_UNDEF;
	nccl_ofi_tuner_point_t p;

	if (region_ctx == NULL || region_ctx->regions[collType] == NULL) {
		/* we do not update cost table. Fall back to NCCL's tuner */
		NCCL_OFI_INFO(NCCL_TUNING, "Region Context is not ready. Fall back to NCCL's tuner.");
		ret = ncclSuccess;
		goto exit;
	}

	p.x = (double)nBytes;
	p.y = (double)region_ctx->dims.num_ranks;

	/* Check all regions */
	for (size_t i = 0; i < region_ctx->num_regions[collType] && in_out < 0; i++) {
		algorithm = region_ctx->regions[collType][i].algorithm;
		protocol = region_ctx->regions[collType][i].protocol;
		if (algorithm >= numAlgo || protocol >= numProto ||
		    table[algorithm][protocol] == NCCL_ALGO_PROTO_IGNORE) {
			/* Either NCCL says this combination is not valid/applicable or the algorithm or protocol is
			 * not in the table, hence it is not supported by this NCCL version. */
			continue;
		}

		in_out = is_inside_region(p, &region_ctx->regions[collType][i]);
		if (in_out >= 0) {
			table[algorithm][protocol] = 0.0;

			NCCL_OFI_INFO(NCCL_TUNING,
				      "Region Tuner choosing algo %d proto %d with cost %.8f µsecs for coll %d size %ld.",
				      algorithm,
				      protocol,
				      table[algorithm][protocol],
				      collType,
				      nBytes);
		}
	}

	if (in_out < 0) {
		NCCL_OFI_INFO(NCCL_TUNING, "Falling back to NCCL's tuner for coll %d size %ld.", collType, nBytes);
	}

exit:
	return ret;
}

ncclResult_t region_destroy_internal(nccl_ofi_tuner_context_t *ctx)
{
	nccl_ofi_tuner_region_context_t *region_ctx = (nccl_ofi_tuner_region_context_t *)ctx->type_ctx;

	if (region_ctx != NULL) {
		for (int collType = 0; collType < NCCL_NUM_FUNCTIONS; collType++) {
			if (region_ctx->regions[collType] != NULL) {
				free(region_ctx->regions[collType]);
			}
		}
		free(region_ctx);
	}

	return ncclSuccess;
}

ncclResult_t region_init_internal(nccl_ofi_tuner_context_t *ctx, enum nccl_ofi_tuner_platform platform,
				  size_t nRanks, size_t nNodes)
{
	ncclResult_t ret = ncclSuccess;

	nccl_ofi_tuner_region_context_t *region_ctx =
		(nccl_ofi_tuner_region_context_t *)calloc(1, sizeof(nccl_ofi_tuner_region_context_t));
	if (region_ctx == NULL) {
		NCCL_OFI_WARN("Region Context allocation failed.");
		ret = ncclInternalError;
		goto exit;
	}
	ctx->type_ctx = (void *)region_ctx;
	region_ctx->dims.num_ranks = nRanks;
	region_ctx->dims.num_nodes = nNodes;
	region_ctx->platform = platform;

	/* Define regions where a certain combination of algorithm and protocol
	 * should be used. Any point not covered by any region would fall back
	 * to NCCL's default tuner. The order of the regions is important in case
	 * of overlapping regions, since this will return the first region which
	 * includes that point. */
	if (platform == NCCL_OFI_TUNER_P5_P5E) {
		ret = region_init_internal_p5_p5e(region_ctx);
	} else if (platform == NCCL_OFI_TUNER_P5EN) {
		ret = region_init_internal_p5en(region_ctx);
	} else {
		ret = ncclInternalError;
		goto exit;
	}

	NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Region Tuner init (platform %d): comm with %ld ranks and %ld nodes.",
		      platform, nRanks, nNodes);

exit:
	if (ret != ncclSuccess && region_ctx != NULL) {
		region_destroy_internal(ctx);
	}

	return ret;
}
