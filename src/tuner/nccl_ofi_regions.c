/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <math.h>

#include "nccl_ofi_tuner.h"

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
static inline nccl_ofi_tuner_point_t vmadd(nccl_ofi_tuner_point_t a, double s, nccl_ofi_tuner_point_t b)
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

	double a = (vcross(x0, dx) - vcross(y0, dx)) / d;
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
ncclResult_t set_regions(nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx,
			 size_t num_regions,
			 const nccl_ofi_tuner_region_t regions[])
{
	nccl_ofi_tuner_ctx->num_regions = num_regions;
	nccl_ofi_tuner_ctx->regions = (nccl_ofi_tuner_region_t *)calloc(num_regions, sizeof(nccl_ofi_tuner_region_t));
	if (nccl_ofi_tuner_ctx->regions == NULL) {
		NCCL_OFI_WARN("Context regions allocation failed.");
		return ncclInternalError;
	}

	memcpy(nccl_ofi_tuner_ctx->regions, &regions[0], num_regions * sizeof(nccl_ofi_tuner_region_t));
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
	double projected_zx = m * z.x + c;

	if (projected_zx > z.y) {
		ret = (nccl_ofi_tuner_point_t){.x = z.x, .y = projected_zx};
	} else {
		ret = (nccl_ofi_tuner_point_t){.x = (z.y - c) / m, .y = z.y};
	}

	return ret;
}
