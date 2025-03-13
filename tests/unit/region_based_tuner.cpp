/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "tuner/nccl_ofi_tuner_region.h"
#include "nccl_ofi_param.h"

static int test_extend_region(void)
{
    nccl_ofi_tuner_point_t extended_point;
    double slope;
    double projected_x, projected_y;

    /* Extend the line on the x-axis */
    extended_point = extend_region((nccl_ofi_tuner_point_t){2, 8},
                                   (nccl_ofi_tuner_point_t){4, 8},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    if (extended_point.x != TUNER_MAX_SIZE || extended_point.y != 8) {
        printf("X-Axis Extend Test Failed : Extended Points : x = %f y = %f\n", extended_point.x, extended_point.y);
        return -1;
    }

    /* Extend the line on the y-axis*/
    extended_point = extend_region((nccl_ofi_tuner_point_t){8, 2},
                                   (nccl_ofi_tuner_point_t){8, 4},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    if (extended_point.x != 8 || extended_point.y != TUNER_MAX_RANKS) {
        printf("Y-Axis Extend Test Failed : Extended Points : x = %f y = %f\n", extended_point.x, extended_point.y);
        return -1;
    }

    /* Extend the line to TUNER_MAX_SIZE (x-axis) */
    extended_point = extend_region((nccl_ofi_tuner_point_t){8, 64},
                                   (nccl_ofi_tuner_point_t){8290304, 72},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    slope = (72.0 - 64.0) / (8290304.0 - 8.0); // slope = (y2 - y1)/(x2 - x1)
    // y3 = mx3 + c and substitute for m=(y2-y1)/(x2-x1) and c = y2 - mx2
    projected_y = 72.0 + slope * (TUNER_MAX_SIZE - 8290304.0); // y3 = y2 + mx3 - mx2
    if (extended_point.x != TUNER_MAX_SIZE || extended_point.y != projected_y) {
        printf("X-Axis Upper Bound Test Failed : Extended Points : x = %f y = %f\n", extended_point.x, extended_point.y);
        return -1;
    }

    /* Extend the line to TUNER_MAX_RANKS (y-axis) */
    extended_point = extend_region((nccl_ofi_tuner_point_t){8, 64},
                                   (nccl_ofi_tuner_point_t){16, 1024},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    slope = (1024.0 - 64.0) / (16.0 - 8.0);
    projected_x = ((TUNER_MAX_RANKS - 1024.0) / slope) + 16;
    if (extended_point.x != projected_x || extended_point.y != TUNER_MAX_RANKS) {
        printf("X-Axis Upper Bound Test Failed : Extended Points : x = %f y = %f\n", extended_point.x, extended_point.y);
        return -1;
    }

    return 0;
}

/*
* (0, TUNER_MAX_RANKS)
|                                                        |
|--------------------------------------------------------|--- (TUNER_MAX_SIZE, TUNER_MAX_RANKS)
|                                e([4M, 16],[288M, 128]) *
|                                                   .    |
|                                              .         |
|                                       *p1(288M, 128)   |
|                                 .                      |
|                           .                            |
|                    *p2(48M, 16)                        |
|                 .                                      |
|            .                                           |
|--------*------*----------*----------*---------*--------*---
|    p3(4M, 2)                                           |p4(TUNER_MAX_SIZE, 2))
|                                                        |
*/
static int test_is_inside_region(void) {
    nccl_ofi_tuner_point_t p1_288M_128 = {288.0 * 1024 * 1024, 128};
    nccl_ofi_tuner_point_t p2_38M_16 = {48.0 * 1024 * 1024, 16};
    nccl_ofi_tuner_point_t p3_4M_2 = {4.0 * 1024 * 1024, 2};
    nccl_ofi_tuner_point_t p5_maxM_2 = {TUNER_MAX_SIZE, 2};
    nccl_ofi_tuner_point_t e_48M_16_288M_128 = extend_region(
        (nccl_ofi_tuner_point_t){(double)48.0 * 1024 * 1024, 16},
        (nccl_ofi_tuner_point_t){(double)288.0 * 1024 * 1024, 128},
        (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

    nccl_ofi_tuner_region_t region = {
        .algorithm = NCCL_ALGO_RING,
        .protocol = NCCL_PROTO_SIMPLE,
        .num_vertices = 5,
        .vertices = {e_48M_16_288M_128,
                     p1_288M_128,
                     p2_38M_16,
                     p3_4M_2,
                     p5_maxM_2}};

    /* Points on the vertices of the polygon should be classified to be on the edge of the region */
    if (is_inside_region(e_48M_16_288M_128, &region) != 0)
        return -1;
    if (is_inside_region(p1_288M_128, &region) != 0)
        return -1;
    if (is_inside_region(p2_38M_16, &region) != 0)
        return -1;
    if (is_inside_region(p3_4M_2, &region) != 0)
        return -1;
    if (is_inside_region(p5_maxM_2, &region) != 0)
        return -1;

    printf("All points on the vertices of the polygon are detected correcltly\n");

    /* Points on the edge of the polygon are classified to be on the edge of the region.
    To find the points on the edge of the polygons:
    1. Consider two vertices of the polygon
    2. Calculate the slope and y-intercept of the line.
    3. Using the equation y = mx + c, get multiple points on the line in powers of 2.
    */
    for (size_t i = 0; i < region.num_vertices; i++) {
        size_t k = (i + 1) % region.num_vertices;
        double slope = (region.vertices[k].y - region.vertices[i].y) / (region.vertices[k].x - region.vertices[i].x);
        double c = region.vertices[k].y - (slope * (region.vertices[i].x));
        for (double x = region.vertices[i].x; x < region.vertices[k].x; x = x * 2) {
            double y = (slope * x) + c;
            if (is_inside_region((nccl_ofi_tuner_point_t){x, y}, &region) != 0)
                return -1;
            // printf(" Is (%.10f, %.10f) inside the region : %d\n", x, y, is_inside_region(
            //     (nccl_ofi_tuner_point_t){x, y}, &region));
        }
    }

    printf("All points on the edges of the polygon are detected correcltly\n");

    size_t num_points = 20;
    const nccl_ofi_tuner_point_t inside_vertices[] = {{16.0 * 1024 * 1024, 4},
                                                      {128.0 * 1024 * 1024, 4},
                                                      {1.0 * 1024 * 1024 * 1024, 4},
                                                      {4.0 * 1024 * 1024 * 1024, 4},
                                                      {16.0 * 1024 * 1024 * 1024, 4},
                                                      {32.0 * 1024 * 1024 * 1024, 4},
                                                      {64.0 * 1024 * 1024, 16},
                                                      {1.0 * 1024 * 1024 * 1024, 16},
                                                      {4.0 * 1024 * 1024 * 1024, 16},
                                                      {16.0 * 1024 * 1024 * 1024, 16},
                                                      {32.0 * 1024 * 1024 * 1024, 16},
                                                      {64.0 * 1024 * 1024 * 1024, 16},
                                                      {512.0 * 1024 * 1024, 128},
                                                      {4.0 * 1024 * 1024 * 1024, 128},
                                                      {16.0 * 1024 * 1024 * 1024, 128},
                                                      {32.0 * 1024 * 1024 * 1024, 128},
                                                      {64.0 * 1024 * 1024 * 1024, 128},
                                                      {64.0 * 1024 * 1024 * 1024, 256},
                                                      {TUNER_MAX_SIZE - 1.0, 128},
                                                      {e_48M_16_288M_128.x - 1.0, e_48M_16_288M_128.y - 1.0}};

    /* These points should be inside the polygon */
    for (size_t i = 0; i < num_points; i++) {
        if (is_inside_region(inside_vertices[i], &region) != 1) {
            printf("%.10f, %.10f\n", inside_vertices[i].x, inside_vertices[i].y);
            return -1;
        };
    }

    printf("All points inside the polygon are detected correcltly\n");

    const nccl_ofi_tuner_point_t outside_vertices[] = {{8.0 * 1024 * 1024, 4},
                                                       {8.0 * 1024 * 1024, 32},
                                                       {8.0 * 1024 * 1024, 128},
                                                       {8.0 * 1024 * 1024, 512},
                                                       {8.0 * 1024 * 1024, TUNER_MAX_RANKS},
                                                       {16.0 * 1024 * 1024, 8},
                                                       {16.0 * 1024 * 1024, 32},
                                                       {16.0 * 1024 * 1024, 128},
                                                       {16.0 * 1024 * 1024, 512},
                                                       {16.0 * 1024 * 1024, TUNER_MAX_RANKS},
                                                       {32 * 1024 * 1024, 16},
                                                       {32.0 * 1024 * 1024, 32},
                                                       {32.0 * 1024 * 1024, 64},
                                                       {32.0 * 1024 * 1024, 128},
                                                       {32.0 * 1024 * 1024, 256},
                                                       {64 * 1024 * 1024, 32},
                                                       {64.0 * 1024 * 1024, 64},
                                                       {64.0 * 1024 * 1024, 256},
                                                       {64.0 * 1024 * 1024, 1024},
                                                       {256.0 * 1024 * 1024, 256},
                                                       {256.0 * 1024 * 1024, 1024},
                                                       {256.0 * 1024 * 1024, 2048},
                                                       {TUNER_MAX_SIZE + 1.0, 128},
                                                       {e_48M_16_288M_128.x + 1.0, e_48M_16_288M_128.y + 1.0}};

    /* These points should be outside the polygons */
    for (size_t i = 0; i < num_points; i++) {
        if (is_inside_region(outside_vertices[i], &region) != -1) {
            printf("%.10f, %.10f\n", outside_vertices[i].x, outside_vertices[i].y);
            return -1;
        };
    }
    if (is_inside_region((nccl_ofi_tuner_point_t){2.0 * 1024 * 1024, 2}, &region) != -1)
        return -1;
    if (is_inside_region((nccl_ofi_tuner_point_t){64.0 * 1024 * 1024, 64}, &region) != -1)
        return -1;
    if (is_inside_region((nccl_ofi_tuner_point_t){200.0 * 1024 * 1024, 128}, &region) != -1)
        return -1;
    if (is_inside_region((nccl_ofi_tuner_point_t){1024.0 * 1024 * 1024, 1024}, &region) != -1)
        return -1;
    if (is_inside_region((nccl_ofi_tuner_point_t){1024.0 * 1024 * 1024, 2048}, &region) != -1)
        return -1;

    printf("All the points outside the polygon are detected correctly\n");

    return 0;
}

int main(int argc, const char **argv) {
    if (test_extend_region() < 0) {
        printf("Extend Region test failed\n");
    };

    if (test_is_inside_region() < 0) {
        printf("Is inside region function failed\n");
    }

    return 0;
}
