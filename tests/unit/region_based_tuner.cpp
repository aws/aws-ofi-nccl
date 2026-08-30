/*
 * Copyright (c) 2024-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "unit_test.h"
#include "internal/tuner/nccl_defaults.h"
#include "tuner/nccl_ofi_tuner_region.h"
#include "nccl_ofi_param.h"

using std::abs;
using std::log2;
using std::pow;
const double eps = 1e-4;

static int test_extend_region(void)
{
    nccl_ofi_tuner_point_t extended_point;
    double slope;
    double projected_x, projected_y;

    /* Extend the line on the x-axis */
    extended_point = extend_region((nccl_ofi_tuner_point_t){2, 8},
                                   (nccl_ofi_tuner_point_t){4, 8},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    if (abs(extended_point.x - TUNER_MAX_SIZE) > eps || extended_point.y != 8) {
        printf("X-Axis Extend Test Failed : Extended Points : x = %f (diff = %f) y = %f\n", extended_point.x,
            extended_point.x - TUNER_MAX_SIZE, extended_point.y);
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
    slope = (log2(72.0) - log2(64.0)) / (log2(8290304.0) - log2(8.0)); // slope = (y2 - y1)/(x2 - x1)
    // y3 = mx3 + c and substitute for m=(y2-y1)/(x2-x1) and c = y2 - mx2
    projected_y = pow(2.0, log2(72.0) + slope * (log2(TUNER_MAX_SIZE) - log2(8290304.0))); // y3 = y2 + mx3 - mx2
    if (abs(extended_point.x - TUNER_MAX_SIZE) > eps || extended_point.y != projected_y) {
        printf("X-Axis Upper Bound Test Failed : Extended Points : x = %f (diff = %f) y = %f (diff = %f) \n",
            extended_point.x, extended_point.x - TUNER_MAX_SIZE,
            extended_point.y, extended_point.y - projected_y);
        return -1;
    }

    /* Extend the line to TUNER_MAX_RANKS (y-axis) */
    extended_point = extend_region((nccl_ofi_tuner_point_t){8, 64},
                                   (nccl_ofi_tuner_point_t){16, 1024},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    slope = (log2(1024.0) - log2(64.0)) / (log2(16) - log2(8.0));
    projected_x = pow(2, ((log2(TUNER_MAX_RANKS) - log2(1024.0)) / slope) + log2(16));
    if (abs(extended_point.x - projected_x) > eps || extended_point.y != TUNER_MAX_RANKS) {
        printf("X-Axis Upper Bound Test 2 Failed : Extended Points : x = %f (diff = %f) y = %f (diff = %f) \n",
            extended_point.x, extended_point.x - projected_x,
            extended_point.y, extended_point.y - TUNER_MAX_RANKS);
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
|    p3(4M, 2)                                           |p5(TUNER_MAX_SIZE, 2))
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
    printf("INFO extended point: %f %f \n", e_48M_16_288M_128.x, e_48M_16_288M_128.y );

    p1_288M_128.transform_log2();
    p2_38M_16.transform_log2();
    p3_4M_2.transform_log2();
    p5_maxM_2.transform_log2();
    e_48M_16_288M_128.transform_log2();
    printf("INFO extended point after transform_log2: %f %f \n", e_48M_16_288M_128.x, e_48M_16_288M_128.y );

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
    3. Using the equation y = m * x + c, get multiple points on the line in powers of 2.
    */
    for (size_t i = 0; i < region.num_vertices; i++) {
        size_t k = (i + 1) % region.num_vertices;
        double slope = (region.vertices[k].y - region.vertices[i].y) / (region.vertices[k].x - region.vertices[i].x);
        double c = region.vertices[k].y - (slope * (region.vertices[i].x));
        for (double x = region.vertices[i].x; x < region.vertices[k].x; x = x * 2) {
            double y = (slope * x) + c;
            nccl_ofi_tuner_point_t test_point {x, y, nccl_ofi_tuner_point_t::LOG2};

            if (is_inside_region(test_point, &region) != 0)
                return -1;
            // printf(" Is (%.10f, %.10f) inside the region : %d\n", x, y, is_inside_region(
            //     (nccl_ofi_tuner_point_t){x, y}, &region));
        }
    }

    printf("All points on the edges of the polygon are detected correcltly\n");

    nccl_ofi_tuner_point_t e_48M_16_288M_128_OFF_ORIGINAL = extend_region(
        (nccl_ofi_tuner_point_t){(double)48.0 * 1024 * 1024, 16},
        (nccl_ofi_tuner_point_t){(double)288.0 * 1024 * 1024, 128},
        (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    e_48M_16_288M_128_OFF_ORIGINAL.x -= 10.0;
    e_48M_16_288M_128_OFF_ORIGINAL.y -= 10.0;

    const size_t num_points = 20;
    nccl_ofi_tuner_point_t inside_vertices[] = {{16.0 * 1024 * 1024, 4},
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
                                                      // Note, set a big enough diff (10.0) below, otherwise
                                                      // the delta after log2 is within floating error (eps).
                                                      {TUNER_MAX_SIZE - 10.0, 128},
                                                      e_48M_16_288M_128_OFF_ORIGINAL};

    /* These points should be inside the polygon */
    for (size_t i = 0; i < num_points; i++) {
        inside_vertices[i].transform_log2();
        int d = is_inside_region(inside_vertices[i], &region);
        if (d != 1) {
            printf("%ld: %.10f, %.10f is_inside_region: %d\n", i, inside_vertices[i].x, inside_vertices[i].y, d);
            return -1;
        };
    }

    printf("All points inside the polygon are detected correcltly\n");

    const size_t outside_num_points = 24;
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
    for (size_t i = 0; i < outside_num_points; i++) {
        int d = is_inside_region(outside_vertices[i], &region);
        if ( d != -1) {
            printf("%ld: %.10f, %.10f is_inside_region: %d\n", i, outside_vertices[i].x, outside_vertices[i].y, d);
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

/*
 * P6-B200 Chunk-Size Tuning Unit Tests
 */

/*
 * Decision-path test for the P6 "confine the AllReduce/AllGather 0x7 polygon
 * extension to 8-16 node clusters" commit.
 *
 * Drives the real tuner decision path: region_init_internal() builds the P6
 * regions for a given (nRanks == nNodes) cluster, then
 * region_get_coll_info_internal_v3() is asked to choose an algo/proto for a
 * (collective, message_size). The chosen entry is the one the tuner sets to
 * cost 0.0 in the cost table (all other entries are left at a sentinel). This
 * mirrors how NCCL core consumes the tuner.
 */

/* Return the (algo, proto) the P6 region tuner selects for a 1-rank-per-node
 * cluster of nNodes nodes, collective collType and message size nBytes.
 * Writes -1/-1 when the tuner makes no selection (NCCL fallback). */
static void p6_region_decision(size_t nNodes, ncclFunc_t collType, size_t nBytes,
                               int *out_algo, int *out_proto)
{
    nccl_ofi_tuner_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));

    /* 0x7 split mask == 1 rank per node, so nRanks == nNodes. */
    if (region_init_internal(&ctx, NCCL_OFI_TUNER_P6, nNodes, nNodes) != ncclSuccess) {
        *out_algo = -2;
        *out_proto = -2;
        return;
    }

    /* Seed the cost table with a sentinel so no region is skipped (entries equal
     * to NCCL_ALGO_PROTO_IGNORE are ignored by the tuner) and the chosen cell is
     * detectable as the one set to 0.0. */
    const float SENTINEL = 1337.0f;
    float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
            table[a][p] = SENTINEL;

    int nChannels = 0;
    region_get_coll_info_internal_v3(&ctx, collType, nBytes, 1, (float **)table,
                                     NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS, &nChannels);

    *out_algo = -1;
    *out_proto = -1;
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
            if (table[a][p] == 0.0f) {
                *out_algo = a;
                *out_proto = p;
            }
        }
    }

    region_destroy_internal(&ctx);
}

static int check_decision(size_t nNodes, ncclFunc_t collType, size_t nBytes,
                          int exp_algo, int exp_proto, const char *label)
{
    int algo = -1, proto = -1;
    p6_region_decision(nNodes, collType, nBytes, &algo, &proto);
    if (algo != exp_algo || proto != exp_proto) {
        printf("FAIL [%s]: nNodes=%zu nBytes=%zu -> algo/proto %d/%d, expected %d/%d\n",
               label, nNodes, nBytes, algo, proto, exp_algo, exp_proto);
        return -1;
    }
    return 0;
}

static int test_p6_polygon_8_16_node_extension(void)
{
    const size_t M = 1024 * 1024;
    int ret = 0;

    /* --- Change is PRESENT for 8-16 node clusters --- */

    /* AllReduce: extended TREE/LL128 region.
     * node 8  -> TREE/LL128 at 1M, 2M (master: RING/LL128). */
    ret |= check_decision(8, ncclFuncAllReduce, 1 * M, NCCL_ALGO_TREE, NCCL_PROTO_LL128, "AR n8 1M");
    ret |= check_decision(8, ncclFuncAllReduce, 2 * M, NCCL_ALGO_TREE, NCCL_PROTO_LL128, "AR n8 2M");
    /* node 16 -> TREE/LL128 at 2M, 4M, 8M. */
    ret |= check_decision(16, ncclFuncAllReduce, 2 * M, NCCL_ALGO_TREE, NCCL_PROTO_LL128, "AR n16 2M");
    ret |= check_decision(16, ncclFuncAllReduce, 4 * M, NCCL_ALGO_TREE, NCCL_PROTO_LL128, "AR n16 4M");
    ret |= check_decision(16, ncclFuncAllReduce, 8 * M, NCCL_ALGO_TREE, NCCL_PROTO_LL128, "AR n16 8M");
    /* Above the extended TREE/LL128 band the selection is unchanged (master:
     * RING/LL128). One large size per node count is enough to confirm the
     * extension did not perturb large-message decisions. */
    ret |= check_decision(8, ncclFuncAllReduce, 4 * M, NCCL_ALGO_RING, NCCL_PROTO_LL128, "AR n8 4M (unchanged)");
    ret |= check_decision(16, ncclFuncAllReduce, 16 * M, NCCL_ALGO_RING, NCCL_PROTO_LL128, "AR n16 16M (unchanged)");

    /* AllGather: extended PAT/SIMPLE region.
     * node 8  -> PAT/SIMPLE at 4M, 8M, 16M (master: RING/LL128). */
    ret |= check_decision(8, ncclFuncAllGather, 4 * M, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, "AG n8 4M");
    ret |= check_decision(8, ncclFuncAllGather, 8 * M, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, "AG n8 8M");
    ret |= check_decision(8, ncclFuncAllGather, 16 * M, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, "AG n8 16M");
    /* node 16 -> PAT/SIMPLE at 8M, 16M, 32M. */
    ret |= check_decision(16, ncclFuncAllGather, 8 * M, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, "AG n16 8M");
    ret |= check_decision(16, ncclFuncAllGather, 16 * M, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, "AG n16 16M");
    ret |= check_decision(16, ncclFuncAllGather, 32 * M, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, "AG n16 32M");
    /* Above the extended PAT/SIMPLE band the selection is unchanged (master:
     * RING/SIMPLE). One large size per node count is enough to confirm the
     * extension did not perturb large-message decisions. */
    ret |= check_decision(8, ncclFuncAllGather, 32 * M, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, "AG n8 32M (unchanged)");
    ret |= check_decision(16, ncclFuncAllGather, 64 * M, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, "AG n16 64M (unchanged)");

    /* --- Change is CONFINED: clusters outside 8-16 nodes are NOT extended --- */

    /* node 4 (< 8): the above cells must remain RING/LL128, i.e. the extension
     * did not reach small clusters. */
    ret |= check_decision(4, ncclFuncAllReduce, 1 * M, NCCL_ALGO_RING, NCCL_PROTO_LL128, "AR n4 1M (unchanged)");
    ret |= check_decision(4, ncclFuncAllReduce, 2 * M, NCCL_ALGO_RING, NCCL_PROTO_LL128, "AR n4 2M (unchanged)");
    ret |= check_decision(4, ncclFuncAllGather, 4 * M, NCCL_ALGO_RING, NCCL_PROTO_LL128, "AG n4 4M (unchanged)");
    ret |= check_decision(4, ncclFuncAllGather, 8 * M, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, "AG n4 8M (unchanged)");

    /* node 32 (> 16): the extended cells must remain RING/LL128 as well. */
    ret |= check_decision(32, ncclFuncAllReduce, 4 * M, NCCL_ALGO_RING, NCCL_PROTO_LL128, "AR n32 4M (unchanged)");
    ret |= check_decision(32, ncclFuncAllReduce, 8 * M, NCCL_ALGO_RING, NCCL_PROTO_LL128, "AR n32 8M (unchanged)");
    ret |= check_decision(32, ncclFuncAllGather, 32 * M, NCCL_ALGO_RING, NCCL_PROTO_LL128, "AG n32 32M (unchanged)");
    ret |= check_decision(32, ncclFuncAllGather, 16 * M, NCCL_ALGO_RING, NCCL_PROTO_LL128, "AG n32 16M (unchanged)");

    /* --- Unusual node counts (10, 12) inside the range interpolate as extended --- */
    ret |= check_decision(10, ncclFuncAllReduce, 4 * M, NCCL_ALGO_TREE, NCCL_PROTO_LL128, "AR n10 4M");
    ret |= check_decision(12, ncclFuncAllReduce, 4 * M, NCCL_ALGO_TREE, NCCL_PROTO_LL128, "AR n12 4M");
    ret |= check_decision(10, ncclFuncAllGather, 16 * M, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, "AG n10 16M");
    ret |= check_decision(12, ncclFuncAllGather, 16 * M, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, "AG n12 16M");

    /* --- Non-power-of-2 message size inside an extended cell --- */
    /* 1.5M at node 16 lies in the AllReduce TREE/LL128 extended band. */
    ret |= check_decision(16, ncclFuncAllReduce, 3 * M / 2, NCCL_ALGO_TREE, NCCL_PROTO_LL128, "AR n16 1.5M (non-pow2)");
    /* 6M at node 8 lies in the AllGather PAT/SIMPLE extended band (4M..16M). */
    ret |= check_decision(8, ncclFuncAllGather, 6 * M, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, "AG n8 6M (non-pow2)");

    if (ret == 0) {
        printf("P6 8-16 node polygon extension: all decision checks passed\n");
    }
    return ret;
}

/*
 * P6-B200 chunk size tuning vs. benchmark data
 * --------------------------------------------
 * For every benchmarked message size on 8 and 16 node clusters (1 rank per
 * node, i.e. the 0x7 split mask) these tests walk the same path NCCL does:
 *
 *   1. region_get_coll_info_internal_v3() picks algo/proto (and may pick nChannels).
 *   2. If (and only if) the pick is the one chunk size tuning applies to
 *      (AllReduce TREE/LL128, AllGather PAT/SIMPLE), the chunk size is asked
 *      from region_get_chunk_size_internal(). NCCL hands the chunk size tuner
 *      the *per-channel* byte count (message size / number of channels), so
 *      nBytes is divided by the channel count before the call.
 *   3. The returned chunk size is compared with the best chunk size as found in
 *      a benchmark sweep. On a mismatch, the master data decides as follows:
 *        - selected != master chunk size          -> FAIL
 *        - selected == master chunk size, and the "Gain Vs. Master" lost by not
 *          using the best chunk > 5% (gain_exceeds_5pct)   -> FAIL
 *        - otherwise                              -> WARNING (test passes)
 */

/* Number of channels NCCL uses on P6-B200 when the tuner leaves nChannels
 * untouched (*nChannels == 0 after region_get_coll_info_internal_v3()). This is
 * always the case for AllReduce 0x7 and, for AllGather PAT/SIMPLE, above
 * num_nodes * 128KB. It is not a tuner decision, but rather is derived from actual
 * benchmark runs. */
static const int P6_NCCL_DEFAULT_NCHANNELS = 4;

struct chunk_bench_point {
    size_t msg_size;            /* total message size, bytes */
    size_t best_chunk;          /* best chunk size */
    size_t master_chunk;        /* current master (before change) chunk size */
    bool gain_exceeds_5pct;     /* Gain Vs. Master > 5%; false when N/A */
};

/* AllReduce 0x7, 8 nodes benchmark reference data. */
static const chunk_bench_point ar_8_nodes_best_chunk[] = {
    /* msg_size        best    master  gain vs master > 5% */
    {8,               17280,   32768,   false},
    {16,              17280,   32768,   false},
    {32,              17280,   32768,   false},
    {64,              17280,   32768,   false},
    {128,             17280,   32768,   false},
    {256,             17280,   32768,   false},
    {512,             17280,   32768,   false},
    {1024,            17280,   32768,   false},
    {2048,            17280,   32768,   false},
    {4096,            17280,   32768,   false},
    {8192,            17280,   32768,   false},
    {16384,           17280,   32768,   false},
    {32768,           34560,   32768,   false},
    {65536,           17280,   17280,   false},
    {131072,          17280,   17280,   false},
    {262144,          17280,   17280,   false},
    {524288,          34560,   17280,   false},
    {1048576,         34560,   576000,  true},
    {2097152,         71040,   576000,  true},
    {4194304,         71040,   576000,  false},
    {8388608,         144000,  576000,  false},
    {16777216,        144000,  4194304, false},
    {33554432,        288000,  4194304, false},
    {67108864,        288000,  4194304, false},
    {134217728,       288000,  4194304, false},
    {268435456,       288000,  4194304, false},
    {536870912,       288000,  4194304, false},
    {1073741824,      288000,  4194304, false},
    {2147483648ULL,   288000,  4194304, false},
    {4294967296ULL,   288000,  4194304, false},
    {8589934592ULL,   288000,  4194304, false},
};

/* AllReduce 0x7, 16 nodes benchmark reference data. */
static const chunk_bench_point ar_16_nodes_best_chunk[] = {
    /* msg_size        best    master  gain vs master > 5% */
    {8,               17280,   32768,   false},
    {16,              17280,   32768,   false},
    {32,              17280,   32768,   false},
    {64,              17280,   32768,   false},
    {128,             17280,   32768,   false},
    {256,             17280,   32768,   false},
    {512,             17280,   32768,   false},
    {1024,            17280,   32768,   false},
    {2048,            17280,   32768,   false},
    {4096,            17280,   32768,   false},
    {8192,            17280,   32768,   false},
    {16384,           17280,   32768,   false},
    {32768,           17280,   32768,   false},
    {65536,           17280,   17280,   false},
    {131072,          17280,   17280,   false},
    {262144,          17280,   17280,   false},
    {524288,          34560,   17280,   false},
    {1048576,         34560,   17280,   true},
    /* NOTE: although benchmark data shows for 2MB best chunk size is 34560,
     * the actual difference from 71040 is very marginal, and the formula is
     * more naturally smooth with this selection. */
    {2097152,         71040,   576000,  true},
    {4194304,         71040,   576000,  true},
    {8388608,         144000,  576000,  true},
    {16777216,        144000,  576000,  false},
    {33554432,        144000,  4194304, false},
    {67108864,        288000,  4194304, false},
    {134217728,       288000,  4194304, false},
    {268435456,       288000,  4194304, false},
    {536870912,       288000,  4194304, false},
    {1073741824,      288000,  4194304, false},
    {2147483648ULL,   288000,  4194304, false},
    {4294967296ULL,   288000,  4194304, false},
    {8589934592ULL,   288000,  4194304, false},
};

/* AllGather 0x7, 8 nodes benchmark reference data. */
static const chunk_bench_point ag_8_nodes_best_chunk[] = {
    /* msg_size        best    master  gain vs master > 5% */
    {128,             32768,   65536,   false},
    {256,             32768,   65536,   false},
    {512,             32768,   65536,   false},
    {1024,            32768,   65536,   false},
    {2048,            32768,   65536,   false},
    {4096,            32768,   65536,   false},
    {8192,            32768,   65536,   false},
    {16384,           32768,   65536,   false},
    {32768,           32768,   65536,   false},
    {65536,           32768,   65536,   false},
    {131072,          32768,   65536,   false},
    {262144,          32768,   65536,   true},
    {524288,          65536,   65536,   false},
    {1048576,         65536,   65536,   false},
    {2097152,         65536,   65536,   false},
    {4194304,         131072,  576000,  true},
    {8388608,         262144,  576000,  true},
    {16777216,        524288,  576000,  true},
    {33554432,        1048576, 4194304, false},
    {67108864,        1048576, 4194304, false},
    {134217728,       524288,  4194304, false},
    {268435456,       1048576, 4194304, false},
    {536870912,       1048576, 4194304, false},
    {1073741824,      1048576, 4194304, false},
    {2147483648ULL,   1048576, 4194304, false},
    {4294967296ULL,   1048576, 4194304, false},
    {8589934592ULL,   1048576, 4194304, false},
};

/* AllGather 0x7, 16 nodes benchmark reference data. */
static const chunk_bench_point ag_16_nodes_best_chunk[] = {
    /* msg_size        best    master  gain vs master > 5% */
    {256,             32768,   65536,   false},
    {512,             32768,   65536,   false},
    {1024,            32768,   65536,   false},
    {2048,            32768,   65536,   false},
    {4096,            32768,   65536,   false},
    {8192,            32768,   65536,   false},
    {16384,           32768,   65536,   false},
    {32768,           32768,   65536,   false},
    {65536,           32768,   65536,   false},
    {131072,          32768,   65536,   true},
    {262144,          32768,   65536,   false},
    {524288,          32768,   65536,   true},
    {1048576,         65536,   65536,   false},
    {2097152,         65536,   65536,   false},
    {4194304,         65536,   65536,   false},
    {8388608,         131072,  576000,  true},
    {16777216,        262144,  576000,  true},
    {33554432,        524288,  576000,  true},
    {67108864,        524288,  4194304, false},
    {134217728,       524288,  4194304, false},
    {268435456,       524288,  4194304, false},
    {536870912,       524288,  4194304, false},
    {1073741824,      524288,  4194304, false},
    {2147483648ULL,   524288,  4194304, false},
    {4294967296ULL,   524288,  4194304, false},
    {8589934592ULL,   524288,  4194304, false},
};

/* NCCL aligns an LL128 chunk to the protocol grain (the same grain
 * calculateChunkSizeTreeLL128() uses in the tuner), e.g. the tuner's 18000 /
 * 36000 / 72000 become the 17280 / 34560 / 71040 seen in the benchmark. */
static size_t ll128_chunk_grain(void)
{
    return NCCL_OFI_TUNER_NCCL_WARP_SIZE * NCCL_OFI_TUNER_NCCL_LL128_SHMEM_ELEMS_PER_THREAD /
           NCCL_OFI_TUNER_NCCL_LL128_LINEELEMS * NCCL_OFI_TUNER_NCCL_LL128_DATAELEMS * sizeof(uint64_t);
}

/* Find the (algo, proto) cell the tuner set to 0.0 in the cost table; -1/-1 if none. */
static void find_selected_algo_proto(float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS],
                                     int *out_algo, int *out_proto)
{
    *out_algo = -1;
    *out_proto = -1;
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
            if (table[a][p] == 0.0f) {
                *out_algo = a;
                *out_proto = p;
            }
        }
    }
}

/*
 * Run the decision + chunk size path for every benchmark point of one
 * (collective, cluster size) and compare with the benchmark "best" chunk size.
 * Only points where the tuner selects (exp_algo, exp_proto) - the algo/proto
 * chunk size tuning applies to - are checked; the others are skipped.
 * Every checked message size is printed with MATCH / WARNING / FAIL; on a
 * mismatch with the benchmark best, the master chunk size and the
 * "Gain Vs. Master" decide between WARNING (pass) and FAIL (see above).
 */
static int check_p6_chunk_size_vs_benchmark(const char *label, ncclFunc_t collType, size_t nNodes,
                                            int exp_algo, int exp_proto,
                                            const chunk_bench_point *points, size_t num_points)
{
    nccl_ofi_tuner_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));

    /* 0x7 split mask == 1 rank per node, so nRanks == nNodes. */
    if (region_init_internal(&ctx, NCCL_OFI_TUNER_P6, nNodes, nNodes) != ncclSuccess) {
        printf("FAIL [%s]: region_init_internal failed for %zu nodes\n", label, nNodes);
        return -1;
    }

    int ret = 0;
    size_t matched = 0, warned = 0, failed = 0, skipped = 0;
    const size_t grain = ll128_chunk_grain();

    printf("[%s] %12s %5s %13s %10s %10s %10s %9s  %s\n", label, "msg_size", "nch", "per_ch_bytes",
           "selected", "best", "master", "gain>5%", "result");

    for (size_t i = 0; i < num_points; i++) {
        const size_t msg_size = points[i].msg_size;

        /* Sentinel so no region is skipped and the chosen cell is the one set to 0.0 */
        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
        for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
            for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
                table[a][p] = 1337.0f;

        /* 0 == "tuner did not override the channel count" */
        int nChannels = 0;
        if (region_get_coll_info_internal_v3(&ctx, collType, msg_size, 1, (float **)table,
                                             NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                                             &nChannels) != ncclSuccess) {
            printf("FAIL [%s]: region_get_coll_info_internal_v3 failed, nBytes=%zu\n", label, msg_size);
            ret = -1;
            continue;
        }

        int algo = -1, proto = -1;
        find_selected_algo_proto(table, &algo, &proto);
        if (algo != exp_algo || proto != exp_proto) {
            skipped++;
            continue;
        }

        /* Channel count NCCL ends up using, and the per-channel byte count it
         * passes to the chunk size tuner. */
        const int eff_nchannels = nChannels > 0 ? nChannels : P6_NCCL_DEFAULT_NCHANNELS;
        const size_t per_channel_bytes = msg_size / eff_nchannels;

        size_t chunk = 0;
        if (region_get_chunk_size_internal(&ctx, collType, per_channel_bytes, algo, proto,
                                           eff_nchannels, &chunk) != ncclSuccess || chunk == 0) {
            printf("FAIL [%s]: nBytes=%zu (per-channel %zu, %d channels): no chunk size selected\n",
                   label, msg_size, per_channel_bytes, eff_nchannels);
            ret = -1;
            failed++;
            continue;
        }
        if (collType == ncclFuncAllReduce)
            chunk = (chunk / grain) * grain;

        const chunk_bench_point *pt = &points[i];
        const char *result;
        if (chunk == pt->best_chunk) {
            result = "MATCH";
            matched++;
        } else if (chunk != pt->master_chunk) {
            /* Neither the benchmark best nor what master uses. */
            result = "FAIL (differs from best and master)";
            failed++;
            ret = -1;
        } else if (pt->gain_exceeds_5pct) {
            /* Same as master, but the gain lost vs. the best chunk exceeds 5%. */
            result = "FAIL (same as master, lost gain > 5%)";
            failed++;
            ret = -1;
        } else {
            /* Same as master and the lost gain is at most 5% (or N/A). */
            result = "WARNING (same as master, lost gain <= 5%)";
            warned++;
        }

        printf("[%s] %12zu %5d %13zu %10zu %10zu %10zu %9s  %s\n", label, pt->msg_size, eff_nchannels,
               per_channel_bytes, chunk, pt->best_chunk, pt->master_chunk,
               pt->gain_exceeds_5pct ? "yes" : "no", result);
    }

    region_destroy_internal(&ctx);

    printf("[%s] %zu match, %zu warning, %zu fail, %zu skipped (tuner did not select the tuned algo/proto)\n",
           label, matched, warned, failed, skipped);

    /* Guard against a test that silently verifies nothing. */
    if (matched + warned + failed == 0) {
        printf("FAIL [%s]: no benchmark point was checked\n", label);
        return -1;
    }
    return ret;
}

static int test_p6_chunk_size_all_reduce_0x7(void)
{
    int ret = 0;
    ret |= check_p6_chunk_size_vs_benchmark("AR 0x7 8 nodes", ncclFuncAllReduce, 8,
                                            NCCL_ALGO_TREE, NCCL_PROTO_LL128, ar_8_nodes_best_chunk,
                                            sizeof(ar_8_nodes_best_chunk) / sizeof(ar_8_nodes_best_chunk[0]));
    ret |= check_p6_chunk_size_vs_benchmark("AR 0x7 16 nodes", ncclFuncAllReduce, 16,
                                            NCCL_ALGO_TREE, NCCL_PROTO_LL128, ar_16_nodes_best_chunk,
                                            sizeof(ar_16_nodes_best_chunk) / sizeof(ar_16_nodes_best_chunk[0]));
    return ret;
}

static int test_p6_chunk_size_all_gather_0x7(void)
{
    int ret = 0;
    ret |= check_p6_chunk_size_vs_benchmark("AG 0x7 8 nodes", ncclFuncAllGather, 8,
                                            NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, ag_8_nodes_best_chunk,
                                            sizeof(ag_8_nodes_best_chunk) / sizeof(ag_8_nodes_best_chunk[0]));
    ret |= check_p6_chunk_size_vs_benchmark("AG 0x7 16 nodes", ncclFuncAllGather, 16,
                                            NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, ag_16_nodes_best_chunk,
                                            sizeof(ag_16_nodes_best_chunk) / sizeof(ag_16_nodes_best_chunk[0]));
    return ret;
}

int main(int argc, const char **argv) {
    int ret = 0;

    unit_test_init();

    if ((ret |= test_extend_region()) < 0) {
        printf("Extend Region test failed\n");
    };

    if ((ret |= test_is_inside_region()) < 0) {
        printf("Is inside region function failed\n");
    }

    if ((ret |= test_p6_polygon_8_16_node_extension()) < 0) {
        printf("P6 8-16 node polygon extension test failed\n");
    }

    int r;
    if ((r = test_p6_chunk_size_all_reduce_0x7()) < 0) {
        printf("P6 AllReduce 0x7 chunk size test failed\n");
    }
    ret |= r;

    if ((r = test_p6_chunk_size_all_gather_0x7()) < 0) {
        printf("P6 AllGather 0x7 chunk size test failed\n");
    }
    ret |= r;

    if (ret == 0) {
        printf("All tests passed.\n");
    }
    return ret;
}
