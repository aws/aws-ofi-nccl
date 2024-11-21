/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "tuner/nccl_ofi_tuner_region.h"
#include "nccl_ofi_param.h"

static int test_geometry_extend_region(void) {
    nccl_ofi_tuner_point_t extended_point;
    double slope;
    double projected_x, projected_y;

    /* Extend the line on the x-axis */
    extended_point = extend_region((nccl_ofi_tuner_point_t){2, 8},
                                   (nccl_ofi_tuner_point_t){4, 8},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    if (extended_point.x != TUNER_MAX_SIZE || extended_point.y != 8){
        printf("X-Axis Extend Test Failed : Extended Points : x = %f y = %f\n", extended_point.x, extended_point.y);
        return -1;
    }

    /* Extend the line on the y-axis*/
    extended_point = extend_region((nccl_ofi_tuner_point_t){8, 2},
                                   (nccl_ofi_tuner_point_t){8, 4},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    if(extended_point.x != 8 || extended_point.y != TUNER_MAX_RANKS){
        printf("Y-Axis Extend Test Failed : Extended Points : x = %f y = %f\n", extended_point.x, extended_point.y);
        return -1;
    }

    /* Extend the line to TUNER_MAX_SIZE (x-axis) */
    extended_point = extend_region((nccl_ofi_tuner_point_t){8, 64},
                                   (nccl_ofi_tuner_point_t){8290304, 72},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    slope = (72.0 - 64.0)/(8290304.0 - 8.0); // slope = (y2 - y1)/(x2 - x1)
    // y3 = mx3 + c and substitute for m=(y2-y1)/(x2-x1) and c = y2 - mx2 
    projected_y = 72.0 + slope*(TUNER_MAX_SIZE - 8290304.0); // y3 = y2 + mx3 - mx2
    if(extended_point.x != TUNER_MAX_SIZE || extended_point.y != projected_y){
        printf("X-Axis Upper Bound Test Failed : Extended Points : x = %f y = %f\n", extended_point.x, extended_point.y);
        return -1;
    }

    /* Extend the line to TUNER_MAX_RANKS (y-axis) */
    extended_point = extend_region((nccl_ofi_tuner_point_t){8, 64},
                                   (nccl_ofi_tuner_point_t){16, 1024},
                                   (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
    slope = (1024.0 - 64.0)/(16.0 - 8.0);
    projected_x = ((TUNER_MAX_RANKS - 1024.0)/slope) + 16;
    if(extended_point.x != projected_x || extended_point.y != TUNER_MAX_RANKS){
        printf("X-Axis Upper Bound Test Failed : Extended Points : x = %f y = %f\n", extended_point.x, extended_point.y);
        return -1;
    }

    return 0;
}

int main(int argc, const char **argv)
{
    if(test_geometry_extend_region() == -1){
        printf("Extend Region test failed");
    };

    return 0;
}
