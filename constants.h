#pragma once

#define FEATURE_SIZE 24
#define IM_WIDTH 384
#define IM_HEIGHT 288
#define FP_FACES_DIR "../dataset/faces/"
#define FP_BGS_DIR "../dataset/backgrounds/"

typedef unsigned char uchar;

enum Result {
    SUCCESS,
    FAILURE
};