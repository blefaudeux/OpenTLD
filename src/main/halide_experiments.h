#ifndef HALIDE_EXPERIMENTS_H
#define HALIDE_EXPERIMENTS_H

#include <opencv/cv.h>

namespace hexp
{
    void halide_blur_minimal(cv::Mat output, cv::Mat input);
}

#endif // HALIDE_EXPERIMENTS_H
