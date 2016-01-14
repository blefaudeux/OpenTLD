#ifndef HALIDE_EXPERIMENTS_H
#define HALIDE_EXPERIMENTS_H

#include <opencv/cv.h>

namespace hexp
{
    void halide_blur(cv::Mat output, cv::Mat input);

    void halide_fast_blur(cv::Mat output, cv::Mat input);

}

#endif // HALIDE_EXPERIMENTS_H
