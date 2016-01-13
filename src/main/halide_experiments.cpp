
#include <Halide/include/Halide.h>
#include <numeric>
#include <halide_experiments.h>
#include <Halide/tools/halide_image_io.h>

namespace hexp
{
    using namespace Halide;

    void halide_blur(cv::Mat output, cv::Mat input)
    {
        // Load the picture from the OpenCV raw buffer
        Image<uint8_t> in( Buffer(UInt(8), input.cols, input.rows, 0, 0, input.data ) );
        Image<uint8_t> out( Buffer(UInt(8), input.cols, input.rows, 0, 0, output.data ) );

        // Define a 3x3 Gaussian Blur with a repeat-edge boundary condition.
        // cf. https://github.com/halide/CVPR2015/blob/master/blur.cpp
        float sigma = 1.5f;

        Var x, y, c;
        Func kernel;
        kernel(x) = exp(-x*x/(2*sigma*sigma)) / (sqrtf(2*M_PI)*sigma);

        Func in_bounded = BoundaryConditions::repeat_edge(in);

        Func blur_y;
        blur_y(x, y, c) = kernel(0) * in_bounded(x, y, c) + kernel(1) * (in_bounded(x, y-1, c) +
                                                                         in_bounded(x, y+1, c));

        Func blur_x;
        blur_x(x, y, c) = (kernel(0) * blur_y(x, y, c) + kernel(1) * (blur_y(x-1, y, c) +
                                                                      blur_y(x+1, y, c)));

        // Schedule it.
        blur_x.compute_root().vectorize(x, 8).parallel(y);
        blur_y.compute_at(blur_x, y).vectorize(x, 8);

        // Run
        blur_x.realize(output);
    }
}
