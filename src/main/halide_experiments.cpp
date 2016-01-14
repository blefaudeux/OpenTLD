
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
        Image<uint8_t> in( Buffer(UInt(8), input.cols, input.rows, 1, 0, input.data ) );
        Image<uint8_t> out( Buffer(UInt(8), input.cols, input.rows, 1, 0, output.data ) );

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
        blur_x(x, y, c) = cast<uint8_t>((kernel(0) * blur_y(x, y, c) + kernel(1) * (blur_y(x-1, y, c) +
                                                                      blur_y(x+1, y, c))));

        // Schedule it.
        blur_x.compute_root().vectorize(x, 8).parallel(y);
        blur_y.compute_at(blur_x, y).vectorize(x, 8);

        // Run
        blur_x.realize(out);
    }

    void halide_fast_blur(cv::Mat output, cv::Mat input)
    {
        // Testing options from https://github.com/victormatheus/halide-casestudies

        // Load the picture from the OpenCV raw buffer
        Image<uint8_t> in( Buffer(UInt(8), input.cols, input.rows, 1, 0, input.data ) );
        Image<uint8_t> out( Buffer(UInt(8), input.cols, input.rows, 1, 0, output.data ) );

        Func blur_x("blur_x"), blur_y("blur_y");
        Var x("x"), y("y"), xi("xi"), yi("yi");

        // The algorithm
        blur_x(x, y) = (in(x, y) + in(x+1, y) + in(x+2, y))/3;
        blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2))/3;

        // How to schedule it
        blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
        blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);

        blur_y.realize(out);
    }
}
