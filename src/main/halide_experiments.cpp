
#include <Halide/include/Halide.h>
#include <numeric>
#include <halide_experiments.h>
#include <Halide/tools/halide_image_io.h>

namespace hexp
{
    using namespace Halide;

    void halide_blur_minimal(cv::Mat output, cv::Mat input)
    {
        // Load the picture from the OpenCV raw buffer
        Image<uint8_t> in( Buffer(UInt(8), input.cols, input.rows, 1, 0, input.data ) );
        Image<uint8_t> out( Buffer(UInt(8), input.cols-2, input.rows-2, 1, 0, output.data ) );

        Var x, y, c;

        // Blur it horizontally:
        Func blur_x("blur_x");
        blur_x(x, y, c) = (in(x-1, y, c) + 2 * in(x, y, c) + in(x+1, y, c)) / 4;

        // Blur it vertically:
        Func blur_y("blur_y");
        blur_y(x, y, c) = (blur_x(x, y-1, c) + 2 * blur_x(x, y, c) + blur_x(x, y+1, c)) / 4;

        out.set_min(1, 1);
        blur_y.realize(out);
    }
}
