/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/
/*
 * VarianceFilter.h
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 *
 *  Coding style modifications by Benjamin Lefaudeux
 *  > moving to C++1x and Halide
 */

#ifndef VARIANCEFILTER_H_
#define VARIANCEFILTER_H_

#include <memory>
#include <opencv/cv.h>
#include "IntegralImage.h"
#include "DetectionResult.h"

using namespace cv;
using namespace std;

namespace tld {

    class VarianceFilter
    {
        public:
            VarianceFilter();
            virtual ~VarianceFilter();

            void release();
            void nextIteration(Mat img);
            bool filter(int idx);
            float calcVariance(int *off);

        public:
            bool enabled;
            int * windowOffsets;
            float minVar;

            shared_ptr<DetectionResult> detectionResult;
            unique_ptr<IntegralImage<int>> integralImg;
            unique_ptr<IntegralImage<long long>>integralImg_squared;
    };

} /* namespace tld */
#endif /* VARIANCEFILTER_H_ */
