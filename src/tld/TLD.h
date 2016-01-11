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
 * TLD.h
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#ifndef TLD_H_
#define TLD_H_

#include <opencv/cv.h>

#include "MedianFlowTracker.h"
#include "DetectorCascade.h"

using namespace cv;
using namespace std;

namespace tld {

    class TLD {
            void storeCurrentData();
            void fuseHypotheses();
            void learn();
            void initialLearning();
        public:
            bool trackerEnabled;
            bool detectorEnabled;
            bool learningEnabled;
            bool alternating;

            bool valid;
            bool wasValid;
            Mat prevImg;
            Mat currImg;

            shared_ptr<Rect> prevBB;
            shared_ptr<Rect> currBB;

            shared_ptr<MedianFlowTracker> medianFlowTracker;
            shared_ptr<DetectorCascade> detectorCascade;
            shared_ptr<NNClassifier> nnClassifier;

            float currConf;
            bool learning;

            TLD();
            virtual ~TLD();
            void release();
            void selectObject(Mat img, Rect const & bb);
            void processImage(Mat img);
            void writeToFile(const char * path);
            void readFromFile(const char * path);
            void drawDetection(IplImage * img) const;
            Mat  drawPosterios();

        private :
            IplImage * _img_posterios;
    };

} /* namespace tld */
#endif /* TLD_H_ */
