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
 * EnsembleClassifier.h
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 */

#ifndef ENSEMBLECLASSIFIER_H_
#define ENSEMBLECLASSIFIER_H_

#include <opencv/cv.h>

using namespace cv;
using namespace std;

namespace tld {

    // Fw
    class DetectorCascade;

    class EnsembleClassifier
    {
            friend class TLD;

        public:
            EnsembleClassifier( DetectorCascade & dc);
            virtual ~EnsembleClassifier();
            void init();
            void initFeatureLocations();
            void initFeatureOffsets();
            void initPosteriors();
            void release();
            void nextIteration(Mat img);
            void classifyWindow(int windowIdx);
            void updatePosterior(int treeIdx, int idx, int positive, int amount);
            void learn(int positive, int * featureVector);
            bool filter(int i);

            float calcConfidence(int * featureVector);
            int calcFernFeature(int windowIdx, int treeIdx);
            void calcFeatureVector(int windowIdx, int * featureVector);
            void updatePosteriors(int *featureVector, int positive, int amount);

        public:
            bool enabled;

        private:
            int* featureOffsets;
            float* features;

            int numIndices;

            float * posteriors;
            int * positives;
            int * negatives;

            DetectorCascade & dtc;
            unsigned char* img;
    };

} /* namespace tld */
#endif /* ENSEMBLECLASSIFIER_H_ */
