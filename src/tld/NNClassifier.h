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
 * NNClassifier.h
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 */

#ifndef NNCLASSIFIER_H_
#define NNCLASSIFIER_H_

#include <vector>
#include <opencv/cv.h>
#include <memory>

#include "NormalizedPatch.h"
#include "DetectionResult.h"

using namespace std;
using namespace cv;

namespace tld {

    class NNClassifier {

        public:
            NNClassifier();
            virtual ~NNClassifier();

            void    release();
            float   classifyPatch(NormalizedPatch const & patch);
            float   classifyBB(Mat img, Rect &bb);
            float   classifyWindow(Mat img, int windowIdx);
            void    learn(vector<NormalizedPatch> const & patches);
            bool    filter(Mat img, int windowIdx);
            float   ncc(float const *f1, float const *f2);

            // Gets / Sets
            inline bool isEnabled() const { return _enabled; }
            inline float tp() const { return _thetaTP; }
            inline float fp() const { return _thetaFP; }
            void setTP(float th) { _thetaTP = th; }
            void setFP(float th) { _thetaFP = th; }

            vector<NormalizedPatch> const & truePositives() const { return _truePositives; }
            vector<NormalizedPatch> const & falsePositives() const { return _falsePositives; }

            void addTruePositive( NormalizedPatch const & patch) { _truePositives.push_back(patch); }
            void addFalsePositive( NormalizedPatch const & patch) { _falsePositives.push_back(patch); }
            void enable(bool status) { _enabled = status; }

        public:
            int * windows;
            std::shared_ptr<DetectionResult> detectionResult;

        private:
            bool    _enabled;
            float   _thetaFP;
            float   _thetaTP;

            vector<NormalizedPatch> _falsePositives;
            vector<NormalizedPatch> _truePositives;

    };

} /* namespace tld */
#endif /* NNCLASSIFIER_H_ */
