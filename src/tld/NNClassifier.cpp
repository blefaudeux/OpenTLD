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
 * NNClassifier.cpp
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 */

#include "NNClassifier.h"
#include "DetectorCascade.h"
#include "TLDUtil.h"

namespace tld {

NNClassifier::NNClassifier() {
	thetaFP = .5;
	thetaTP = .65;
}

NNClassifier::~NNClassifier() {
    release();
}

void NNClassifier::release() {
    falsePositives.clear();
    truePositives.clear();
}

float NNClassifier::ncc(float const * f1,float const * f2) {
	double corr = 0;
	double norm1 = 0;
	double norm2 = 0;

	int size = TLD_PATCH_SIZE*TLD_PATCH_SIZE;

	for (int i = 0; i<size; i++) {
		corr += f1[i]*f2[i];
		norm1 += f1[i]*f1[i];
		norm2 += f2[i]*f2[i];
	}
	// normalization to <0,1>

	return (corr / sqrt(norm1*norm2) + 1) / 2.0;
}

float NNClassifier::classifyPatch(NormalizedPatch & patch) {

    if(truePositives.empty()) {
		return 0;
	}

    if(falsePositives.empty()) {
		return 1;
	}

	float ccorr_max_p = 0;

	//Compare patch to positive patches
    for (auto const & item : truePositives)
    {
        float const ccorr = ncc( item.values, patch.values );

        if(ccorr > ccorr_max_p)
        {
            ccorr_max_p = ccorr;
        }
    }

	float ccorr_max_n = 0;

	//Compare patch to positive patches
    for(auto const & item : falsePositives)
    {
        float const ccorr = ncc(item.values, patch.values);

        if(ccorr > ccorr_max_n)
        {
			ccorr_max_n = ccorr;
		}
	}

    float const dN = 1-ccorr_max_n;
    float const dP = 1-ccorr_max_p;

    return dN/(dN+dP);
}

float NNClassifier::classifyBB(Mat img, Rect & bb) {
	NormalizedPatch patch;

	tldExtractNormalizedPatchRect(img, bb, patch.values);
    return classifyPatch(patch);

}

float NNClassifier::classifyWindow(Mat img, int windowIdx) {
	NormalizedPatch patch;

	int * bbox = &windows[TLD_WINDOW_SIZE*windowIdx];
	tldExtractNormalizedPatchBB(img, bbox, patch.values);

    return classifyPatch(patch);
}

bool NNClassifier::filter(Mat img, int windowIdx) {
	if(!enabled) return true;

	float conf = classifyWindow(img, windowIdx);

	if(conf < thetaTP) {
		return false;
	}

	return true;
}

void NNClassifier::learn(vector<NormalizedPatch> & patches) {
	//TODO: Randomization might be a good idea here

    for (auto & patch : patches)
    {
        float conf = classifyPatch(patch);

		if(patch.positive && conf <= thetaTP) {
            truePositives.push_back(patch);
		}

		if(!patch.positive && conf >= thetaFP) {
            falsePositives.push_back(patch);
		}
	}
}


} /* namespace tld */
