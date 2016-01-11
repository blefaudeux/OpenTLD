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
 * DetectionResult.cpp
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 */

#include "DetectionResult.h"
#include "TLDUtil.h"

namespace tld {

DetectionResult::DetectionResult() {
	containsValidData = false;
    numClusters = 0;
}

DetectionResult::~DetectionResult() {
	release();
}

void DetectionResult::init(int numWindows, int numTrees) {
    variances.resize(numWindows);
    posteriors.resize(numWindows);
    featureVectors.resize(numWindows*numTrees);
}

void DetectionResult::reset() {
	containsValidData = false;
	fgList.clear();
	confidentIndices.clear();
	numClusters = 0;
}

void DetectionResult::release() {
	fgList.clear();
    confidentIndices.clear();
	containsValidData = false;
}

} /* namespace tld */
