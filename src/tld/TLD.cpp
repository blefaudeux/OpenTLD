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
 * TLD.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */
#include "highgui.h"
#include "TLD.h"
#include "NNClassifier.h"
#include "TLDUtil.h"
#include <iostream>

using namespace std;

namespace tld {

TLD::TLD() {
	trackerEnabled = true;
	detectorEnabled = true;
	learningEnabled = true;
	alternating = false;
	valid = false;
	wasValid = false;
	learning = false;

    detectorCascade.reset( new DetectorCascade() );
    medianFlowTracker.reset( new MedianFlowTracker() );

    nnClassifier = detectorCascade->nnClassifier;
	_img_posterios = NULL;
}

TLD::~TLD() {
	storeCurrentData();
    if(_img_posterios)cvReleaseImage(&_img_posterios);
}

void TLD::release() {
	detectorCascade->release();
    medianFlowTracker->cleanPreviousData();
}

void TLD::storeCurrentData() {
	prevImg.release();
	prevImg = currImg; //Store old image (if any)
	prevBB = currBB;		//Store old bounding box (if any)

	detectorCascade->cleanPreviousData(); //Reset detector results
	medianFlowTracker->cleanPreviousData();

	wasValid = valid;
}

void TLD::selectObject(Mat img, Rect const & bb) {
	//Delete old object
	detectorCascade->release();

	//Init detector cascade
    detectorCascade->objWidth = bb.width;
    detectorCascade->objHeight = bb.height;
	detectorCascade->init();

	currImg = img;
    currBB.reset(new Rect(bb));
	currConf = 1;
	valid = true;

	initialLearning();

}

void TLD::processImage(Mat img)
{
	storeCurrentData();
	Mat grey_frame;
	cvtColor( img,grey_frame, CV_RGB2GRAY );
	currImg = grey_frame; // Store new image , right after storeCurrentData();

    if(trackerEnabled && prevBB)
    {
        medianFlowTracker->track(prevImg, currImg, *prevBB.get());
	}

	if(detectorEnabled && (!alternating || medianFlowTracker->trackerBB == NULL)) {
		detectorCascade->detect(grey_frame);
	}

	fuseHypotheses();
	learn();
}


void TLD::drawDetection( IplImage * img) const {
	detectorCascade->drawDetection(img);
}


void TLD::fuseHypotheses() {
    auto const & trackerBB = medianFlowTracker->trackerBB;
    auto const & detectorBB = detectorCascade->detectionResult->detectorBB;
    int const numClusters = detectorCascade->detectionResult->numClusters;

    currBB.reset();
	currConf = 0;
	valid = false;

	float confDetector = 0;

    if(numClusters == 1)
    {
        confDetector = nnClassifier->classifyBB(currImg, *detectorBB.get());
	}

    if( trackerBB )
    {
        float const confTracker = nnClassifier->classifyBB(currImg, *trackerBB.get());

        if(numClusters == 1 && confDetector > confTracker && tldOverlapRectRect(*trackerBB.get(), *detectorBB.get()) < 0.5)
        {

            currBB.reset(new Rect(*detectorBB.get()));
			currConf = confDetector;
        }
        else
        {
            currBB.reset( new Rect(*trackerBB.get()));
			currConf = confTracker;

            if(confTracker > nnClassifier->thetaTP)
            {
				valid = true;

            } else if(wasValid && confTracker > nnClassifier->thetaFP)
            {
				valid = true;
			}
		}
    }
    else if(numClusters == 1)
    {
        currBB.reset( new Rect(*detectorBB.get()));
		currConf = confDetector;
	}

	/*
	float var = CalculateVariance(patch.values, nn->patch_size*nn->patch_size);

	if(var < min_var) { //TODO: Think about incorporating this
		printf("%f, %f: Variance too low \n", var, classifier->min_var);
		valid = 0;
	}*/
}

void TLD::initialLearning() {
	learning = true; //This is just for display purposes

    std::shared_ptr<DetectionResult> detectionResult = detectorCascade->detectionResult;

	detectorCascade->detect(currImg);

	//This is the positive patch
	NormalizedPatch patch;
    tldExtractNormalizedPatchRect(currImg, *currBB.get(), patch.values);
	patch.positive = 1;

	float initVar = tldCalcVariance(patch.values, TLD_PATCH_SIZE*TLD_PATCH_SIZE);
	detectorCascade->varianceFilter->minVar = initVar/2;


	float * overlap = new float[detectorCascade->numWindows];
    tldOverlapRect(detectorCascade->windows, detectorCascade->numWindows, *currBB.get(), overlap);

	//Add all bounding boxes with high overlap

	vector< pair<int,float> > positiveIndices;
	vector<int> negativeIndices;

	//First: Find overlapping positive and negative patches

	for(int i = 0; i < detectorCascade->numWindows; i++) {

		if(overlap[i] > 0.6) {
			positiveIndices.push_back(pair<int,float>(i,overlap[i]));
		}

		if(overlap[i] < 0.2) {
			float variance = detectionResult->variances[i];

			if(!detectorCascade->varianceFilter->enabled || variance > detectorCascade->varianceFilter->minVar) { //TODO: This check is unnecessary if minVar would be set before calling detect.
				negativeIndices.push_back(i);
			}
		}
	}

	sort(positiveIndices.begin(), positiveIndices.end(), tldSortByOverlapDesc);

	vector<NormalizedPatch> patches;

	patches.push_back(patch); //Add first patch to patch list

	int numIterations = min<size_t>(positiveIndices.size(), 10); //Take at most 10 bounding boxes (sorted by overlap)
	for(int i = 0; i < numIterations; i++) {
		int idx = positiveIndices.at(i).first;
		//Learn this bounding box
		//TODO: Somewhere here image warping might be possible
		detectorCascade->ensembleClassifier->learn(true, &detectionResult->featureVectors[detectorCascade->numTrees*idx]);
	}

	srand(1); //TODO: This is not guaranteed to affect random_shuffle

	random_shuffle(negativeIndices.begin(), negativeIndices.end());

	//Choose 100 random patches for negative examples
	for(size_t i = 0; i < min<size_t>(100,negativeIndices.size()); i++) {
		int idx = negativeIndices.at(i);

		NormalizedPatch patch;
		tldExtractNormalizedPatchBB(currImg, &detectorCascade->windows[TLD_WINDOW_SIZE*idx], patch.values);
		patch.positive = 0;
		patches.push_back(patch);
	}

	detectorCascade->nnClassifier->learn(patches);

	delete[] overlap;

}

//Do this when current trajectory is valid
void TLD::learn() {
	if(!learningEnabled || !valid || !detectorEnabled) {
		learning = false;
		return;
	}
	learning = true;

    std::shared_ptr<DetectionResult> detectionResult = detectorCascade->detectionResult;

	if(!detectionResult->containsValidData) {
		detectorCascade->detect(currImg);
	}

	//This is the positive patch
	NormalizedPatch patch;
    tldExtractNormalizedPatchRect(currImg, *currBB.get(), patch.values);

	float * overlap = new float[detectorCascade->numWindows];
    tldOverlapRect(detectorCascade->windows, detectorCascade->numWindows, *currBB.get(), overlap);

	//Add all bounding boxes with high overlap

	vector<pair<int,float> > positiveIndices;
	vector<int> negativeIndices;
	vector<int> negativeIndicesForNN;

	//First: Find overlapping positive and negative patches

	for(int i = 0; i < detectorCascade->numWindows; i++) {

		if(overlap[i] > 0.6) {
			positiveIndices.push_back(pair<int,float>(i,overlap[i]));
		}

		if(overlap[i] < 0.2) {
			if(!detectorCascade->ensembleClassifier->enabled || detectionResult->posteriors[i] > 0.1) { //TODO: Shouldn't this read as 0.5?
				negativeIndices.push_back(i);
			}

			if(!detectorCascade->ensembleClassifier->enabled || detectionResult->posteriors[i] > 0.5) {
				negativeIndicesForNN.push_back(i);
			}

		}
	}

	sort(positiveIndices.begin(), positiveIndices.end(), tldSortByOverlapDesc);

	vector<NormalizedPatch> patches;

	patch.positive = 1;
	patches.push_back(patch);
	//TODO: Flip


	int numIterations = min<size_t>(positiveIndices.size(), 10); //Take at most 10 bounding boxes (sorted by overlap)
	for(size_t i = 0; i < negativeIndices.size(); i++) {
		int idx = negativeIndices.at(i);
		//TODO: Somewhere here image warping might be possible
		detectorCascade->ensembleClassifier->learn(false, &detectionResult->featureVectors[detectorCascade->numTrees*idx]);
	}

	//TODO: Randomization might be a good idea
	for(int i = 0; i < numIterations; i++) {
		int idx = positiveIndices.at(i).first;
		//TODO: Somewhere here image warping might be possible
		detectorCascade->ensembleClassifier->learn(true, &detectionResult->featureVectors[detectorCascade->numTrees*idx]);
	}

	for(size_t i = 0; i < negativeIndicesForNN.size(); i++) {
		int idx = negativeIndicesForNN.at(i);

		NormalizedPatch patch;
		tldExtractNormalizedPatchBB(currImg, &detectorCascade->windows[TLD_WINDOW_SIZE*idx], patch.values);
		patch.positive = 0;
		patches.push_back(patch);
	}

	detectorCascade->nnClassifier->learn(patches);

	//cout << "NN has now " << detectorCascade->nnClassifier->truePositives->size() << " positives and " << detectorCascade->nnClassifier->falsePositives->size() << " negatives.\n";

	delete[] overlap;
}

typedef struct {
	int index;
	int P;
	int N;
} TldExportEntry;

void TLD::writeToFile(const char * path) {
    std::shared_ptr<NNClassifier> nn = detectorCascade->nnClassifier;
    std::shared_ptr<EnsembleClassifier> ec = detectorCascade->ensembleClassifier;

	FILE * file = fopen(path, "w");
	fprintf(file,"#Tld ModelExport\n");
	fprintf(file,"%d #width\n", detectorCascade->objWidth);
	fprintf(file,"%d #height\n", detectorCascade->objHeight);
	fprintf(file,"%f #min_var\n", detectorCascade->varianceFilter->minVar);
    fprintf(file,"%d #Positive Sample Size\n", nn->truePositives.size());


    for(size_t s = 0; s < nn->truePositives.size();s++) {
        float * imageData =nn->truePositives[s].values;
		for(int i = 0; i < TLD_PATCH_SIZE; i++) {
			for(int j = 0; j < TLD_PATCH_SIZE; j++) {
				fprintf(file, "%f ", imageData[i*TLD_PATCH_SIZE+j]);
			}
			fprintf(file, "\n");
		}
	}

    fprintf(file,"%d #Negative Sample Size\n", nn->falsePositives.size());

    for(size_t s = 0; s < nn->falsePositives.size();s++)
    {
        float * imageData = nn->falsePositives[s].values;
        for(int i = 0; i < TLD_PATCH_SIZE; i++)
        {
            for(int j = 0; j < TLD_PATCH_SIZE; j++)
            {
				fprintf(file, "%f ", imageData[i*TLD_PATCH_SIZE+j]);
			}
			fprintf(file, "\n");
		}
	}

    fprintf(file,"%d #numtrees\n", ec->dtc.numTrees);
    detectorCascade->numTrees = ec->dtc.numTrees;
    fprintf(file,"%d #numFeatures\n", ec->dtc.numFeatures);
    detectorCascade->numFeatures = ec->dtc.numFeatures;
    for(int i = 0; i < ec->dtc.numTrees; i++) {
		fprintf(file, "#Tree %d\n", i);

        for(int j = 0; j < ec->dtc.numFeatures; j++) {
            float * features = ec->features + 4*ec->dtc.numFeatures*i + 4*j;
			fprintf(file,"%f %f %f %f # Feature %d\n", features[0], features[1], features[2], features[3], j);
		}

		//Collect indices
		vector<TldExportEntry> list;

        for(int index = 0; index < pow(2.0f, ec->dtc.numFeatures); index++) {
			int p = ec->positives[i * ec->numIndices + index];
			if(p != 0) {
				TldExportEntry entry;
				entry.index = index;
				entry.P = p;
				entry.N = ec->negatives[i * ec->numIndices + index];
				list.push_back(entry);
			}
		}

		fprintf(file,"%d #numLeaves\n", list.size());
		for(size_t j = 0; j < list.size(); j++) {
			TldExportEntry entry = list.at(j);
			fprintf(file,"%d %d %d\n", entry.index, entry.P, entry.N);
		}
	}

	fclose(file);

}

void TLD::readFromFile(const char * path) {
	release();

    std::shared_ptr<NNClassifier> nn = detectorCascade->nnClassifier;
    std::shared_ptr<EnsembleClassifier> ec = detectorCascade->ensembleClassifier;

	FILE * file = fopen(path, "r");

	if(file == NULL) {
		printf("Error: Model not found: %s\n", path);
		exit(1);
	}

	int MAX_LEN=255;
	char str_buf[255];
	fgets(str_buf, MAX_LEN, file); /*Skip line*/

	fscanf(file,"%d \n", &detectorCascade->objWidth);
	fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/
	fscanf(file,"%d \n", &detectorCascade->objHeight);
	fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

	fscanf(file,"%f \n", &detectorCascade->varianceFilter->minVar);
	fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

	int numPositivePatches;
	fscanf(file, "%d \n", &numPositivePatches);
	fgets(str_buf, MAX_LEN, file); /*Skip line*/


	for(int s = 0; s < numPositivePatches; s++) {
		NormalizedPatch patch;

		for(int i = 0; i < 15; i++) { //Do 15 times

			fgets(str_buf, MAX_LEN, file); /*Read sample*/

			char * pch;
			pch = strtok (str_buf," ");
			int j = 0;
			while (pch != NULL)
			{
				float val = atof(pch);
				patch.values[i*TLD_PATCH_SIZE+j] = val;

				pch = strtok (NULL, " ");

				j++;
			}
		}

        nn->truePositives.push_back(patch);
	}

	int numNegativePatches;
	fscanf(file, "%d \n", &numNegativePatches);
	fgets(str_buf, MAX_LEN, file); /*Skip line*/


	for(int s = 0; s < numNegativePatches; s++) {
		NormalizedPatch patch;
		for(int i = 0; i < 15; i++) { //Do 15 times

			fgets(str_buf, MAX_LEN, file); /*Read sample*/

			char * pch;
			pch = strtok (str_buf," ");
			int j = 0;
			while (pch != NULL)
			{
				float val = atof(pch);
				patch.values[i*TLD_PATCH_SIZE+j] = val;

				pch = strtok (NULL, " ");

				j++;
			}
		}

        nn->falsePositives.push_back(patch);
	}

    fscanf(file,"%d \n", &ec->dtc.numTrees);
    detectorCascade->numTrees = ec->dtc.numTrees;
	fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

    fscanf(file,"%d \n", &ec->dtc.numFeatures);
    detectorCascade->numFeatures = ec->dtc.numFeatures;
	fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

    int size = 2 * 2 * ec->dtc.numFeatures * ec->dtc.numTrees;
	ec->features = new float[size];
    ec->numIndices = pow(2.0f, ec->dtc.numFeatures);
	ec->initPosteriors();

    for(int i = 0; i < ec->dtc.numTrees; i++) {
		fgets(str_buf, MAX_LEN, file); /*Skip line*/

        for(int j = 0; j < ec->dtc.numFeatures; j++) {
            float * features = ec->features + 4*ec->dtc.numFeatures*i + 4*j;
			fscanf(file, "%f %f %f %f",&features[0], &features[1], &features[2], &features[3]);
			fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/
		}

		/* read number of leaves*/
		int numLeaves;
		fscanf(file,"%d \n", &numLeaves);
		fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

		for(int j = 0; j < numLeaves; j++) {
			TldExportEntry entry;
			fscanf(file,"%d %d %d \n", &entry.index, &entry.P, &entry.N);
			ec->updatePosterior(i, entry.index, 1, entry.P);
			ec->updatePosterior(i, entry.index, 0, entry.N);
		}
	}

	detectorCascade->initWindowsAndScales();
	detectorCascade->initWindowOffsets();

	detectorCascade->propagateMembers();

	detectorCascade->initialised = true;

	ec->initFeatureOffsets();

}

static void hsv2rgb(double h,double s,double v,unsigned char *R,unsigned char*G,unsigned char *B) {
	double r=0,g=0,b=0;
	double f,p,q,t;
	int i;

	if( s == 0 ) { r = v;g = v;b = v; }
	else {
		if(h >= 360.)
			h = 0.0;
		h /= 60.;
		i = (int) h;
		f = h - i;
		p = v*(1-s);
		q = v*(1-(s*f));
		t = v*(1-s*(1-f));
		switch(i) {
		case 0: r = v;g = t;b = p;break;
		case 1: r = q;g = v;b = p;break;
		case 2: r = p;g = v;b = t;break;
		case 3: r = p;g = q;b = v;break;
		case 4: r = t;g = p;b = v;break;
		case 5: r = v;g = p;b = q;break;
		default: r = 1.0; b = 1.0;b = 1.0; break;
		}
	}
	*R = (unsigned char)(255*r);
	*G = (unsigned char)(255*g);
	*B = (unsigned char)(255*b);
}

#define MAX_NB_COLOR 1024
static CvScalar COLOR[MAX_NB_COLOR];
static CvScalar * DRAW_vector_map(int size) {
	size = MIN(size , 1024);
	double e = 240./size;
	double h = 0.;
	double s = 1.;
	double v = 1.;
	int i;

	for ( i = size-1 ; i >=0 ; i--) {
		unsigned char R,G,B;
		hsv2rgb(h,s,v,&R,&G,&B);
		COLOR[i] = CV_RGB(R,G,B);
		h += e;
	}
	return &COLOR[0];
}

Mat TLD::drawPosterios() {
    int T = detectorCascade->numTrees;
	int I = detectorCascade->ensembleClassifier->numIndices;
	if(_img_posterios == NULL) {
		_img_posterios = cvCreateImage(cvSize(I,T),8,3);
	}
	CvScalar * c = DRAW_vector_map(1000);
	for (int t = 0 ; t < T ;t++) {
		unsigned char * ptr = (unsigned char *)_img_posterios->imageData + t*_img_posterios->widthStep;
		for (int i = 0 ; i < I ; i++) {
			float p = MIN(0.999,detectorCascade->ensembleClassifier->posteriors[i + t * I]*10);
			CvScalar color = c[int(p*999)];
			ptr[i*3+0] = color.val[0];
			ptr[i*3+1] = color.val[1];
			ptr[i*3+2] = color.val[2];
		}
	}
	Mat img4;
	cv::resize(Mat(_img_posterios),img4,cv::Size(I*2,T*10),0.,0.,0);
	//imshow("Posterios",img4);
	return img4;
}

} /* namespace tld */
