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
 * MainX.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#include "Main.h"
#include "config.h"
#include "imAcq.h"
#include "gui.h"
#include "TLDUtil.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>

 void Main::doWork() {

    IplImage * img = imAcqGetImg(imAcq);
    IplImage * grey = cvCreateImage( cvGetSize(img), 8, 1 ); // FIXME: Memleak over there
    cvCvtColor( img,grey, CV_BGR2GRAY );

    auto detector = tld->detector();
    detector->imgWidth = grey->width;
    detector->imgHeight = grey->height;
    detector->imgWidthStep = grey->widthStep;

    if(selectManually) {

        CvRect box;
        if(getBBFromUser(img, box, gui) == PROGRAM_EXIT) {
            return;
        }

        if(initialBB == NULL) {
            initialBB = new int[4];
        }

        initialBB[0] = box.x;
        initialBB[1] = box.y;

        if (box.width < 0) {
            initialBB[0] += box.width;
            initialBB[2] = abs(box.width);
        }
        else {
            initialBB[2] = box.width;
        }
        if (box.height < 0) {
            initialBB[1] += box.height;
            initialBB[3] = abs(box.height);
        }
        else {
            initialBB[3] = box.height;
        }
    }



    FILE * resultsFile = NULL;

    if(printResults != NULL) {
        resultsFile = fopen(printResults, "w");
    }

    bool reuseFrameOnce = false;
    bool skipProcessingOnce = false;
    if(loadModel && modelPath != NULL) {
        tld->readFromFile(modelPath);
        reuseFrameOnce = true;
    } else if(initialBB != NULL) {
        Rect bb = tldArrayToRect(initialBB);

        printf("Starting at %d %d %d %d\n", bb.x, bb.y, bb.width, bb.height);

        tld->selectObject(grey, bb);
        skipProcessingOnce = true;
        reuseFrameOnce = true;
    }

    cvReleaseImage(&grey);

    while(imAcqHasMoreFrames(imAcq)) {
        double tic = cvGetTickCount();


        if(!reuseFrameOnce) {
            img = imAcqGetImg(imAcq);
            if(img == NULL) {
                printf("current image is NULL, assuming end of input.\n");
                break;
            }
            grey = cvCreateImage( cvGetSize(img), 8, 1 );
            cvCvtColor( img, grey, CV_BGR2GRAY );
        }

        if(!skipProcessingOnce) {
            cv::blur(Mat(img),Mat(img),cv::Size(3,3));
            tld->processImage(img);

        } else {
            skipProcessingOnce = false;
        }

        if(printResults != NULL)
        {
            auto currBB = tld->boundingBox();

            if( currBB.get() )
            {
                fprintf(resultsFile, "%d %.2d %.2d %.2d %.2d %f\n", imAcq->currentFrame-1,
                        currBB->x, currBB->y, currBB->width, currBB->height, tld->confidence());
            }
            else
            {
                fprintf(resultsFile, "%d NaN NaN NaN NaN NaN\n", imAcq->currentFrame-1);
            }
        }

        float const toc = (cvGetTickCount() - tic)/cvGetTickFrequency() / 1000000;
        float const fps = 1/toc;

        float const tldConfidence = tld->confidence();
        int const confident = (tldConfidence >= threshold) ? 1 : 0;

        if(showOutput || saveDir != NULL)
        {
            char string[128];

            char learningString[10] = "";

            if(tld->isLearning())
            {
                strcpy(learningString, "Learning");
            }

            sprintf(string, "#%d,Posterior %.2f; fps: %.2f, #numwindows:%d, %s", imAcq->currentFrame-1,
                    tldConfidence, fps, tld->detector()->numWindows, learningString);

            CvScalar yellow = CV_RGB(255,255,0);
            CvScalar blue = CV_RGB(0,0,255);
            CvScalar black = CV_RGB(0,0,0);
            CvScalar white = CV_RGB(255,255,255);

            auto currBB = tld->boundingBox();

            if( currBB.get() )
            {
                int x_avg = (currBB->tl().x >> 1) + (currBB->br().x >> 1) + 100;
                int y_avg = (currBB->tl().y >> 1) + (currBB->br().y >> 1) + 120;
                std::ostringstream sin1;
                sin1 << (x_avg);
                std::string s2 = sin1.str();
                std::ostringstream sin2;
                sin2 << (y_avg);
                std::string s3 = sin2.str();
                std::string cmd = "./click -x " + s2 + " -y " + s3;
                //system(cmd.c_str());
                tld->drawDetection(img);
                cvRectangle(img,cvPoint(currBB->x, currBB->y),cvPoint(currBB->x + currBB->width, currBB->y + currBB->height),CV_RGB(0,255,255),2);
                Mat img_desciptor = tld->drawPosterios();
                Mat mat(img);
                img_desciptor.copyTo(mat(cv::Rect(0,0,img_desciptor.cols,img_desciptor.rows)));
            }

            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, 8);
            cvRectangle(img, cvPoint(0,0), cvPoint(img->width,50), black, CV_FILLED, 8, 0);
            cvPutText(img, string, cvPoint(25,25), &font, white);

            if(showForeground)
            {
                for(auto const & r : tld->detector()->detectionResult->fgList )
                {
                    cvRectangle(img, r.tl(),r.br(), white, 1);
                }

            }

            if(showOutput) {
                gui->showImage(img);
                char key = gui->getKey();

                if(key == 'q') break;

                if(key == 'b')
                {
                    auto fg = tld->detector()->foregroundDetector;

                    if(fg->bgImg.empty())
                    {
                        fg->bgImg = cvCloneImage(grey);
                    }
                    else
                    {
                        fg->bgImg.release();
                    }
                }

                if(key == 'c')
                {
                    //clear everything
                    tld->release();
                }

                if(key == 'l') {
                    tld->setLearning(!tld->isLearning());
                    printf("LearningEnabled: %d\n", tld->isLearning());
                }

                if(key == 'a') {
                    tld->setAlternating(!tld->isAlternating());
                    printf("alternating: %d\n", tld->isAlternating());
                }

                if(key == 'e') {
                    tld->writeToFile(modelExportFile);
                }

                if(key == 'i') {
                    tld->readFromFile(modelPath);
                }

                if(key == 'r') {
                    CvRect box;

                    if(getBBFromUser(img, box, gui) == PROGRAM_EXIT) {
                        break;
                    }

                    tld->selectObject(grey, box);
                }
            }

            if(saveDir != NULL) {
                char fileName[256];
                sprintf(fileName, "%s/%.5d.png", saveDir, imAcq->currentFrame-1);

                cvSaveImage(fileName, img);
            }
        }

        if(!reuseFrameOnce) {
            cvReleaseImage(&img);
            cvReleaseImage(&grey);
        } else {
            reuseFrameOnce = false;
        }
    }

    if(exportModelAfterRun) {
        tld->writeToFile(modelExportFile);
    }
}
