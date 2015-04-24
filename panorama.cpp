/**
 *  @file    panorama.cpp
 *  @author  LI Haonan(20026517)
 *  @date    04/2015
 *
 *  @brief COMP 5421, Project 2, Panorama Stitching
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <dirent.h>
#include <stdio.h>
#include <vector>
#include <cstring>
#include <iostream>
#define Q 1e-6f
#define R 1e-1f

std::vector<int> surfMatch (cv::Mat image1,cv::Mat image2, int lastX, int lastY) {

    cv::Mat image_matches;
    cv::SurfFeatureDetector detector(400);

    //Calculate keypoints
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<int> shift;

    detector.detect( image1, keypoints_1 );
    detector.detect( image2, keypoints_2 );

    //Calculate descriptors (feature vectors)
    cv::SurfDescriptorExtractor extractor;

    cv::Mat descriptors_1, descriptors_2;

    extractor.compute( image1, keypoints_1, descriptors_1 );
    extractor.compute( image2, keypoints_2, descriptors_2 );

    //Matching descriptor vectors using FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector< cv::DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;

    //Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ ) {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< cv::DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ ) {
        if( matches[i].distance <  std::max (2*min_dist, 0.02)) {
            good_matches.push_back( matches[i]);
        }
    }

    //-- Draw only "good" matches
    drawMatches( image1, keypoints_1, image2, keypoints_2,
                  good_matches, image_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                  std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imshow( "Good Matches", image_matches);
    cv::waitKey(0);

    // Using Kalman Filter to reduce possible noise.
    // Assume noise are distributed as Gaussian form.
    double x = 0, y = 0;
    double xHat = 0, yHat = 0;
    double kGainX = 0, kGainY = 0;
    double pX = 0, pY = 0;
    double kalmanPX = 0, kalmanPY = 0;
    double detX = 0, detY = 0;
    int counter;
    double xtemp = 0, ytemp = 0;

    for (int i = 0; i < good_matches.size(); i++){
        x = xHat;
        pX = kalmanPX;

        y = yHat;
        pY = kalmanPY;

        detX = keypoints_2[good_matches[i].trainIdx].pt.x-keypoints_1[good_matches[i].queryIdx].pt.x;
        detY = keypoints_2[good_matches[i].trainIdx].pt.y-keypoints_1[good_matches[i].queryIdx].pt.y;

        //Lazy evaluation to remove unwanted matches.
        //1. det between every pair should share same sign.
        //2. the vary range should be limited in narrow area.
        if((detY*lastY>=0)&&(abs(detY) < abs(lastY)+10)){
            if ((detX*lastX>=0)&&((lastX > 0 ? (abs(detX)< image2.cols):(abs(detX) > (image1.cols-image2.cols))))){

            kGainX = pX/(pX+R);
            kalmanPX = (1-kGainX*pX);
            xHat = x + kGainX*(detX-x);

            kGainY = pY/(pY+R);
            kalmanPY = (1-kGainY*pY);
            yHat = y + kGainY*(detY-y);

            xtemp = detX;
            ytemp = detY;
            counter ++;
            }
        }
    }

    shift.push_back(counter==1?int(xtemp):int(xHat));
    shift.push_back(counter==1?int(ytemp):int(yHat));

    return shift;
}

int main(int argc, char const* argv[])
{
    //Read images from directory(argv)
    DIR*dir;
    struct dirent *ent;
    std::vector<cv::Mat> image;
    std::vector<std::string> files;
    if ((dir = opendir (argv[1])) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            if (ent -> d_type == 0x8) {
                std::string filePath = std::string(argv[1])+"/"+std::string(ent->d_name);
                files.push_back(filePath);
            }
        }
        //Sorting before storing into vector, in order to make sequence correct
        std::sort(files.begin(),files.end());
        for (int i = 0; i < files.size(); i++){
            std::cout << files[i]<< std::endl;
            image.push_back(cv::imread(files[i],CV_LOAD_IMAGE_COLOR));
            resize(image[i], image[i], cvSize(0,0), 0.5, 0.5,cv::INTER_CUBIC);
        }
        closedir (dir);
    } else {
        printf ("%s\n","Cannot Read File");
        return -1;
    }
    printf ("Total number of images: %d\n", image.size());

    cv::Mat result = image[0];

    /*
     * For loop:
     *     `result` and a new image:
     *     Calculate diff x and diff y
     *     align them together as `temp`
     *     `result` = `temp`
     */
    int detX = 0, detY = 0;
    int row = 0, col = 0;
    int prevRow = 0, prevCol = 0;
    for (int i = 1; i < image.size(); i++) {
        /*
         * input:
         *      result, image
         *
         * output:
         *
         * if x > 0
         *  ------------------------
         *  |   result   |  image  |
         *  ------------------------
         *  x < 0
         *  ------------------------
         *  |  image  |   result   |
         *  ------------------------
         *
         * if y > 0
         *  -----------
         *  |  image  |
         *  -----------
         *  |  result |
         *  -----------
         *  y < 0
         *  -----------
         *  |  result |
         *  -----------
         *  |  image  |
         *  -----------
         */
        printf ("Working on the %dth image.\n",i);
        std::vector<int> matched_diff= surfMatch (result,image[i],detX,detY);

        detX = matched_diff[0];
        detY = matched_diff[1];
        printf ("dx:%d\n", matched_diff[0]);
        printf ("dy:%d\n", matched_diff[1]);

        //Calculate new size of next `result`
        row = int(((matched_diff[1]<0)?image[i].rows:result.rows)+abs(matched_diff[1]));
        col = int(((matched_diff[0]<0)?image[i].cols:result.cols)+abs(matched_diff[0]));

        //The new image size should not be smaller than previous one, or Memory Corruption will happen
        cv::Mat temp(row>prevRow?row:prevRow,col>prevCol?col:prevCol,CV_8UC3);
        prevCol = temp.cols;
        prevRow = temp.rows;

        //Set all pixel into Black
        temp.setTo(0);

        //Calculate the new position of pixels within old `result` and `image[i]` when copy them into new `result`
        int resultx_begin = 0             + int((matched_diff[0]<0)?0:abs(matched_diff[0]));
        int resultx_end   = result.cols   + int((matched_diff[0]<0)?0:abs(matched_diff[0]));
        int imagex_begin  = 0             + int((matched_diff[0]>0)?0:abs(matched_diff[0]));
        int imagex_end    = image[i].cols + int((matched_diff[0]>0)?0:abs(matched_diff[0]));

        int resulty_begin = 0             + int((matched_diff[1]<0)?0:abs(matched_diff[1]));
        int resulty_end   = result.rows   + int((matched_diff[1]<0)?0:abs(matched_diff[1]));
        int imagey_begin  = 0             + int((matched_diff[1]>0)?0:abs(matched_diff[1]));
        int imagey_end    = image[i].rows + int((matched_diff[1]>0)?0:abs(matched_diff[1]));

        //Calculate the coordinate of mixed rectangle in new `result`
        int mixedx_begin  = int(abs(matched_diff[0]));
        int mixedy_begin  = int(abs(matched_diff[1]));
        int mixedx_end    = int(matched_diff[0]<0?result.cols:image[i].cols);
        int mixedy_end    = int(matched_diff[1]<0?result.rows:image[i].rows);

        // Copy old `result` to its new position
        int m = 0;
        for (int j = resultx_begin ; j < resultx_end;j++) {
            int n = 0;
            for (int k = resulty_begin; k< resulty_end;k++) {
                temp.at<cv::Vec3b>(k,j) = result.at<cv::Vec3b>(n,m);
                n++;
            }
            m++;
        }

        // Copy image[i] to its new position
        m = 0;
        for (int j = imagex_begin; j < imagex_end;j++) {
            int n = 0;
            for (int k = imagey_begin; k< imagey_end;k++){
                temp.at<cv::Vec3b>(k,j) = image[i].at<cv::Vec3b>(n,m);
                n++;
            }
            m++;
        }


        /* Calculate pixel value of mixed area
         *
         * using image feathering to blend the mixed area
         *
         * window size: # of cols
         * weight change:
         * left  Image 1 ---------> 0
         * right Image 0 ---------> 1
         */
        m = int(matched_diff[0]<0?(-(matched_diff[0])):0);
        int p = int(matched_diff[0]>0?(matched_diff[0]):0);
        for (int j = mixedx_begin; j < mixedx_end; j++) {
            int n = int(matched_diff[1]>0?0:-matched_diff[1]);
            int q = int(matched_diff[1]<0?0:matched_diff[1]);
            for (int k = mixedy_begin; k < mixedy_end; k++) {
                temp.at<cv::Vec3b>(k,j)[0] = (((matched_diff[0]<0)?(mixedx_end-j):(j-mixedx_begin))*result.at<cv::Vec3b>(n,m)[0] +((matched_diff[0]>0)?(mixedx_end-j):(j-mixedx_begin))*image[i].at<cv::Vec3b>(q,p)[0])/(mixedx_end-mixedx_begin);
                temp.at<cv::Vec3b>(k,j)[1] = (((matched_diff[0]<0)?(mixedx_end-j):(j-mixedx_begin))*result.at<cv::Vec3b>(n,m)[1] +((matched_diff[0]>0)?(mixedx_end-j):(j-mixedx_begin))*image[i].at<cv::Vec3b>(q,p)[1])/(mixedx_end-mixedx_begin);
                temp.at<cv::Vec3b>(k,j)[2] = (((matched_diff[0]<0)?(mixedx_end-j):(j-mixedx_begin))*result.at<cv::Vec3b>(n,m)[2] +((matched_diff[0]>0)?(mixedx_end-j):(j-mixedx_begin))*image[i].at<cv::Vec3b>(q,p)[2])/(mixedx_end-mixedx_begin);
                n ++;
                q ++;
            }
            m ++;
            p ++;
        }
        result = temp.clone();
        cv::imshow( "Result", result);
        cv::waitKey(0);
    }
    cv::imwrite( "result.jpg", result);
    printf("Done, image stored as %s/result.jpg\n",argv[1]);

    return 0;
}
