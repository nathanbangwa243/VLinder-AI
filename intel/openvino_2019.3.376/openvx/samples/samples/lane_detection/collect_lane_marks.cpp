/*
        Copyright 2019 Intel Corporation.
        This software and the related documents are Intel copyrighted materials,
        and your use of them is governed by the express license under which they
        were provided to you (End User License Agreement for the Intel(R) Software
        Development Products (Version May 2017)). Unless the License provides
        otherwise, you may not use, modify, copy, publish, distribute, disclose or
        transmit this software or the related documents without Intel's prior
        written permission.

        This software and the related documents are provided as is, with no
        express or implied warranties, other than those that are expressly
        stated in the License.
*/

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "collect_lane_marks.hpp"
#include <intel/vx_samples/perfprof.hpp>

// search range for line to scan edge pixels
#define LINE_SCAN_NEIGHBORHOOD 3

// maximal allowed slope for input lines
// if line slope dy/dx > 1/LINE_SLOPE_MAX_INV then line is ignored
#define LINE_SLOPE_MAX_INV 3

// minimal allowed length for detected lane marks in percent of processed image width
#define LANE_MIN_LENGTH_RATIO 0.125f

// number of iteration for RANSAC procedure
#define RANSAC_ITERATIONS 50

// threshol value for RANSAC procedure to detect valid inline points
#define RANSAC_THRESHOLD 3

/*! \brief Collect points from binary edges based on lines detected by HoughTransformP and stored in lines array.
* \details Function scans each line from lines array and checks if any non zero pixel in "edge" is exist. If pixel exists then its coordinates are added into output points array.
* \param [in] 8UC1 image with edge responce.
* \param [in] threshold that is used to detect points edge image.
* \param [in] line segments detected by HoughTransformP for example.
* \param [inout] output array of cv::Point. All detected point will be added into this array
*/
static void CollectValidPoints(
    const cv::Mat                   edge,
    const int                       threshold,
    const std::vector<cv::Vec4i>&   lines,
    std::vector<cv::Point>&         points)
{
    points.clear();
    for(int i = 0; i < lines.size(); ++i )
    {  // check each line detected by Hough transform function for intersection with scanned line
        cv::Point   p1(lines[i][0], lines[i][1]);
        cv::Point   p2(lines[i][2], lines[i][3]);
        int         dx = p2.x - p1.x;
        int         dy = p2.y - p1.y;

        //keep only horizontal lines
        if(LINE_SLOPE_MAX_INV * abs(dy) > abs(dx))
            continue;

        //swap points to always be left to right
        if(p1.x > p2.x)
            std::swap(p1,p2);

        for(int x = p1.x; x <= p2.x; x++)
        {// scan input line segment
            // calc y coordinate of intersection point between
            // line segment and current vertical line given by x
            // based on ratio (y-p1.y)/(x-p1.x)=dy/dx
            int y = (p1.y + ((x - p1.x) * dy) / dx);
            // calc range for edge pixels scaning
            int y0 = max(y - LINE_SCAN_NEIGHBORHOOD / 2, 0);
            int y1 = min(y + LINE_SCAN_NEIGHBORHOOD / 2, edge.rows - 1);
            int yBest = -1;
            int valBest = threshold;
            for(int y = y0; y <= y1; y++)
            {//scan vertical small segment with size LINE_SCAN_NEIGHBORHOOD
             // and find pixel with the best edge responce.
                int val = edge.at<unsigned char>(y, x);
                if(valBest < val)
                {
                    valBest = val;
                    yBest = y;
                }
            }
            if(yBest >= 0)
            {//finaly add new point into output array
                points.push_back(cv::Point(x, yBest));
            }
        }// check next x position
    }//check next line
}

/*! \brief Fitline using RANSAC outlier detection and sort input points into valid and novalid.
* \details You can find some detail here https://en.wikipedia.org/wiki/RANSAC
* \details M.A. Fischler and R.C. Bolles. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381?395, 1981.
* \param [inout] input point array. After return this array contain outliers
* \param [in]  minimal allowed line length.
* \param [out] output line parameters array.
* \param [out] array of inlier points that were used for line parameter estimation.
*/
static void FitLineRANSAC(
    std::vector<cv::Point>& points,
    const int               minLen,
    cv::Vec2f&              lineParams,
    std::vector<cv::Point>& pointsGood)
{
    int         countBest = 0;
    cv::Point   p0Best;
    cv::Point   p1Best;
    int         N = points.size();
    cv::Point*  p = &points[0];
    for(int iter = 0; iter < RANSAC_ITERATIONS; iter++ )
    {
        // take 2 random points to estimate line
        cv::Point p0 = p[rand() % N];
        cv::Point p1 = p[rand() % N];
        int       dx = p1.x - p0.x;
        int       dy = p1.y - p0.y;
        if(LINE_SLOPE_MAX_INV * abs(dy) > abs(dx))
            continue;
        if(abs(dx) < LINE_SCAN_NEIGHBORHOOD)
            continue; //early reject short lines

        // calc line parameters based on 2 points
        // the line equation y = K*x+Y0
        float       K = (float)dy / (float)dx;
        float       Y0 = p0.y - p0.x * K;
        int         count = 0;
        float       T2 = RANSAC_THRESHOLD*RANSAC_THRESHOLD;
        for(int k = 0; k < N; ++k)
        {
            float err = K * p[k].x + Y0 - p[k].y;
            if(err * err < T2)
                count++;
        }

        if(count > countBest)
        {  //remember best points pair
            p0Best = p0;
            p1Best = p1;
            countBest = count;
        }
    }// next RANSAC iteration

    pointsGood.clear();
    if(countBest > 2)
    {
        // calc line parameters based on 2 best points
        float   K = (float)(p1Best.y - p0Best.y) / (float)(p1Best.x - p0Best.x);
        float   Y0 = p0Best.y - p0Best.x * K;

        int x0 = min(p0Best.x, p1Best.x);
        int x1 = max(p0Best.x, p1Best.x);
        for(int k = 0; k < N; ++k)
        {
            cv::Point p = points[k];
            if(fabs( K * p.x + Y0 - p.y) < RANSAC_THRESHOLD)
            {
                x0 = min(x0, p.x);
                x1 = max(x1, p.x);
            }
        }
        if((x1 - x0) > minLen)
        {
            //accumulators for accurate line parameter estimation
            int XXSum = 0;
            int XYSum = 0;
            int YSum = 0;
            int XSum = 0;

            for(int k = 0,j = 0; k < N; ++k)
            {
                cv::Point p = points[k];
                if(fabs( K * p.x + Y0 - p.y)<RANSAC_THRESHOLD)
                {
                    pointsGood.push_back(p);
                    XSum += p.x;
                    YSum += p.y;
                    XYSum += p.x * p.y;
                    XXSum += p.x * p.x;
                }
                else
                {
                    points[j++] = p;
                }
            }
            points.resize(N - pointsGood.size());


            // new estiamtion
            float S = 1.0f / pointsGood.size();
            float XAver = XSum * S;
            float YAver = YSum * S;
            float XXCorr = XXSum * S - XAver * XAver;
            float XYCorr = XYSum * S - YAver * XAver;
            lineParams[0] = XYCorr / XXCorr;
            lineParams[1] = YAver - lineParams[0] * XAver;
        }
    }
}//FitLineRANSAC

PERFPROF_REGION_DEFINE(CollectLaneMarks)

/*! \brief Process lines estimated by Hough Transform
 * \param [in] image with filter responce that is used to check point to be part of lane border
 * \param [in] threshold to define strong and weak edge responce.
 * \param [in] array of line segments from Hough transform as (x0,y0,x1,y1).
 */
void CollectLaneMarks::Process(
    const cv::Mat&           edges,
    const int                edgeThreshold,
    const vector<cv::Vec4i>& lines)
{
    PERFPROF_REGION_AUTO(CollectLaneMarks)

    int H = edges.rows;

    CollectValidPoints(edges, edgeThreshold, lines, m_PointsAll);

    for(int lane = 0; lane < MAX_LANE_NUM; ++lane)
    {
        if(m_PointsAll.size() < 2)
        {// not enough points just clear point storage
            m_Points[lane].clear();
        }
        else
        {
            FitLineRANSAC(
                m_PointsAll,        // input points
                (int)(LANE_MIN_LENGTH_RATIO * edges.cols), // minimal lane length
                m_Lanes[lane],      // detected lane marks
                m_Points[lane]);    // inliers points
        }
    }// process next lane

    // choose 2 main lanes to draw detected road area
    m_L0 = -1;
    m_L1 = -1;
    float minH = H;
    for(int i0=0; i0<MAX_LANE_NUM; ++i0)for(int i1=i0+1; i1<MAX_LANE_NUM; ++i1)
    {
        if(m_Points[i0].size() == 0 || m_Points[i1].size() == 0)
           continue;
        float Y0 = m_Lanes[i0][1];
        float Y1 = m_Lanes[i1][1];
        float h = fabs(Y1 - Y0);
        if((2 * Y0 - H) * (2 * Y1 - H) > 0.0f)
             continue; // both lane marks on the same side
        if( h < 0.35*H)
             continue; // lanes are too close to each other
        if( h > 0.65*H)
            continue; // lanes are too far from each other
        if( h < minH )
        {
            minH = h;
            m_L0 = i0;
            m_L1 = i1;
        }
    }

    edges.copyTo(m_Edges8U);
    m_Lines = lines;
}


/*! \brief get bound of detected lane marks
 * \param [in] index of lane
 */
cv::Vec4i CollectLaneMarks::GetLaneBound(const int lane)
{
    cv::Vec4i res(-1, -1, -1, -1);
    if(m_Points[lane].size() > 0)
    {
        cv::Point p = m_Points[lane][0];
        res[0] = p.x;
        res[1] = p.y;
        res[2] = p.x;
        res[3] = p.y;
    }
    for(int i = 1; i < m_Points[lane].size(); ++i)
    {
        cv::Point p = m_Points[lane][i];
        if(res[0] > p.x ) res[0] = p.x;
        if(res[1] > p.y ) res[1] = p.y;
        if(res[2] < p.x ) res[2] = p.x;
        if(res[3] < p.y ) res[3] = p.y;
    }
    return res;
}

/*! \brief Draw detected lines over given image using given perspective transform
 * \param [inout] image to draw detected lane marks
 * \param [in] 3x3 perspective tranform matrix that is used to transform detected lane marks into debug image coordinate system.
 * \param [in] thickness of drawing line.
 * \param [in] thickness of drawing line ends.
 */
void CollectLaneMarks::DrawLanes(
    cv::Mat&        debugOut,
    const cv::Mat&  matPerspectiveTransform,
    const int       thicknessLine,
    const int       thicknessEnds,
    const bool      imageBGR)
{// draw OCV debug window with result
    cv::Scalar red =  (imageBGR)?cv::Scalar(0,0,255):cv::Scalar(255,0,0);
    for(int lane = 0; lane < MAX_LANE_NUM; ++lane)
    {// iterate over lane edges
        if(m_Points[lane].size() == 0)
            continue;
        float K = m_Lanes[lane][0];
        float Y = m_Lanes[lane][1];
        cv::Vec4i bound = GetLaneBound(lane);
        int   x0 = bound[0];
        int   x1 = bound[2];
        std::vector<cv::Point2f> in(2);
        std::vector<cv::Point2f> out(2);
        in[0] = cv::Point2f( x0, (x0 * K + Y) );
        in[1] = cv::Point2f( x1, (x1 * K + Y) );
        cv::perspectiveTransform(in, out, matPerspectiveTransform);
        if(thicknessLine > 0)
        {
            cv::line(debugOut, cv::Point(out[0]), cv::Point(out[1]), red, thicknessLine);
        }
        if(thicknessEnds > 0)
        {
            cv::circle(debugOut, cv::Point(out[0]), thicknessEnds, red, thicknessEnds);
            cv::circle(debugOut, cv::Point(out[1]), thicknessEnds, red, thicknessEnds);
        }
    }
}

/*! \brief Draw detection result over given image using given perspective transform
 * \param [inout] image to draw detected lane marks
 * \param [in] 3x3 perspective tranform matrix that is used to transform detected lane marks into debug image coordinate system.
 * \param [in] 1 means only base result drawing, 2 means additional Hough transform result drawiing.
 */
void CollectLaneMarks::DrawResult(
    cv::Mat&        debugOut,
    const cv::Mat&  matPerspectiveTransform,
    const int       visualization,
    const bool      imageBGR)
{
    cv::Scalar blue = (imageBGR)?cv::Scalar(255,0,0):cv::Scalar(0,0,255);
    cv::Scalar red =  (imageBGR)?cv::Scalar(0,0,255):cv::Scalar(255,0,0);
    cv::Scalar green =  cv::Scalar(0,255,0);
    int W = m_Edges8U.cols;
    int H = m_Edges8U.rows;
    if(visualization>1)
    {// draw additional debug info
        // draw Hough transform result for OCV pipline
        cv::Rect r = cv::Rect(0, 0, m_Edges8U.cols, m_Edges8U.rows);
        cv::Mat out = debugOut(r);
        cv::cvtColor(m_Edges8U,out,cv::COLOR_GRAY2BGR);
        DrawLanes(out, cv::Mat::eye(3, 3, CV_32F),0,3, imageBGR);
        for(int i=0; i<m_Lines.size(); ++i)
        {
            cv::Vec4i l = m_Lines[i];
            cv::line(out, cv::Point(l[0],l[1]), cv::Point(l[2],l[3]), blue);
            cv::circle(out, cv::Point(l[0],l[1]), 2, green);
            cv::circle(out, cv::Point(l[2],l[3]), 2, green);
        }
        cv::rectangle(debugOut,r, blue);
    }

    debugOut.copyTo(m_Overlay);

    if(m_L0>=0 && m_L1>=0)
    {// fill road area between 2 selected lanes
        std::vector<cv::Point>   cnts(4);
        float K0 = m_Lanes[m_L0][0];
        float Y0 = m_Lanes[m_L0][1];
        float K1 = m_Lanes[m_L1][0];
        float Y1 = m_Lanes[m_L1][1];
        int   x0 = 0;
        int   x1 = W;
        std::vector<cv::Point2f> in(4);
        std::vector<cv::Point2f> out(4);
        in[0] = cv::Point2f( x0, (x0 * K0 + Y0) );
        in[1] = cv::Point2f( x1, (x1 * K0 + Y0) );
        in[2] = cv::Point2f( x0, (x0 * K1 + Y1) );
        in[3] = cv::Point2f( x1, (x1 * K1 + Y1) );
        cv::perspectiveTransform(in, out, matPerspectiveTransform);
        cnts[0] = out[0];
        cnts[1] = out[1];
        cnts[2] = out[3];
        cnts[3] = out[2];
        if(m_LaneW<0)
            m_LaneW = H*0.5f;
        m_LaneW = m_LaneW*0.99f + fabs(Y0-Y1)*0.01f;
        // choose color based on distance to the nearest lane
        float dist = min(fabs(Y0-H*0.5f),fabs(Y1-H*0.5f));
        cv::fillConvexPoly(m_Overlay, cnts, (dist<m_LaneW*0.25f) ? red : green );
    }

    {//draw search area
        std::vector<cv::Point2f> in(4);
        std::vector<cv::Point2f> out(4);
        in[0] = cv::Point2f( 0, 0 );
        in[1] = cv::Point2f( W, 0 );
        in[2] = cv::Point2f( W, H );
        in[3] = cv::Point2f( 0, H );
        cv::perspectiveTransform(in, out, matPerspectiveTransform);
        cv::Point points[4] = {out[0],out[1],out[2],out[3]};
        cv::Point* ppt[1] = {points};
        int        nums[1] = {4};
        cv::polylines(m_Overlay, ppt, nums, 1, true, blue);
    }
    cv::addWeighted(m_Overlay,0.3f,debugOut,0.7f,0,debugOut);

    DrawLanes(debugOut, matPerspectiveTransform,3,0, imageBGR);
}

