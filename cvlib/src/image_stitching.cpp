/* Image Stitcher algorithm implementation.
 * @file
 * @date 2018-12-05
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
void Stitcher::makeStitchedImg(const cv::Mat& testImg, const std::vector<cv::KeyPoint>& testCorners, const cv::Mat& refImg,
                               const std::vector<cv::KeyPoint>& refCorners, const std::vector<std::vector<cv::DMatch>>& pairs, cv::Mat& stitchedImg)
{
    m_isStitched = false;

    calcHomography(testCorners, refCorners, pairs);

    if (!m_homography.empty())
    {
        cv::Size size = refImg.size();

        if (m_dx >= 0 && m_dy >= 0)
        {
            size.width += m_dx;
            size.height += m_dy;

            cv::warpPerspective(refImg, m_stitchedImg, m_homography, size);
            testImg.copyTo(m_stitchedImg.rowRange(0, testImg.rows).colRange(0, testImg.cols));

            m_isStitched = true;
        }
        else if (m_dx < 0 && m_dy < 0)
        {
            size.width -= m_dx;
            size.height -= m_dy;

            cv::warpPerspective(testImg, m_stitchedImg, m_homography, size);
            refImg.copyTo(m_stitchedImg.rowRange(0, refImg.rows).colRange(0, refImg.cols));

            m_isStitched = true;
        }
    }

    if (m_isStitched)
        m_stitchedImg.copyTo(stitchedImg);
    else
        refImg.copyTo(stitchedImg);
}

void Stitcher::stitch(std::vector<cv::KeyPoint> testCorners, std::vector<cv::KeyPoint>& refCorners, const cv::Mat& testDescriptors,
                      cv::Mat& refDescriptors, cv::Mat& refImg)
{
    if (m_isStitched)
    {
        if (m_dx >= 0 && m_dy >= 0)
            warpPerspective(refCorners);
        else if (m_dx < 0 && m_dy < 0)
            warpPerspective(testCorners);

        refCorners.insert(refCorners.end(), testCorners.begin(), testCorners.end());
        cv::vconcat(refDescriptors, testDescriptors, refDescriptors);

        m_stitchedImg.copyTo(refImg);
    }
}

void Stitcher::calcHomography(const std::vector<cv::KeyPoint>& testCorners, const std::vector<cv::KeyPoint>& refCorners,
                              const std::vector<std::vector<cv::DMatch>>& pairs)
{
    std::vector<cv::Point2f> refPoints, testPoints;
    for (int i = 0; i < pairs.size(); i++)
    {
        if (pairs[i].size() > 0)
        {
            refPoints.push_back(refCorners[pairs[i][0].trainIdx].pt);
            testPoints.push_back(testCorners[i].pt);
        }
    }

    if (refPoints.size() != 0)
        m_homography = cv::findHomography(refPoints, testPoints, CV_RANSAC);

    if (!m_homography.empty())
    {
        m_dx = m_homography.at<double>(0, 2);
        m_dy = m_homography.at<double>(1, 2);

        if (m_dx < 0 && m_dy < 0)
            m_homography = m_homography.inv();
    }
}

void Stitcher::warpPerspective(std::vector<cv::KeyPoint>& srcDdst)
{
    cv::Mat_<double> P(3, 1);
    for (int i = 0; i < srcDdst.size(); i++)
    {
        P(0, 0) = srcDdst[i].pt.x;
        P(1, 0) = srcDdst[i].pt.y;
        P(2, 0) = 1.0;

        P = m_homography * P;

        srcDdst[i].pt.x = P(0, 0) / P(2, 0);
        srcDdst[i].pt.y = P(1, 0) / P(2, 0);
    }
}
} // namespace cvlib
