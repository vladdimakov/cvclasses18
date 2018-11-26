/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
corner_detector_fast::corner_detector_fast()
{
    m_testAreaSize = 31;
    m_testPointsNum = 256;
    m_descriptorBytesNum = m_testPointsNum / 8;
    m_sigma = m_testAreaSize / 5.0;

    generateTestPoints();
}

void corner_detector_fast::generateTestPoints()
{
    cv::RNG rng;
    cv::Point2i point1, point2;

    for (int i = 0; i < m_testPointsNum; i++)
    {
        point1.x = cvRound(rng.gaussian(m_sigma));
        point1.y = cvRound(rng.gaussian(m_sigma));
        point2.x = cvRound(rng.gaussian(m_sigma));
        point2.y = cvRound(rng.gaussian(m_sigma));

        m_testPoints.push_back(std::make_pair(point1, point2));
    }
}

// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::setThreshold(int threshold)
{
    m_threshold = threshold;
}

bool corner_detector_fast::isCorner(const cv::Point2i& point, int step, int pointNumThreshold)
{
    int points[2] = {0, 0}, j = 0;

    for (int i = 0; i < 16; i += step)
        if (cv::abs(m_imageForDetector.at<uchar>(point + m_template[i]) - m_imageForDetector.at<uchar>(point)) >= m_threshold)
            points[j]++;
        else if (points[j] >= pointNumThreshold)
            return true;
        else
            points[j = 1] = 0;

    return points[0] + points[1] >= pointNumThreshold;
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();

    if (image.empty())
        return;
    else if (image.channels() == 1)
        image.copyTo(m_imageForDetector);
    else if (image.channels() == 3)
        cv::cvtColor(image, m_imageForDetector, cv::COLOR_BGR2GRAY);

    cv::Point2i point;
    for (point.y = 3; point.y < m_imageForDetector.rows - 3; point.y++)
        for (point.x = 3; point.x < m_imageForDetector.cols - 3; point.x++)
            if (isCorner(point, 4, 3) && isCorner(point, 1, 9))
                keypoints.push_back(cv::KeyPoint(point, 2, 0, 0, 0, 1));
}

void corner_detector_fast::calcDescriptor(const cv::Point2i& keypoint, cv::Mat& descriptor)
{
    cv::Point2i point1, point2;
    bool testResult;

    for (int byteNum = 0; byteNum < m_descriptorBytesNum; byteNum++)
    {
        descriptor.at<uchar>(byteNum) = 0;
        for (int bitNum = 0; bitNum < 8; bitNum++)
        {
            point1 = keypoint + m_testPoints[byteNum * 8 + bitNum].first;
            point2 = keypoint + m_testPoints[byteNum * 8 + bitNum].second;

            testResult = m_imageForDescriptor.at<uchar>(point1) < m_imageForDescriptor.at<uchar>(point2);
            descriptor.at<uchar>(byteNum) |= (testResult << bitNum);
        }
    }
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    if (image.empty())
        return;
    else if (image.channels() == 1)
        image.copyTo(m_imageForDescriptor);
    else if (image.channels() == 3)
        cv::cvtColor(image, m_imageForDescriptor, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(m_imageForDescriptor, m_imageForDescriptor, cv::Size(9, 9), 2.0, 2.0, cv::BORDER_REPLICATE);

    cv::copyMakeBorder(m_imageForDescriptor, m_imageForDescriptor, m_testAreaSize, m_testAreaSize, m_testAreaSize, m_testAreaSize, cv::BORDER_REPLICATE);

    cv::Mat descriptorsMat(keypoints.size(), m_descriptorBytesNum, CV_8U, cv::Scalar(0));

    for (int i = 0; i < keypoints.size(); i++)
        calcDescriptor(cv::Point2i(keypoints[i].pt) + cv::Point2i(m_testAreaSize, m_testAreaSize), descriptorsMat.row(i));

    descriptors.assign(descriptorsMat);
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray /*mask = cv::noArray()*/, std::vector<cv::KeyPoint>& keypoints,
                                            cv::OutputArray descriptors, bool /*= false*/)
{
    detect(image, keypoints);
    compute(image, keypoints, descriptors);
}

cv::String corner_detector_fast::getDefaultName() const
{
    return "FAST_Binary";
}
} // namespace cvlib
