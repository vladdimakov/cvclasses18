/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::setThreshold(int threshold)
{
    m_threshold = threshold;
}

inline bool corner_detector_fast::isCorner(const cv::Point2i& point, int step, int pointNumThreshold)
{
    int points[2] = {0, 0}, j = 0;

    for (int i = 0; i < 16; i += step)
        if (cv::abs(m_image.at<uchar>(point + m_template[i]) - m_image.at<uchar>(point)) >= m_threshold)
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
        image.copyTo(m_image);
    else if (image.channels() == 3)
        cv::cvtColor(image, m_image, cv::COLOR_BGR2GRAY);

    cv::Point2i point;

    for (point.y = 3; point.y < m_image.rows - 3; point.y++)
        for (point.x = 3; point.x < m_image.cols - 3; point.x++)
            if (isCorner(point, 4, 3) && isCorner(point, 1, 9))
                keypoints.push_back(cv::KeyPoint(point, 1.f));
}
} // namespace cvlib
