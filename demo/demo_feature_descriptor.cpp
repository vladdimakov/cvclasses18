/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

namespace
{
void on_trackbar(int threshold, void* obj)
{
    cvlib::corner_detector_fast* detector = (cvlib::corner_detector_fast*)obj;
    detector->setThreshold(threshold);
}

void get_distances(const cv::Mat& descriptors, std::vector<double>& distances, double& maxDistance)
{
    maxDistance = 0;
    double distance;
    for (int i = 0; i < descriptors.rows; i++)
    {
        for (int j = i + 1; j < descriptors.rows; j++)
        {
            if (descriptors.type() == CV_32F)
                distance = cv::norm(descriptors.row(i) - descriptors.row(j), cv::NORM_L1);
            else if (descriptors.type() == CV_8U)
                distance = cv::norm(descriptors.row(i) - descriptors.row(j), cv::NORM_HAMMING2);

            distances.push_back(distance);

            if (distance > maxDistance)
                maxDistance = distance;
        }
    }
}

void get_bins(const cv::Mat& descriptors, cv::Mat& bins)
{
    std::vector<double> distances;
    double maxDistance;

    get_distances(descriptors, distances, maxDistance);

    bins = cv::Mat(128, 1, CV_32F, cv::Scalar(0));

    size_t index;
    for (int i = 0; i < distances.size(); i++)
    {
        index = size_t(distances[i] / maxDistance * 127);
        bins.at<float>(index) += 1;
    }
}
}; // namespace

int demo_feature_descriptor(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";
    const auto hist_a_wnd = "hist_a";
    const auto hist_b_wnd = "hist_b";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    auto detector_a = cvlib::corner_detector_fast::create();
    auto detector_b = cv::ORB::create();
    std::vector<cv::KeyPoint> corners;
    cv::Mat descriptors_a, descriptors_b;

    int threshold = 30;
    cv::createTrackbar("Threshold", demo_wnd, &threshold, 255, on_trackbar, (void*)detector_a);
    detector_a->setThreshold(threshold);

    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector_a->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));

        utils::put_fps_text(frame, fps);
        cv::putText(frame, std::to_string(corners.size()), cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        cv::imshow(demo_wnd, frame);

        detector_a->compute(frame, corners, descriptors_a);
        detector_b->compute(frame, corners, descriptors_b);

        cv::Mat bins_a, bins_b;
        get_bins(descriptors_a, bins_a);
        get_bins(descriptors_b, bins_b);

        utils::showHist(hist_a_wnd, bins_a, 512, 384);
        utils::showHist(hist_b_wnd, bins_b, 512, 384);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);
    cv::destroyWindow(hist_a_wnd);
    cv::destroyWindow(hist_b_wnd);

    return 0;
}
