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
}; // namespace

int demo_corner_detector(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "origin";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    auto detector = cvlib::corner_detector_fast::create();
    std::vector<cv::KeyPoint> corners;

    int threshold = 30;
    cv::createTrackbar("Threshold", demo_wnd, &threshold, 255, on_trackbar, (void*)detector);
    detector->setThreshold(threshold);

    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));
        utils::put_fps_text(frame, fps);

        cv::putText(frame, std::to_string(corners.size()), cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

        cv::imshow(demo_wnd, frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
