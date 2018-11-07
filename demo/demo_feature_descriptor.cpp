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

int demo_feature_descriptor(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    auto detector_a = cvlib::corner_detector_fast::create();
    auto detector_b = cv::KAZE::create();
    std::vector<cv::KeyPoint> corners;
    cv::Mat descriptors;

    int threshold = 30;
    cv::createTrackbar("Threshold", demo_wnd, &threshold, 255, on_trackbar, (void*)detector_a);
    detector_a->setThreshold(threshold);

    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector_a->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));

        utils::put_fps_text(frame, fps);
        cv::putText(frame, std::to_string(corners.size()), cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        cv::imshow(demo_wnd, frame);

        pressed_key = cv::waitKey(30);
        /*if (pressed_key == ' ') // space
        {
            cv::FileStorage file("descriptor.json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

            detector_a->compute(frame, corners, descriptors);
            file << detector_a->getDefaultName() << descriptors;

            detector_b->compute(frame, corners, descriptors);
            file << "detector_b" << descriptors;

            std::cout << "Dump descriptors complete! \n";
        }*/
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
