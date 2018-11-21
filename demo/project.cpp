/* Demo application for Course Project
 * @file
 * @date 2018-11-21
 * @author Anonymous
 */

#include "utils.hpp"
#include <cvlib.hpp>

int project_demo(int argc, char* argv[])
{
    const cv::String keys = "{video          |      | video file           }";
    cv::CommandLineParser parser(argc, argv, keys);
    auto videoFile = parser.get<cv::String>("video");

    const bool showAuxImages = false;
    const auto origin_wnd = "Origin";
    const auto deviation_image_wnd = "Deviation image";
    const auto background_image_wnd = "Background image";
    const auto binary_image_wnd = "Binary image";

    const auto wnd_width = 640;
    const auto wnd_height = 480;

    cv::namedWindow(origin_wnd, CV_WINDOW_NORMAL);
    cv::namedWindow(binary_image_wnd, CV_WINDOW_NORMAL);
    cv::resizeWindow(origin_wnd, wnd_width, wnd_height);
    cv::resizeWindow(binary_image_wnd, wnd_width, wnd_height);

    if (showAuxImages)
    {
        cv::namedWindow(deviation_image_wnd, CV_WINDOW_NORMAL);
        cv::namedWindow(background_image_wnd, CV_WINDOW_NORMAL);
        cv::resizeWindow(deviation_image_wnd, wnd_width, wnd_height);
        cv::resizeWindow(background_image_wnd, wnd_width, wnd_height);
    }

    float refreshRate = 0.02f; // TODO
    float deviationFactor = 5.5f; // TODO
    float targetsFactor = 15.0f; // TODO
    int maxCornersNum = 64;
    int minCornersNum = 16;
    cvlib::AdvancedMotionSegmentation segmenter(refreshRate, deviationFactor, targetsFactor, maxCornersNum, minCornersNum);

    cv::Mat frame, deviationImage, backgroundImage, binaryImage;

    cv::VideoCapture cap(videoFile);
    if (!cap.isOpened())
        return -1;

    utils::fps_counter fps;
    while (true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        segmenter.process(frame);
        segmenter.getBinaryImage(binaryImage);

        cv::line(frame, cv::Point(frame.cols / 2, 200), cv::Point(frame.cols / 2, frame.rows - 200), cv::Scalar(0, 0, 255), 2, 8);
        utils::put_fps_text(frame, fps);

        cv::imshow(origin_wnd, frame);
        cv::imshow(binary_image_wnd, binaryImage);

        if (showAuxImages)
        {
            segmenter.getDeviationImage(deviationImage);
            segmenter.getBackgroundImage(backgroundImage);
            cv::imshow(deviation_image_wnd, deviationImage);
            cv::imshow(background_image_wnd, backgroundImage);
        }

        char key = (char)cv::waitKey(1);
        if (key == 27)
        {
            break;
        }
        else if (key == ' ')
        {
            segmenter.setNeedToInit(true);
            continue;
        }
    }

    cv::destroyWindow(origin_wnd);
    cv::destroyWindow(binary_image_wnd);

    if (showAuxImages)
    {
        cv::destroyWindow(deviation_image_wnd);
        cv::destroyWindow(background_image_wnd);
    }

    return 0;
}
