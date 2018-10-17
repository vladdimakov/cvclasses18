/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

void on_trackbar(int threshold, void* obj)
{
	cvlib::motion_segmentation* mseg = (cvlib::motion_segmentation*)obj;
	mseg->setThreshold(threshold / 10.0);
}

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    cvlib::motion_segmentation* mseg = new cvlib::motion_segmentation();
    const auto main_wnd = "main";
    const auto demo_wnd = "demo";

    int threshold = 25;
    int learningRate = 10;

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    cv::createTrackbar("T (10^-1)", demo_wnd, &threshold, 100, on_trackbar, (void*)mseg);
    cv::createTrackbar("R (10^-4)", demo_wnd, &learningRate, 1000);

    cv::Mat frame;
    cv::Mat frame_mseg;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        mseg->apply(frame, frame_mseg, learningRate / 10000.0);

        if (!frame_mseg.empty())
            cv::imshow(demo_wnd, frame_mseg);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    delete mseg;

    return 0;
}
