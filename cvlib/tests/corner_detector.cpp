/* FAST corner detector algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("empty image corner_detector_fast", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    cv::Mat image;
    std::vector<cv::KeyPoint> out;
    fast->setThreshold(1);
    fast->detect(image, out);
    REQUIRE(out.empty());
}

TEST_CASE("constant image corner_detector_fast", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    cv::Mat image(7, 7, CV_8U, cv::Scalar(100));
    std::vector<cv::KeyPoint> out;
    fast->setThreshold(1);
    fast->detect(image, out);
    REQUIRE(out.empty());
}

TEST_CASE("complex images corner_detector_fast", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    cv::Mat image(7, 7, CV_8U);
    std::vector<cv::KeyPoint> out;
    fast->setThreshold(1);
    const cv::Point2i points[16] = {cv::Point2i(3, 0), cv::Point2i(4, 0), cv::Point2i(5, 1), cv::Point2i(6, 2), cv::Point2i(6, 3), cv::Point2i(6, 4),
                                    cv::Point2i(5, 5), cv::Point2i(4, 6), cv::Point2i(3, 6), cv::Point2i(2, 6), cv::Point2i(1, 5), cv::Point2i(0, 4),
                                    cv::Point2i(0, 3), cv::Point2i(0, 2), cv::Point2i(1, 1), cv::Point2i(2, 0)};
    SECTION("4 points")
    {
        image.setTo(cv::Scalar(0));
        for (int i = 0; i < 16; i += 4)
            image.at<uchar>(points[i]) = 255;

        fast->detect(image, out);
        REQUIRE(out.size() == 0);
    }

    SECTION("8 points")
    {
        image.setTo(cv::Scalar(0));
        for (int i = 0; i < 16; i += 2)
            image.at<uchar>(points[i]) = 255;

        fast->detect(image, out);
        REQUIRE(out.size() == 0);
    }

    SECTION("9 points")
    {
        for (int i = 0; i < 16; i++)
        {
            image.setTo(cv::Scalar(0));
            for (int j = 0; j < 9; j++)
                image.at<uchar>(points[(i + j) % 16]) = 255;

            fast->detect(image, out);

            if (i == 0 || i == 4 || i == 8 || i == 12)
                REQUIRE(out.size() == 1);
            else
                REQUIRE(out.size() == 0);
        }
    }

    SECTION("10 points")
    {
        image.setTo(cv::Scalar(0));
        for (int i = 0; i < 9; i++)
            image.at<uchar>(points[i]) = 255;

        image.at<uchar>(points[15]) = 255;

        fast->detect(image, out);
        REQUIRE(out.size() == 1);

        image.at<uchar>(points[1]) = 0;
        image.at<uchar>(points[14]) = 255;

        fast->detect(image, out);
        REQUIRE(out.size() == 0);
    }

    SECTION("11 points")
    {
        image.setTo(cv::Scalar(0));
        for (int i = 0; i < 8; i++)
            image.at<uchar>(points[i]) = 255;

        image.at<uchar>(points[12]) = 255;
        image.at<uchar>(points[14]) = 255;
        image.at<uchar>(points[15]) = 255;

        fast->detect(image, out);
        REQUIRE(out.size() == 1);

        image.at<uchar>(points[14]) = 0;
        image.at<uchar>(points[15]) = 0;

        fast->detect(image, out);
        REQUIRE(out.size() == 0);
    }
}
