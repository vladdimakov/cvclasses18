/* Split and merge segmentation algorithm testing.
 * @file
 * @date 2018-09-17
 * @author Vladislav Dimakov
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("empty image motion_segmentation", "[motion_segmentation]")
{
    const cv::Mat image;
    cv::Mat foregroundMask;
    cvlib::motion_segmentation mseg;
    mseg.setThreshold(2.5);
    mseg.apply(image, foregroundMask, 0.001);

    REQUIRE(foregroundMask.empty());
}

TEST_CASE("constant image motion_segmentation", "[motion_segmentation]")
{
    const cv::Mat image(100, 100, CV_8UC1, cv::Scalar(128));
    cv::Mat foregroundMask, backgroundImage;
    cvlib::motion_segmentation mseg;
    mseg.setThreshold(2.5);
    mseg.apply(image, foregroundMask, 0.001);
    mseg.getBackgroundImage(backgroundImage);

    REQUIRE(image.size() == foregroundMask.size());
    REQUIRE(image.size() == backgroundImage.size());
    REQUIRE(image.type() == foregroundMask.type());
    REQUIRE(image.type() == backgroundImage.type());
    REQUIRE(cv::countNonZero(foregroundMask) == 0);
    REQUIRE(cv::countNonZero(image - backgroundImage) == 0);
}
