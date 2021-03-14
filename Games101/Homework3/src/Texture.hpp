//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
		auto u_img = u * width;
		auto v_img = (1 - v) * height;
		int u_i = std::min(width - 1, (int)u_img);
		u_i = std::max(0, u_i);
		int v_i = std::min(height - 1, (int)v_img);
		v_i = std::max(0, v_i);
		auto color = image_data.at<cv::Vec3b>(v_i, u_i);
		return Eigen::Vector3f(color[0], color[1], color[2]);
    }

	Eigen::Vector3f getColor2(float u, float v)
	{
		int u_i = std::min(width - 1, (int)u);
		u_i = std::max(0, u_i);
		int v_i = std::min(height - 1, (int)v);
		v_i = std::max(0, v_i);
		auto color = image_data.at<cv::Vec3b>(v_i, u_i);
		return Eigen::Vector3f(color[0], color[1], color[2]);
	}

	Eigen::Vector3f getColorBilinear(float u, float v)
	{
		auto u_img = u * width;
		auto v_img = (1 - v) * height;

		Eigen::Vector2f u00 = { std::floor(u_img),std::floor(v_img) };
		Eigen::Vector2f u01 = { std::ceil(u_img),std::floor(v_img) };
		Eigen::Vector2f u10 = { std::floor(u_img),std::ceil(v_img) };
		Eigen::Vector2f u11 = { std::ceil(u_img),std::ceil(v_img) };

		float s = u_img - std::floor(u_img);
		float t = v_img - std::floor(v_img);

		Eigen::Vector3f tmp0 = getColor2(u00.x(), u00.y()) + s * (getColor2(u01.x(), u01.y()) - getColor2(u00.x(), u00.y()));
		Eigen::Vector3f tmp1 = getColor2(u10.x(), u10.y()) + s * (getColor2(u11.x(), u11.y()) - getColor2(u10.x(), u10.y()));

		return tmp0 + t * (tmp1 - tmp0);
	}


};
#endif //RASTERIZER_TEXTURE_H
