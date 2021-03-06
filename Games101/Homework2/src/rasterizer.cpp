// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(int x, int y, const Vector3f* _v)
{
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Eigen::Vector3f A = _v[0];
    Eigen::Vector3f B = _v[1];
    Eigen::Vector3f C = _v[2];

    Eigen::Vector3f P(x, y, 1);

    Eigen::Vector3f AP = P - A;
    Eigen::Vector3f BP = P - B;
    Eigen::Vector3f CP = P - C;

    Eigen::Vector3f AB = B - A;
	Eigen::Vector3f BA = -AB;
	Eigen::Vector3f AC = C - A;
    Eigen::Vector3f CA = -AC;
	Eigen::Vector3f BC = C - B;
    Eigen::Vector3f CB = -BC;

    // 通过叉乘来判断是否位于三角形内部，实际计算量大于求重心坐标
    if (AB.cross(AP).z() * AP.cross(AC).z() < 0 || BC.cross(BP).z() * BP.cross(BA).z() < 0 || CA.cross(CP).z() * CP.cross(CB).z()<0)
    {
        return false;
    }

    return true;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };

        //Homogeneous division
        //透视除法
        for (auto& vec : v) {
            // vec.w() 就是view空间的深度z，
            vec /= vec.w();
            float z = vec.z();
        }
        //Viewport transformation
        //屏幕映射
        for (auto & vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }

    // resolve msaa
    // http://games-cn.org/forums/topic/zuoye2guanyu-ssaa/
    if (bUseMSAA)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                Vector3f res = Vector3f{ 0, 0, 0 };
                for (int i = 0; i < 4; i++)
                {
                    int msaa_index = get_msaa_index(x, y, i);
                    res += frame_buf_msaa[msaa_index];
                }
                res /= 4;
                frame_buf[get_index(x, y)] = res;
            }
        }
    }

}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    
	// If so, use the following code to get the interpolated z value.
	//auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
	//float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
	//float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
	//z_interpolated *= w_reciprocal;

	// TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.

    int MinX = std::min(std::min(v[0].x(), v[1].x()), v[2].x());
    MinX = std::max(0, MinX);
    int MinY = std::min(std::min(v[0].y(), v[1].y()), v[2].y());
    MinY = std::max(0, MinY);
    int MaxX = std::max(std::max(v[0].x(), v[1].x()), v[2].x()) + 0.5;
    MaxX = std::min(width - 1, MaxX);
    int MaxY = std::max(std::max(v[0].y(), v[1].y()), v[2].y()) + 0.5;
    MaxY = std::min(height - 1, MaxY);

    for (int y = MinY; y <= MaxY; ++y)
    {
        for (int x = MinX; x <= MaxX; ++x)
        {
            if (bUseMSAA)
            {
                int count = 0;
                std::vector<Vector2f> offset{
                    Vector2f(0.25,0.25),
                    Vector2f(0.25,0.75),
                    Vector2f(0.75,0.25),
                    Vector2f(0.75,0.75)
                };

                for (int i = 0; i < 4; ++i)
                {
                    float _x = x + offset[i].x();
                    float _y = y + offset[i].y();

					auto BarycentricTexCoord = computeBarycentric2D(_x, _y, t.v);
					float alpha = std::get<0>(BarycentricTexCoord);
					float beta = std::get<1>(BarycentricTexCoord);
					float gamma = std::get<2>(BarycentricTexCoord);

                    if (alpha > 0 && beta > 0 && gamma > 0)
                    {
                        int msaa_index = get_msaa_index(x, y, i);

                        float& real_z = depth_buf_msaa[msaa_index];

						float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
						float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
						z_interpolated *= w_reciprocal;

                        // 深度测试
						if (z_interpolated < real_z)
						{
                            real_z = z_interpolated;
                            // 写到msaa的帧缓冲中，而不是写到framebuffer中
                            frame_buf_msaa[msaa_index] = t.getColor();
						}
                    }
                }
            }
            else
            {
				//bool bInTriangle = insideTriangle(x, y, t.v);
                //if (!bInTriangle) continue;
                auto BarycentricTexCoord = computeBarycentric2D(x, y, t.v);
                float alpha = std::get<0>(BarycentricTexCoord);
                float beta = std::get<1>(BarycentricTexCoord);
                float gamma = std::get<2>(BarycentricTexCoord);

                if (alpha > 0 && beta > 0 && gamma > 0)
                {
			        // 深度插值 投射校正
                    // https://zhuanlan.zhihu.com/p/144331875
                    // https://zhuanlan.zhihu.com/p/105639446
                    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    // 深度测试
                    float& real_z = depth_buf[get_index(x,y)];
                    if (z_interpolated < real_z)
                    {
                        real_z = z_interpolated;
				        Eigen::Vector3f point = Eigen::Vector3f(x, y, 1.0f);
				        set_pixel(point, t.getColor());
                    }
                }
			}
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(frame_buf_msaa.begin(), frame_buf_msaa.end(), Eigen::Vector3f{ 0, 0, 0 });
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(depth_buf_msaa.begin(), depth_buf_msaa.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    frame_buf_msaa.resize(w * h * MSAA_FACTOR);
    depth_buf_msaa.resize(w * h * MSAA_FACTOR);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}


int rst::rasterizer::get_msaa_index(int x, int y, int z)
{
    return ((height - 1 - y) * width + x) * MSAA_FACTOR + z;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on