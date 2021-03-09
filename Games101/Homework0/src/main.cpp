#define _USE_MATH_DEFINES
#include<cmath>

#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<iostream>


/*
* 作业描述：
* 给定一个点 P=(2,1), 将该点绕原点先逆时针旋转 45◦，再平移 (1,2), 计算出变换后点的坐标（要求用齐次坐标进行计算）。
*/

int main()
{

#if 0
    // vector definition
    Eigen::Vector3f v(1.0f,2.0f,3.0f);
    Eigen::Vector3f w(1.0f,0.0f,0.0f);
    // vector output
    std::cout << "Example of output \n";
    std::cout << v << std::endl;
    // vector add
    std::cout << "Example of add \n";
    std::cout << v + w << std::endl;
    // vector scalar multiply
    std::cout << "Example of scalar multiply \n";
    std::cout << 2.0f * v << std::endl;

    // Example of matrix
    std::cout << "Example of matrix \n";
    // matrix definition
    Eigen::Matrix3f i,j;
    i << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    std::cout << i << std::endl;
#endif

    //////////////////////////////////////////////////////////////////////////
    // 齐次坐标表示
	std::cout << "Input Point: " << std::endl;
    Eigen::Vector4f P(2.0f, 1.0f, 0.0f, 1.0f);
    std::cout << P << std::endl;

    // 大于0表示顺时针
    Eigen::AngleAxisf V1(-M_PI/4, Eigen::Vector3f(0, 0, 1));
    Eigen::Matrix3f RotateMat = V1.matrix();
    //std::cout << RotateMat << std::endl;

    /*
        R 0
        T 1
    */
	Eigen::Matrix4f mat4 = Eigen::Matrix4f::Identity();
	mat4.block(0, 0, 3, 3) = RotateMat;
    // Translate
    mat4(3, 0) = 1;
    mat4(3, 1) = 2;

    std::cout << "Transform Matrix: " << std::endl;
    std::cout << mat4 << std::endl;

    // 向量和矩阵相乘
    std::cout << "Output Point: " << std::endl;
    Eigen::Vector4f Res = P.transpose() * mat4;
    std::cout << Res << std::endl;

    return 0;
}