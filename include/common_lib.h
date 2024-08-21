#ifndef COMMON_LIB_H1
#define COMMON_LIB_H1

#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sfast_lio/Pose6D.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>

using namespace std;
using namespace Eigen;

#define PI_M (3.14159265358)
#define G_m_s2 (9.81)               // Gravaty const in GuangDong/China
#define NUM_MATCH_POINTS    (5)     

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]                                     // 把3*1向量v拆散
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]       // 把3*3矩阵v打平 拆散
#define SKEW_SYM_MATRX(v)        0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0       // 求向量v的反对称矩阵
#define DEBUG_FILE_DIR(name)     (string(string(ROOT_DIR) + "Log/"+ name))

typedef sfast_lio::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

struct MeasureGroup     // Lidar data and imu dates for the current process
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;  // 该雷达帧的开始时间
    double lidar_end_time;  // 结束时间
    PointCloudXYZI::Ptr lidar;  // 点云
    deque<sensor_msgs::Imu::ConstPtr> imu;  // 该帧时间范围内的 imu数据 队列
};

template<typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g, \
                const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
    }
    return move(rot_kp);
}


float calc_dist(PointType p1, PointType p2){
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

/**
 * @brief: 拟合平面方程 求 A/Dx + B/Dy + C/Dz + 1 = 0 的参数 
 * @param {Matrix<T, 4, 1>} &pca_result 拟合结果
 * @param {PointVector} &point
 * @param {const T} &threshold 阈值
 * @return {*}
 */
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Matrix<T, NUM_MATCH_POINTS, 3> A;   // 5*3
    Matrix<T, NUM_MATCH_POINTS, 1> b;   // 5*1
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    // * A = [[x1,y1,z1], [x2,y2,z2],...,[x5,y5,z5]]
    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b); // 使用 QR 分解 求解超定方程Ax=b 结果是平面方程的系数 A/D  B/D C/D

    T n = normvec.norm();
    //pca_result是平面方程的4个参数  /n 将平面方程化成标准形式，即ax+by+cz+d=0，其中[a,b,c]是单位法向量
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    // 如果几个近邻点中，出现了任意一个到拟合平面的距离 > threshold的点 认为这几个近邻点拟合出来的平面不好（或者说它们不近似在一个平面上） 返回false
    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold) // 点到平面的距离大于阈值
        {
            return false;
        }
    }
    return true;
}

#endif