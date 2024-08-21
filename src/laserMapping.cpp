#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include "IMU_Processing.hpp"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
int add_point_size = 0, kdtree_delete_counter = 0;
bool pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int scan_count = 0, publish_count = 0;
int feats_down_size = 0; // 在main()中被赋值为了 经过运动补偿和下采样之后剩下的点的数量
int NUM_MAX_ITERATIONS = 0, pcd_save_interval = -1, pcd_index = 0;

bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;             // 一帧点云中，每个点的近邻点
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
deque<double> time_buffer;                      // 雷达帧时间戳的队列，和 lidar_buffer 里面的雷达帧是一一对应的。因为lidar_buffer里面的点云帧没有时间戳，所以作者在这里又单独创建了一个存储时间戳的队列
deque<PointCloudXYZI::Ptr> lidar_buffer;        // 存储雷达点云，注意：已经转换成PCL点云的类型了
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;   // 存储imu消息的队列，deque（双端队列）类型，比vector更高效

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());  //畸变纠正后降采样的单帧点云，//! 注意，虽然写了body 但是其实是在lidar系下描述的
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); //畸变纠正后降采样的单帧点云，W系下描述

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;

esekfom::esekf kf;  // 误差卡尔曼滤波器

state_ikfom state_point;    // imu 坐标系在世界坐标系下的位姿（在main中使用，先是被赋值为前向传播的位姿，然后再被赋值为迭代优化之后的后验位姿）
Eigen::Vector3d pos_lid;    // 雷达坐标系在世界坐标系下的位置

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

/**
 * @brief: 激光点云话题的回调函数
 * @param {ConstPtr} &msg
 * @return {*}
 */
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();  // 操作 lidar_buffer 之前首先加锁
    scan_count++;       // 雷达帧计数器++
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)   // 如果当前帧的时间戳小于上一帧，显然是有问题的，抛出警告并且清空队列
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());  //这里新建了一个PCL的点云
    p_pre->process(msg, ptr);   // 通过process函数将ROS的 sensor_msgs::PointCloud2 点云转换成 PCL 点云，并进行预处理
    lidar_buffer.push_back(ptr);    // 添加到buffer中
    time_buffer.push_back(msg->header.stamp.toSec());   // 把雷达帧的时间戳也添加到队列里面去
    last_timestamp_lidar = msg->header.stamp.toSec();   // 更新 last_timestamp_lidar
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

/**
 * @brief: imu的回调函数，将接收到的 IMU 数据添加到Buffer里面
 * @param {sensor_msgs::Imu::ConstPtr} &msg_in 单个输入的imu数据
 * @return {*}
 */
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));   // 创建一个指针 用于接收收到的IMU数据

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)  // 如果手动标定了雷达和IMU之间的同步时间差的话
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu); // 否则就是使用自动标定的同步时间差

    double timestamp = msg->header.stamp.toSec();   // 这个IMU数据的时间戳

    mtx_buffer.lock();  // 操作 imu_buffer 之前首先要加锁

    if (timestamp < last_timestamp_imu) // 如果当前一次的IMU数据的时间比上一次的还小，那肯定就是有问题的
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();     // 清空 buffer
    }

    last_timestamp_imu = timestamp; // 更新上一次收到IMU时间的记录 last_timestamp_imu

    imu_buffer.push_back(msg);  // 将imu数据pushback到imu_buffer里面 imu_buffer 是一个 c++ 的队列类型
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;

/**
 * @brief: 把一帧雷达帧和该雷达帧时间范围内的所有IMU数据，一起打包到 meas 结构体里面
 * @param {MeasureGroup} &meas
 * @return {*}
 */
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())     // 首先看一看 两个Buffer里面有没有数据
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)      // ? 这个标志位有啥用？
    {
        meas.lidar = lidar_buffer.front();          // 直接把最新一帧的点云塞到 打包里面去
        meas.lidar_beg_time = time_buffer.front();  // 点云帧的起始时刻
        
        if (meas.lidar->points.size() <= 5) // time too little // 如果点云的数量太少，还要给出一个ROS的警告
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)    // 如果这一帧的持续时间 < 0.5倍的平均扫描时间，那么这一帧点云的时间戳很有可能也是有问题的
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;     // 这样的话，就直接让结束时间 = 起始时间 + 平均扫描时间
        }
        else    // 绝大多数情况会执行这里
        {
            scan_num++;     // 计数+1
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);  // 结束时间 = 起始时间 + 该帧的持续时间（也是最后一个点的curvature里面存的内容）
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;  // 更新平均扫描时间（使用平均值的递推算法）
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)  //如果当前最新的imu的时间戳 < 要打包的雷达帧结束时间，证明还没有收集足够的imu数据，break
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();     // 获取IMU队列里面最早的一个IMU消息的时间戳
    meas.imu.clear();   // 同样的，把IMU打包进去之前，先把之前的数据清空
    
    // 从最早的一个IMU开始，遍历IMU数据
    // ? 直接把所有的 (imu_time < lidar_end_time)的imu数据全都打包给了meas，第一帧时不会有雷达帧开始时间之前的imu被打包进来吗？或者有当前帧和上一帧的两帧之间的IMU数据也被打包给了当前帧？
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)  // 如果现在遍历到的这个IMU是雷达结束之后采集的，那就可以退出了（注意这个数据也是不要的，要在pushback之前break）
            break;
        meas.imu.push_back(imu_buffer.front()); // 添加到 meas 里面去
        imu_buffer.pop_front();                 // 把这个IMU数据从队列里面删除
    }

    lidar_buffer.pop_front();   // lidar_buffer 的 front 已经被打包到meas里面去了，把他删除。
    time_buffer.pop_front();    // 删除对应的时间戳的记录
    lidar_pushed = false;
    return true;
}

/**
 * @brief: 将雷达坐标系下描述的点转换到世界坐标系下描述。//! 注意，这里虽然写的是body，但是其实是雷达坐标系
 * @param {PointType} *po
 * @return {*}
 */
void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

BoxPointType LocalMap_Points;      // ikd-tree地图立方体的2个角点
bool Localmap_Initialized = false; // 局部地图是否初始化
/**
 * @brief: 
 * @return {*}
 */
void lasermap_fov_segment()
{
    cub_needrm.clear(); // 需要移除的点的范围 首先将其清空
    kdtree_delete_counter = 0;  // 需要删除的点的数量

    V3D pos_LiD = pos_lid; // 在调用这个函数之前，pos_lid已经被更新为雷达坐标系在世界坐标系下的位置
    
    // 如果是第一次，还没有初始化，就先初始化局部地图范围，以pos_LiD为中心,长宽高均为cube_len
    if (!Localmap_Initialized)
    {
        //如果是第一次进入这个函数，就要对局部地图的范围进行初始化
        for (int i = 0; i < 3; i++)
        {
            // 立方体范围的两个对角的坐标
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0; //注意，cube_len是在launch里面指定的参数，表示局部地图的长度
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    // 计算各个方向上pos_LiD与局部地图边界的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（1.5*300m）太小（当前雷达的位置已经太靠近边界了），标记需要移除need_move(FAST-LIO2论文Fig.3)
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return; //如果不需要，直接返回，不更改局部地图

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    //需要移动的距离，经验公式
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    // 遍历xyz三个方向
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            // 如果现在雷达移动到离左边界太近了，就将局部地图的范围往左移动 mov_dist
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist; // 需要移除的点的范围
            cub_needrm.push_back(tmp_boxpoints);    // 将这个方向上需要移除的点的范围记录下来
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;  // 更新 LocalMap的范围 为 New_LocalMap_Points

    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);//历史记录，实际上没有用到

    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm); // 删除指定范围内的点
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I.matrix() * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

/**
 * @brief: 根据最新估计位姿  增量添加点云到map
 * @return {*}
 */
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        // 将lidar坐标系下的点转换到世界坐标系
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i]; // 将点i的近邻点都拿出来
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType mid_point;
            // 计算点i所在体素的中心
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point); // 计算点i 距离其所在体素的中心的距离
            
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                // 如果点i的所有近邻点中，距离点i最近的那个 都不在 点i所在的体素 内。也就是说点i在一个新的体素（因为近邻点是从kdtree里面取的，也就是历史地图点。这说明历史上没有任何一个点离着点i比较近）
                // 那么点i不需要被下采样了，它应该直接被添加到kdtree里面
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            
            // 遍历点i的5个近邻点 NUM_MATCH_POINTS = 5
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)  // 如果近邻点的数量小于5直接break
                    break;
                if (calc_dist(points_near[j], mid_point) < dist) //如果近邻点到体素中心距离 < 点i到体素中心距离，也就是说该体素已经有了一个不错的点了，一个体素留一个点就可以了，就不需要再添加点i
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)   // 如果点i是需要被添加的话
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            // * 如果点i没有近邻点（不可能出现，feats_down_world里面的点都是有近邻点的）或者是开始的0.1s的点云，就直接添加
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);  // * 需要下采样的点，下采样之后被添加到ikdtree中
    ikdtree.Add_Points(PointNoNeedDownsample, false);       // * 不需要下采样的点，直接添加到ikdtree中
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher &pubLaserCloudFull_)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&laserCloudFullRes->points[i],
                             &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull_.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&feats_undistort->points[i],
                             &laserCloudWorld->points[i]);
        }

        static int scan_wait_num = 0;
        scan_wait_num++;

        if (scan_wait_num % 4 == 0)
            *pcl_wait_save += *laserCloudWorld;

        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);

    auto q_ = Eigen::Quaterniond(state_point.rot.matrix());
    out.pose.orientation.x = q_.coeffs()[0];
    out.pose.orientation.y = q_.coeffs()[1];
    out.pose.orientation.z = q_.coeffs()[2];
    out.pose.orientation.w = q_.coeffs()[3];
}

/**
 * @brief: 发布里程计和TF
 * @param {Publisher} &pubOdomAftMapped
 * @return {*}
 */
void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    // * 发布 odom
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);  // 用填充 pose 部分
    pubOdomAftMapped.publish(odomAftMapped);

    // 填充协方差部分，nav_msgs::Odometry 里只发布了xyz和rpy，所以也只要这两部分的协方差。
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3); //? 这里是不是不太对，为什么要把 P 的旋转的部分存到 covariance 的平移的部分呢？参考 https://github.com/hku-mars/FAST_LIO/issues/233
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    // * 发布 TF
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);            // 是否发布当前正在扫描的点云的topic
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);          // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true); // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，需要该变量和上一个变量同时为true才发布
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);                   // 卡尔曼滤波的最大迭代次数
    nh.param<string>("map_file_path", map_file_path, "");                    // 地图保存路径
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");         // 雷达点云topic名称
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");           // IMU的topic名称
    nh.param<bool>("common/time_sync_en", time_sync_en, false);              // 是否需要时间同步，只有当外部未进行时间同步时设为true
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5); // VoxelGrid降采样时的体素大小
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);    // 地图的局部区域的长度（FastLio2论文中有解释）
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f); // 激光雷达的最大探测范围
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);               // IMU陀螺仪的协方差
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);               // IMU加速度计的协方差
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);        // IMU陀螺仪偏置的协方差
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);        // IMU加速度计偏置的协方差
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);        // 最小距离阈值，即过滤掉0～blind范围内的点云
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA); // 激光雷达的类型
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);       // 激光雷达扫描的线数（livox avia为6线）
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);           // 采样间隔，即每隔point_filter_num个点取1个点
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false); // 是否提取特征点（FAST_LIO2默认不进行特征点提取）
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false); // 是否将点云地图保存到PCD文件
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); // 雷达相对于IMU的外参R

    cout << "Lidar_type: " << p_pre->lidar_type << endl;
    // 初始化path的header（包括时间戳和帧id），path用于保存odemetry的路径
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);   // 订阅点云消息。由于AVIA激光雷达的消息格式比较特殊，所以要单独处理
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);     // 订阅imu
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);   // 设置面点下采样栅格的大小
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);       // 设置地图点下采样栅格大小

    shared_ptr<ImuProcess> p_imu1(new ImuProcess());    // 实例化一个ImuProcess类
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT); // 外参 平移部分
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR); // 外参 旋转部分
    p_imu1->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov),
                      V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, b_acc_cov));  // 将上面获取的参数输入到imu处理类里面去

    signal(SIGINT, SigHandle); //当程序检测到signal信号（例如ctrl+c） 时  执行 SigHandle 函数
    ros::Rate rate(5000);

    while (ros::ok())
    {
        if (flg_exit)   // 收到了退出信号（例如ctrl+c） 退出
            break;
        ros::spinOnce();

        if (sync_packages(Measures)) //* 把一帧LIDAR数据和该帧时间范围内的IMU数据和打包到Measures
        {
            double t00 = omp_get_wtime();

            if (flg_first_scan)
            {
                // 如果是第一帧
                first_lidar_time = Measures.lidar_beg_time;
                p_imu1->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;   // 只有一帧是没办法计算的，所以处理到这里就结束了
            }

            // * step 1. 初始化、前向传播、反向传播、点云去畸变
            p_imu1->Process(Measures, kf, feats_undistort); // 前向传播之后的状态量存到了 kf.x_ 的成员变量中，反向传播和去畸变之后的点云存到了feats_undistort里面

            //如果feats_undistort为空 ROS_WARN
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            state_point = kf.get_x();   // 获取前向传播之后的状态
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;    // 计算雷达坐标系在世界坐标系下的位置。（state_point.pos=kf.get_x()是IMU在世界坐标系下的位置，他们之间差了个外参）

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;   // 是不是开头的0.1s之内的帧

            // * step 2. 更新localmap边界，然后降采样当前帧点云
            lasermap_fov_segment();
            
            // * step 3. 再一次对运动补偿之后的点云进行下采样。（之前在点云话题的回调函数中进行点云预处理时，有过一个粗暴的下采样）
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();

            // std::cout << "feats_down_size :" << feats_down_size << std::endl;
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");    // 如果下采样之后的点云太少了，抛出警告并且跳过这一帧
                continue;
            }

            // * step 4. 初始化ikdtree(如果ikdtree为空)
            if (ikdtree.Root_Node == nullptr)
            {
                ikdtree.set_downsample_param(filter_size_map_min); //ikdtree构建时可以同时进行下采样，这里设置下采样的大小参数为filter_size_map_min
                
                // 将lidar坐标系下描述的点云feats_down_body转到世界坐标系下描述，存到feats_down_world里
                feats_down_world->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // 将点云从lidar坐标系转到世界坐标系下描述
                }
                
                ikdtree.Build(feats_down_world->points); // 用转换到世界坐标系下的点云构建ikdtree
                continue;
            }

            // 是否要查看全局地图，其实也就是ikdtree里面的所有的点。
            if (0) // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
                // std::cout << "ikdtree size: " << featsFromMap->points.size() << std::endl;
            }

            // * step 5. 迭代更新状态估计
            Nearest_Points.resize(feats_down_size); // Nearest_Points存储了这一帧中每个点的近邻点的集合，所以先 resize 成这一帧点云的维度
            //ieskf迭代更新
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);

            state_point = kf.get_x();   // 更新为迭代优化之后的后验位姿
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            // * step 6. 发布里程计和TF
            publish_odometry(pubOdomAftMapped); // 发布里程计和TF

            // * step 7. 地图更新 把这一帧的点云特征加入到ikdtree里面
            feats_down_world->resize(feats_down_size);
            map_incremental();  //地图更新

            // * step 8. 发布
            if (path_en)
                publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)
                publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);
            // publish_map(pubLaserCloudMap);

            double t11 = omp_get_wtime();
            std::cout << "feats_down_size: " << feats_down_size << "  Whole mapping time(ms):  " << (t11 - t00) * 1000 << std::endl
                      << std::endl;
        }

        rate.sleep();
    }

    // * step 9. 保存地图
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 1; i <= pcd_index; i++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(i) + string(".pcd"));
            pcl::PCDReader reader;
            reader.read(all_points_dir, *cloud_temp);
            *cloud = *cloud + *cloud_temp;
        }

        string file_name = string("GlobalMap.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *cloud);

        //////////////////////////////////////
        PointVector().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        featsFromMap->clear();
        featsFromMap->points = ikdtree.PCL_Storage;
        std::cout << "ikdtree size: " << featsFromMap->points.size() << std::endl;
        string file_name1 = string("GlobalMap_ikdtree.pcd");
        pcl::PCDWriter pcd_writer1;
        string all_points_dir1(string(string(ROOT_DIR) + "PCD/") + file_name1);
        cout << "current scan saved to /PCD/" << file_name1 << endl;
        pcd_writer1.writeBinary(all_points_dir1, *featsFromMap);
    }

    return 0;
}
