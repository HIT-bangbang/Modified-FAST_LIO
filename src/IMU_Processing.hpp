#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>

#include "use-ikfom.hpp"
#include "esekfom.hpp"

/*
这个hpp主要包含：
IMU数据预处理：IMU初始化，IMU正向传播，反向传播补偿运动失真   
*/

#define MAX_INI_COUNT (10)  //最大迭代次数
//判断点的时间先后顺序(注意curvature中存储的是时间戳)
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void set_param(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias);
  Eigen::Matrix<double, 12, 12> Q;    //噪声协方差矩阵  对应论文式(8)中的Q
  void Process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &pcl_un_);

  V3D cov_acc;             //加速度协方差
  V3D cov_gyr;             //角速度协方差
  V3D cov_acc_scale;       //外部传入的 初始加速度协方差
  V3D cov_gyr_scale;       //外部传入的 初始角速度协方差
  V3D cov_bias_gyr;        //角速度bias的协方差
  V3D cov_bias_acc;        //加速度bias的协方差
  double first_lidar_time; //当前帧第一个点云时间

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;        //当前帧点云未去畸变
  sensor_msgs::ImuConstPtr last_imu_;     // 上一帧imu
  vector<Pose6D> IMUpose;                 // 存储imu位姿(反向传播用)
  M3D Lidar_R_wrt_IMU;                    // lidar到IMU的旋转外参
  V3D Lidar_T_wrt_IMU;                    // lidar到IMU的平移外参
  V3D mean_acc;                           //加速度均值,用于计算方差
  V3D mean_gyr;                           //角速度均值，用于计算方差
  V3D angvel_last;                        //上一帧角速度
  V3D acc_s_last;                         //上一帧加速度
  double start_timestamp_;                //开始时间戳
  double last_lidar_end_time_;            //上一帧结束时间戳
  int init_iter_num = 1;                  //初始化迭代次数
  bool b_first_frame_ = true;             //是否是第一帧
  bool imu_need_init_ = true;             //是否需要初始化imu
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;                          //初始化迭代次数
  Q = process_noise_cov();                    //调用use-ikfom.hpp里面的process_noise_cov初始化噪声协方差
  cov_acc = V3D(0.1, 0.1, 0.1);               //加速度协方差初始化
  cov_gyr = V3D(0.1, 0.1, 0.1);               //角速度协方差初始化
  cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001); //角速度bias协方差初始化
  cov_bias_acc = V3D(0.0001, 0.0001, 0.0001); //加速度bias协方差初始化
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;                       //上一帧角速度初始化
  Lidar_T_wrt_IMU = Zero3d;                   // lidar到IMU的位置外参初始化
  Lidar_R_wrt_IMU = Eye3d;                    // lidar到IMU的旋转外参初始化
  last_imu_.reset(new sensor_msgs::Imu());    //上一帧imu初始化
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()   //重置参数
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init_ = true;                   //是否需要初始化imu
  start_timestamp_ = -1;                   //开始时间戳
  init_iter_num = 1;                       //初始化迭代次数
  IMUpose.clear();                         // imu位姿清空
  last_imu_.reset(new sensor_msgs::Imu()); //上一帧imu初始化
  cur_pcl_un_.reset(new PointCloudXYZI()); //当前帧点云未去畸变初始化
}

//传入外部参数
void ImuProcess::set_param(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias)  
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
  cov_gyr_scale = gyr;
  cov_acc_scale = acc;
  cov_bias_gyr = gyr_bias;
  cov_bias_acc = acc_bias;
}


//IMU初始化：利用开始的IMU帧的平均值初始化状态量x
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N)
{
  //MeasureGroup这个struct表示当前过程中正在处理的所有数据，包含IMU队列和一帧lidar的点云 以及lidar的起始和结束时间
  //初始化重力、陀螺仪偏差、acc和陀螺仪协方差  将加速度测量值归一化为单位重力   **/
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_) //如果为第一帧IMU
  {
    Reset();    //重置IMU参数
    N = 1;      //将迭代次数置1
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;    //IMU初始时刻的加速度
    const auto &gyr_acc = meas.imu.front()->angular_velocity;       //IMU初始时刻的角速度
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;              //第一帧加速度值作为初始化均值
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;              //第一帧角速度值作为初始化均值
    first_lidar_time = meas.lidar_beg_time;                   //将当前IMU帧对应的lidar起始时间 作为初始时间
  }

  for (const auto &imu : meas.imu)    //根据所有IMU数据，计算平均值和方差
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc  += (cur_acc - mean_acc) / N;    //根据当前帧和均值差作为均值的更新
    mean_gyr  += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc)  / N;
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr)  / N / N * (N-1);

    N ++;
  }
  
  state_ikfom init_state = kf_state.get_x();        //在esekfom.hpp获得x_的状态
  init_state.grav = - mean_acc / mean_acc.norm() * G_m_s2;    //得平均测量的单位方向向量 * 重力加速度预设值。这里的处理是为了适应不同厂家的IMU可能重力不是9.81
  
  // 设置初始的state
  //这里并没有初始化ba，直接默认ba初始时是等于0的
  init_state.bg  = mean_gyr;      //角速度测量作为陀螺仪偏差
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;      //将lidar和imu外参传入
  init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);
  kf_state.change_x(init_state);      //将初始化后的状态传入esekfom.hpp中的x_

  // 设置协方差的初始值
  Matrix<double, 24, 24> init_P = MatrixXd::Identity(24,24);      //在esekfom.hpp获得P_的协方差矩阵
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = init_P(23,23) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back(); //记录最后一个IMU数据

  // std::cout << "IMU init new -- init_state  " << init_state.pos  <<" " << init_state.bg <<" " << init_state.ba <<" " << init_state.grav << std::endl;
}

//前向传播、反向传播、去畸变
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_out)
{
  /***将上一帧尾部最后一个imu添加到当前帧头部 ***/
  auto v_imu = meas.imu;         //取出当前帧的IMU队列
  v_imu.push_front(last_imu_);   //将上一帧最后尾部的imu添加到当前帧头部的imu
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();    //拿到当前帧尾部的imu的时间
  const double &pcl_beg_time = meas.lidar_beg_time;      // 点云开始和结束的时间戳
  const double &pcl_end_time = meas.lidar_end_time;
  
  // 根据点云中每个点的时间戳对点云进行重排序
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);  //这里curvature中存放了时间戳（在preprocess.cpp中）


  state_ikfom imu_state = kf_state.get_x();  // 获取上一次KF估计的后验状态作为本次IMU预测的初始状态
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
  //将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵

  /*** 前向传播 ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu; // angvel_avr为平均角速度，acc_avr为平均加速度，acc_imu为imu加速度，vel_imu为imu速度，pos_imu为imu位置
  M3D R_imu;    //IMU旋转矩阵 消除运动失真的时候用

  double dt = 0;

  input_ikfom in;
  // 遍历当前雷达帧时间内的所有IMU测量并且进行积分，离散中值法 前向传播
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);        //第it_imu个imu数据
    auto &&tail = *(it_imu + 1);    //第it_imu+1个的imu数据
    //判断时间先后顺序：如果下一个IMU的时间戳小于上一个IMU结束时间戳，显然发生了错误，直接continue
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),      // 中值积分
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr  = acc_avr * G_m_s2 / mean_acc.norm(); //通过重力数值对加速度进行调整(除上初始化的IMU大小*9.8)

    //会有一次IMU数据开始时刻早于上次雷达帧结束时刻，单独处理一下(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_; //从上次雷达时刻末尾开始传播 计算与tail之间的时间差
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();     //两个IMU时刻之间的时间间隔
    }
    
    in.acc = acc_avr;     // 两帧IMU的中值作为输入in  用于前向传播
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;         // 配置协方差矩阵
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

    kf_state.predict(dt, Q, in);    // IMU前向传播，每次传播的时间间隔为dt

    imu_state = kf_state.get_x();   //更新IMU状态为前向传播后的状态
    
    //更新angvel_last = tail角速度-前向传播更新后的bias  
    angvel_last = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
    
    //更新世界坐标系下的加速度acc_s_last = R*(加速度-bias) - g
    acc_s_last  = V3D(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z) * G_m_s2 / mean_acc.norm();   
    // std::cout << "acc_s_last: " << acc_s_last.transpose() << std::endl;
    // std::cout << "imu_state.ba: " << imu_state.ba.transpose() << std::endl;
    // std::cout << "imu_state.grav: " << imu_state.grav.transpose() << std::endl;
    acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba) + imu_state.grav;
    // std::cout << "--acc_s_last: " << acc_s_last.transpose() << std::endl<< std::endl;

    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;    //IMU时刻和这一雷达帧开始时刻的时间间隔，将会存到每个IMUpose的offset_time里
    //将这一帧雷达点云时间内的所有的IMU的状态记录下来、包括与雷达帧开始时刻的时间间隔、加速度、角速度、速度、位置、旋转
    IMUpose.push_back( set_pose6d( offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix() ) );
  }

  // 前面的for循环里面没处理最后一个IMU数据到雷达帧末尾的传播，这里补上单独处理一下
  dt = abs(pcl_end_time - imu_end_time); //dt就是从最后一个IMU数据时间戳到该雷达帧结束时刻的时间戳
  kf_state.predict(dt, Q, in);  //前向传播
  imu_state = kf_state.get_x(); //更新imu_state
  last_imu_ = meas.imu.back();              //保存最后一个IMU测量，以便于下一帧使用
  last_lidar_end_time_ = pcl_end_time;      //保存这一帧最后一个雷达测量的结束时间，以便于下一帧使用

   /***消除每个激光雷达点的失真（反向传播）***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return; // 检查点云是空的，直接返回
  auto it_pcl = pcl_out.points.end() - 1;

  //遍历雷达帧覆盖时间内的每个IMU，注意这里是从最晚的一个开始，往前遍历
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);   //head IMU旋转矩阵
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);     //head IMU速度
    pos_imu<<VEC_FROM_ARRAY(head->pos);     //head IMU位置
    acc_imu<<VEC_FROM_ARRAY(tail->acc);     //tail IMU加速度
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);  //tail IMU角速度

    //之前点云按照时间从小到大排序过，IMUpose也同样是按照时间从小到大push进入的
    //此时从IMUpose的末尾开始循环，也就是从时间最大处开始，因此只需要判断 点云时间需>IMU head时刻即可，不需要判断点云时间<IMU tail
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;    //点和head IMU时刻之间的时间间隔。curvature是点到雷达帧开始时刻的时间差。offset_time是该IMU_pose到这一帧雷达帧开始时刻的时间差

      /*    P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)    */

      M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt).matrix() );   //该点时间戳下IMU坐标系相对于世界坐标系的旋转：head IMU旋转矩阵 * exp(tail角速度*dt)   
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);   //点在自身时间戳时雷达坐标系下的位置()
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);   //该点时间戳下IMU坐标系的世界位置 - 雷达帧结束时刻的IMU世界位置
      // 畸变矫正之后的点的位置（在雷达帧结束时刻的雷达坐标系中描述）
      V3D P_compensate = imu_state.offset_R_L_I.matrix().transpose() * (imu_state.rot.matrix().transpose() * (R_i * (imu_state.offset_R_L_I.matrix() * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}


double T1,T2;
void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &cur_pcl_un_)
{
  // T1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  // 如果IMU 初始化还没有完成，那么这一帧的IMU数据继续拿来用作初始化
  if (imu_need_init_)   
  {
    // The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);  //如果开头几帧  需要初始化IMU参数

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();

    // IMU_init()里面迭代次数比 MAX_INI_COUNT 多的话，就可以结束初始化了
    if (init_iter_num > MAX_INI_COUNT)
    {
      //what?这里写了些啥？？IMU_init()初始化的cov_acc白算了？
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
    }

    return;
  }

  // 初始化完成后，进行前向传播和反向传播点云去畸变，去畸变之后的点云存放在cur_pcl_un_中。
  UndistortPcl(meas, kf_state, *cur_pcl_un_); 

  // T2 = omp_get_wtime();
  // cout<<"[ IMU Process ]: Time: "<<T2 - T1<<endl;
}
