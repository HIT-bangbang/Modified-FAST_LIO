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
  int init_iter_num = 1;                  // 初始化迭代次数，其实就是记录初始化时 一共用了多少个IMU测量来求平均
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

/**
 * @brief: 重置参数
 * @return {*}
 */
void ImuProcess::Reset()   
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

/**
 * @brief: 向ImuProcess类中成员变量传入参数
 * @param {V3D} &transl
 * @param {M3D} &rot
 * @param {V3D} &gyr
 * @param {V3D} &acc
 * @param {V3D} &gyr_bias
 * @param {V3D} &acc_bias
 * @return {*}
 */
void ImuProcess::set_param(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias)  
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
  cov_gyr_scale = gyr;
  cov_acc_scale = acc;
  cov_bias_gyr = gyr_bias;
  cov_bias_acc = acc_bias;
}


/**
 * @brief: IMU初始化：利用开始的IMU帧的平均值初始化状态量x。程序启动的几秒钟保持设备静止，因为本函数中要求平均值的方式初始化状态量
 * g 
 * @param {MeasureGroup} &meas
 * @param {esekf} &kf_state
 * @param {int} &N 记录有多少个IMU数据参与了初始化求平均。
 * @return {*}
 */
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N)
{
  //MeasureGroup这个struct表示当前过程中正在处理的所有数据，包含IMU队列和一帧lidar的点云 以及lidar的起始和结束时间
  //初始化重力、陀螺仪偏差、acc和陀螺仪协方差  将加速度测量值归一化为单位重力   **/
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_) // 如果现在输入的meas为第一帧
  {
    Reset();    //重置参数
    N = 1;      //将迭代次数置1
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;    //IMU初始时刻的加速度
    const auto &gyr_acc = meas.imu.front()->angular_velocity;       //IMU初始时刻的角速度
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;              //第一帧加速度值作为初始化均值
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;              //第一帧角速度值作为初始化均值
    first_lidar_time = meas.lidar_beg_time;                   //将当前IMU帧对应的lidar起始时间 作为初始时间
  }

  for (const auto &imu : meas.imu)    //根据所有IMU数据，计算平均值和方差（采用递推的方式计算平均值）
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc  += (cur_acc - mean_acc) / N;    // 更新加速度和角速度的均值
    mean_gyr  += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc)  / N;               // 更新协方差
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr)  / N / N * (N-1);

    N ++;
  }
  
  state_ikfom init_state = kf_state.get_x();                  // 在esekfom.hpp获得x_的默认状态
  init_state.grav = - mean_acc / mean_acc.norm() * G_m_s2;    // 初始的重力加速度 = 静止放置时加速度测量值的单位方向向量 * 重力加速度预设值。这里的处理是为了适应不同厂家的IMU可能重力不是9.81
  
  // 设置初始的state
  // *这里并没有初始化ba，直接默认ba初始时是等于0的
  init_state.bg  = mean_gyr;                      // 陀螺仪bias 初始化为静止放置时角速度的测量值
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;      // 设置lidar和imu外参初始值
  init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);
  kf_state.change_x(init_state);      //将初始化后的状态传入esekfom.hpp中的x_

  // 设置协方差的初始值
  Matrix<double, 24, 24> init_P = MatrixXd::Identity(24,24);      // 临时变量
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = init_P(23,23) = 0.00001; 
  kf_state.change_P(init_P);      // 将初始化后的状态传入esekfom.hpp中的P_
  last_imu_ = meas.imu.back();    // 记录meas这一帧中的最后一个IMU数据。后续第二帧来了以后，进行传播的时候会用到。

  // std::cout << "IMU init new -- init_state  " << init_state.pos  <<" " << init_state.bg <<" " << init_state.ba <<" " << init_state.grav << std::endl;
}

/**
 * @brief: 前向传播，反向传播，去畸变
 * @param {MeasureGroup} &meas  一帧测量（一帧点云 + 该帧时间范围内的imu）
 * @param {esekf} &kf_state 状态
 * @param {PointCloudXYZI} &pcl_out 输出去畸变的点云
 * @return {*}
 */
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_out)
{
  // * 将上一帧尾部最后一个imu添加到当前帧头部
  auto v_imu = meas.imu;         //取出当前帧的IMU队列
  v_imu.push_front(last_imu_);   //将上一帧最后尾部的imu添加到当前帧头部
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();    //拿到当前帧尾部的imu的时间
  const double &pcl_beg_time = meas.lidar_beg_time;      // 点云开始和结束的时间戳   //?按道理讲，这里的 pcl_end_time 应该和 last_lidar_end_time_ 相等吧？
  const double &pcl_end_time = meas.lidar_end_time;
  
  // 根据点云中每个点的时间戳对点云进行重排序
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);  //注意，curvature中存放了每个点相对于帧开始的相对时间（在preprocess.cpp中）


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
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),      // 求中值
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr  = acc_avr * G_m_s2 / mean_acc.norm(); //通过重力数值对加速度进行调整(除上初始化的IMU大小*9.8)

    //会有一次IMU数据开始时刻早于上次雷达帧结束时刻，单独处理一下(因为将上次最后一个IMU插入到此次开头了，所以会出现一次这种情况)
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_; //从上次雷达时刻末尾（或者说是这一帧起始时刻）开始传播 计算与tail之间的时间差
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
    
    // 更新angvel_last = tail角速度-前向传播更新后的bias  
    angvel_last = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
    
    // 更新世界坐标系下的加速度acc_s_last = R*(加速度-bias) - g
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

  // 前面的for循环里面没做最后一个IMU数据到雷达帧末尾的传播，这里补上单独处理一下
  dt = abs(pcl_end_time - imu_end_time); //dt就是从最后一个IMU数据时间戳到该雷达帧结束时刻的时间间隔
  kf_state.predict(dt, Q, in);  // 前向传播，这里的in就比较粗暴了，直接用了上面for循环最后一次的in
  imu_state = kf_state.get_x(); //更新imu_state
  last_imu_ = meas.imu.back();              //保存最后一个IMU测量，以便于下一帧使用
  last_lidar_end_time_ = pcl_end_time;      //保存这一帧最后一个雷达测量的结束时间，以便于下一帧使用

  /** 
   * *前向传播结束 下面开始反向传播，畸变矫正
   */

  if (pcl_out.points.begin() == pcl_out.points.end()) return; // 头指针=尾指针 即点云是空的，直接返回

  //遍历雷达帧覆盖时间内的每个IMU状态，注意这里是end（时间戳最小）开始，往前遍历该帧时间范围内的所有IMUpose
  /**
   * begin----------------------------end
   * ---<--head(it_k-1)---tail(it_k)-----
   */
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);       // head IMU旋转矩阵
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);     // head IMU速度
    pos_imu<<VEC_FROM_ARRAY(head->pos);     // head IMU位置
    acc_imu<<VEC_FROM_ARRAY(tail->acc);     // tail IMU加速度
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);  // tail IMU角速度

    //之前点云按照时间从小到大排序过，IMUpose也同样是按照时间从小到大push进入的。时间戳：begin--小  end--大
    //此时从pcl_out的末尾开始往前遍历，也就是从时间戳最大的点开始往小遍历，因此只需要判断 点的时间戳 >IMU head时刻即可，不需要判断点云时间<IMU tail
    auto it_pcl = pcl_out.points.end() - 1;
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;    //点和head IMU时刻之间的时间间隔。curvature是点到雷达帧开始时刻的时间差。offset_time是该IMU_pose到这一帧雷达帧开始时刻的时间差

      /*    P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)    */

      M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt).matrix() );   //该点时间戳时刻，IMU坐标系相对于世界坐标系的旋转：head IMU旋转矩阵 * exp(tail角速度*dt)   // ? 这里使用的是tail角速度作为该时间间隔内的平均角速度
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);   //点在(该点时间戳时刻的雷达坐标系)下的位置()
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);   //该点时间戳时刻，IMU坐标系的世界位置 - 雷达帧结束时刻的IMU世界位置
      // 畸变矫正之后的点的坐标（在雷达帧结束时刻的雷达坐标系中描述）
      V3D P_compensate = imu_state.offset_R_L_I.matrix().transpose() * (imu_state.rot.matrix().transpose() * (R_i * (imu_state.offset_R_L_I.matrix() * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;  //? 我感觉这一句执行不到，因为 即便是 points.begin 的时间肯定比 IMUpose.begin 的时间小，进不了这个for循环
    }
  }
}


double T1,T2;
/**
 * @brief: IMU初始化、初始化完成后调用UndistortPcl()进行前向传播反向传播和运动补偿
 * @param {MeasureGroup} &meas  打包的数据（包括一帧雷达点云，以及在该帧雷达时间范围内的IMU测量）
 * @param {esekf} &kf_state 待优化的状态量
 * @param {Ptr} &cur_pcl_un_ 输出 运动补偿后的点云
 * @return {*}
 */
void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &cur_pcl_un_)
{
  // T1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};   // 判断imu是不是空的
  ROS_ASSERT(meas.lidar != nullptr);  // 判断雷达点云是不是空的

  // 如果IMU 初始化还没有完成，那么这一帧的IMU数据继续拿来用作初始化
  if (imu_need_init_)   
  {
    // The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);  // IMU 初始化

    imu_need_init_ = true;          // ! 这一句似乎也没用
    
    last_imu_ = meas.imu.back();    // ! 这一句在IMU_init()函数的末尾已经做完了

    state_ikfom imu_state = kf_state.get_x(); // ! 这个局部变量后面也没用到啊？

    // init_iter_num 其实具体就是 到现在为止有多少个IMU测量值参与了求平均值， 要是我们用到了足够多的IMU数据，比MAX_INI_COUNT多的话，就认为这个初始化的状态比较准了
    if (init_iter_num > MAX_INI_COUNT)
    {
      // ! what? 这里写了些啥？？IMU_init()初始化的cov_acc白算了？
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false; // 初始化结束，标记一下“不再需要初始化了”

      cov_acc = cov_acc_scale;  //! 这里又是在干嘛？ 上面的也白算了？直接用用户传入的协方差初始值了？issue有人提出了这个问题但是没人回答 https://github.com/hku-mars/FAST_LIO/issues/341
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
    }

    return;
  }

  // 初始化完成后，之后的帧到来了就会进行前向传播和反向传播点云去畸变（运动补偿），运动补偿之后的点云存放在cur_pcl_un_中。
  UndistortPcl(meas, kf_state, *cur_pcl_un_); 

  // T2 = omp_get_wtime();
  // cout<<"[ IMU Process ]: Time: "<<T2 - T1<<endl;
}
