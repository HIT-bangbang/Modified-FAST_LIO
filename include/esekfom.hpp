#ifndef ESEKFOM_EKF_HPP1
#define ESEKFOM_EKF_HPP1

#include <vector>
#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "use-ikfom.hpp"
#include <ikd-Tree/ikd_Tree.h>

//该hpp主要包含：广义加减法，前向传播主函数，计算特征点残差及其雅可比，ESKF主函数

const double epsi = 0.001; // ESKF迭代时，如果dx<epsi 认为收敛

namespace esekfom
{
	using namespace Eigen;

	PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));		  //特征点在地图中对应的平面参数(平面的单位法向量,以及当前点到平面距离)
	PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); //有效特征点
	PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); //有效特征点对应点法相量
	bool point_selected_surf[100000] = {1};							  //判断是否是有效特征点

	struct dyn_share_datastruct
	{
		bool valid;												   // 有效特征点数量是否满足要求
		bool converge;											   // 迭代时，是否已经收敛
		Eigen::Matrix<double, Eigen::Dynamic, 1> h;				   // 残差	(公式(14)中的z)
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x; // 雅可比矩阵H (公式(14)中的H) m*12 //* 实际上应该是 m*24 但是后12列与残差无关，肯定是0，为了节省内存和算力就省略了
	};

	class esekf
	{
	public:
		typedef Matrix<double, 24, 24> cov;				// 24X24的协方差矩阵
		typedef Matrix<double, 24, 1> vectorized_state; // 24X1的向量

		esekf(){};
		~esekf(){};

		state_ikfom get_x()
		{
			return x_;
		}

		cov get_P()
		{
			return P_;
		}

		void change_x(state_ikfom &input_state)
		{
			x_ = input_state;
		}

		void change_P(cov &input_cov)
		{
			P_ = input_cov;
		}

		/**
		 * @brief: 广义加法  公式(4)
		 * @param {state_ikfom} x
		 * @param {Matrix<double, 24, 1>} f_
		 * @return {*}
		 */
		state_ikfom boxplus(state_ikfom x, Eigen::Matrix<double, 24, 1> f_)
		{
			state_ikfom x_r;
			x_r.pos = x.pos + f_.block<3, 1>(0, 0);

			x_r.rot = x.rot * Sophus::SO3::exp(f_.block<3, 1>(3, 0));
			x_r.offset_R_L_I = x.offset_R_L_I * Sophus::SO3::exp(f_.block<3, 1>(6, 0));

			x_r.offset_T_L_I = x.offset_T_L_I + f_.block<3, 1>(9, 0);
			x_r.vel = x.vel + f_.block<3, 1>(12, 0);
			x_r.bg = x.bg + f_.block<3, 1>(15, 0);
			x_r.ba = x.ba + f_.block<3, 1>(18, 0);
			x_r.grav = x.grav + f_.block<3, 1>(21, 0);

			return x_r;
		}

		/**
		 * @brief: IMU前向传播  公式(4-8)
		 * @param {double} &dt
		 * @param {Matrix<double, 12, 12>} &Q
		 * @param {input_ikfom} &i_in
		 * @return {*}
		 */
		void predict(double &dt, Eigen::Matrix<double, 12, 12> &Q, const input_ikfom &i_in)
		{
			Eigen::Matrix<double, 24, 1> f_ = get_f(x_, i_in);	  //公式(3)的f
			Eigen::Matrix<double, 24, 24> f_x_ = df_dx(x_, i_in); //公式(7)的df/dx
			Eigen::Matrix<double, 24, 12> f_w_ = df_dw(x_, i_in); //公式(7)的df/dw

			x_ = boxplus(x_, f_ * dt); //前向传播 公式(4)

			f_x_ = Matrix<double, 24, 24>::Identity() + f_x_ * dt; //get_f()计算的Fx矩阵 没加单位阵，也没乘dt   这里补上

			P_ = (f_x_)*P_ * (f_x_).transpose() + (dt * f_w_) * Q * (dt * f_w_).transpose(); //传播协方差矩阵，即公式(8) f_w_之前也没乘dt，这里乘上
		}

		/**
		 * @brief: 寻找近邻点、计算残差及H矩阵
		 * @param {Ptr} &feats_down_body
		 * @param {KD_TREE<PointType>} &ikdtree
		 * @param {vector<PointVector>} &Nearest_Points	每个点的近邻vector
		 * @param {bool} extrinsic_est	是否估计外参
		 * @return {*}
		 */
		void h_share_model(dyn_share_datastruct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
						   KD_TREE<PointType> &ikdtree, vector<PointVector> &Nearest_Points, bool extrinsic_est)
		{
			int feats_down_size = feats_down_body->points.size();	// 先拿到点云的数量
			laserCloudOri->clear();
			corr_normvect->clear();
// 开多线程
#ifdef MP_EN
			omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
			// * 对feats_down_body中的每个点i寻找5个近邻点
			for (int i = 0; i < feats_down_size; i++) //遍历这一帧中所有的激光点
			{
				PointType &point_body = feats_down_body->points[i];		// 拿出来一个点i
				PointType point_world;	// 存点i在世界坐标系下的位姿

				V3D p_body(point_body.x, point_body.y, point_body.z);
				// 把Lidar坐标系的点先转到IMU坐标系，再根据前向传播估计的位姿x，转到世界坐标系，存到了point_world
				V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);
				point_world.x = p_global(0);
				point_world.y = p_global(1);
				point_world.z = p_global(2);
				point_world.intensity = point_body.intensity;

				vector<float> pointSearchSqDis(NUM_MATCH_POINTS);// 维度为NUM_MATCH_POINTS(近邻点数量)的vector，存储点i和它的每个近邻点之间的距离
				auto &points_near = Nearest_Points[i]; // Nearest_Points[i]打印出来发现是按照离point_world距离，从小到大的顺序的vector

				double ta = omp_get_wtime();
				if (ekfom_data.converge)				// 判断是否收敛
				{
					// * step 1.对点i寻找5个近邻点
					// 寻找point_world的最近邻的平面点。搜索到的近邻点存到points_near，和每个近邻点的距离存到pointSearchSqDis
					ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
					// 判断使用点i进行匹配是否可靠，与loam系列类似，要求特征点最近邻的地图点数量>阈值，最远的近邻点的距离<阈值  满足条件的才置为true
					point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
																																		: true;
					//! 注意 points_near 和 pointSearchSqDis 是按照从小到大排序的。也就是 离i点越远的近邻点，其索引越大。所以这里pointSearchSqDis[NUM_MATCH_POINTS - 1]就是5个近邻点中离点i最远的那个
				}
				if (!point_selected_surf[i])	// 如果点i不满足条件 跳过 不进行下面步骤
					continue; 

				Matrix<float, 4, 1> pabcd;		//平面点信息
				point_selected_surf[i] = false; //用来判断是否满足条件 首先将该点设置为无效点，
				
				// * step 2. 拟合平面方程
				if (esti_plane(pabcd, points_near, 0.1f)) // 拟合平面方程ax+by+cz+d=0并求解点到平面距离，返回 算出来的平面是不是一个好的平面
				{
					float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3); //当前点到平面的距离
					float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());												   //如果残差大于经验阈值，则认为该点是有效点  简言之，距离原点越近的lidar点  要求点到平面的距离越苛刻

					if (s > 0.9) //如果残差大于阈值，则认为该点是有效点
					{
						point_selected_surf[i] = true;		// 该点是有效点
						normvec->points[i].x = pabcd(0); 	// 用xyz存储平面的单位法向量
						normvec->points[i].y = pabcd(1);
						normvec->points[i].z = pabcd(2);
						normvec->points[i].intensity = pd2;	// 使用 intensity 的位置存一下点到平面的距离
					}
				}
			}

			int effct_feat_num = 0; //	有效的点的数量
			// 开一个for循环，把前面找到的所有满足要求的点i存到laserCloudOri里，它们对应的面的法向量和它们到平面的距离存到 corr_normvect 里面
			for (int i = 0; i < feats_down_size; i++)
			{
				if (point_selected_surf[i]) //对于满足要求的点
				{
					laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; //把这些点重新存到laserCloudOri中
					corr_normvect->points[effct_feat_num] = normvec->points[i];			//存储这些点对应的法向量和到平面的距离
					effct_feat_num++;
				}
			}

			//满足要求的点太少了
			if (effct_feat_num < 1)
			{
				ekfom_data.valid = false;
				ROS_WARN("No Effective Points! \n");
				return;
			}

			// * step 3. 计算雅可比矩阵H和残差向量
			ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);	// H 		m*12
			ekfom_data.h.resize(effct_feat_num);					// 残差		m*1

			for (int i = 0; i < effct_feat_num; i++)
			{
				V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
				M3D point_crossmat;
				point_crossmat << SKEW_SYM_MATRX(point_);
				V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;	// 将点i从雷达坐标系转换到IMU坐标系下
				M3D point_I_crossmat;
				point_I_crossmat << SKEW_SYM_MATRX(point_I_);

				const PointType &norm_p = corr_normvect->points[i];		// 拿出来 点i的近邻点构成平面的法向量，以及点i到平面的距离
				V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);		// 平面的法向量

				// * 计算雅可比矩阵H
				V3D C(x_.rot.matrix().transpose() * norm_vec);
				V3D A(point_I_crossmat * C);
				if (extrinsic_est)
				{
					//如果需要优化外参，则雅可比的形式形式如下
					V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
					ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
				}
				else
				{
					//如果不需要优化外参，则雅可比的形式形式如下
					ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
				}

				// * 计算残差，其实就是点i到面的距离
				ekfom_data.h(i) = -norm_p.intensity;	// 注意，前面在点的 intensity 位置上存放了点到直线的距离。这里把它存到 ekfom_data 里面去
								//? 这里的负号？？
			}
		}

		/**
		 * @brief: 广义减法
		 * @param {state_ikfom} x1
		 * @param {state_ikfom} x2
		 * @return {*}
		 */
		vectorized_state boxminus(state_ikfom x1, state_ikfom x2)
		{
			vectorized_state x_r = vectorized_state::Zero();

			x_r.block<3, 1>(0, 0) = x1.pos - x2.pos;

			x_r.block<3, 1>(3, 0) = Sophus::SO3(x2.rot.matrix().transpose() * x1.rot.matrix()).log();
			x_r.block<3, 1>(6, 0) = Sophus::SO3(x2.offset_R_L_I.matrix().transpose() * x1.offset_R_L_I.matrix()).log();

			x_r.block<3, 1>(9, 0) = x1.offset_T_L_I - x2.offset_T_L_I;
			x_r.block<3, 1>(12, 0) = x1.vel - x2.vel;
			x_r.block<3, 1>(15, 0) = x1.bg - x2.bg;
			x_r.block<3, 1>(18, 0) = x1.ba - x2.ba;
			x_r.block<3, 1>(21, 0) = x1.grav - x2.grav;

			return x_r;
		}

		/**
		 * @brief: ESKF 更新，包括寻找近邻点计算残差和雅可比
		 * @param {double} R 对应论文中的R矩阵，这里认为所有的元素都是相等的，所以用一个double
		 * @param {Ptr} &feats_down_body 	输入 下采样之后的当前帧点云
		 * @param {KD_TREE<PointType>} &ikdtree 	输入
		 * @param {vector<PointVector>} &Nearest_Points 	输出，每个点的近邻点集合构成的vector<vector>
		 * @param {int} maximum_iter	最大迭代次数
		 * @param {bool} extrinsic_est	是否要自动估计外参
		 * @return {*}
		 */
		void update_iterated_dyn_share_modified(double R, PointCloudXYZI::Ptr &feats_down_body,
												KD_TREE<PointType> &ikdtree, vector<PointVector> &Nearest_Points, int maximum_iter, bool extrinsic_est)
		{
			normvec->resize(int(feats_down_body->points.size()));

			dyn_share_datastruct dyn_share;
			dyn_share.valid = true;
			dyn_share.converge = true;
			int t = 0;//迭代次数t
			state_ikfom x_propagated = x_; //这里的x_和P_分别是经过正向传播后的状态量和协方差矩阵，这里先把它存下来。对应公式18 里面的 x_k_hat 和 P
			cov P_propagated = P_;

			vectorized_state dx_new = vectorized_state::Zero(); // 24X1的向量，对应公式(18)广义加法后面的部分

			// * 开始更新 x_ 和 P_
			for (int i = -1; i < maximum_iter; i++) // maximum_iter是卡尔曼滤波的最大迭代次数
			{
				dyn_share.valid = true;
				// * 寻找近邻点（如果dyn_share.converge = true）、计算雅克比（点面残差的导数 H(代码里是h_x)）和残差。存到 dyn_share 里面输出
				h_share_model(dyn_share, feats_down_body, ikdtree, Nearest_Points, extrinsic_est);

				if (!dyn_share.valid)
				{
					continue;
				}

				vectorized_state dx;
				dx_new = boxminus(x_, x_propagated); //公式(18)中的 x^k - x^

				auto H = dyn_share.h_x;													// 取出来前面计算好的雅可比矩阵 m*12 //* 实际上应该是 m*24 但是后12列与残差无关，求导肯定是0，为了节省内存和算力就省略了
				Eigen::Matrix<double, 24, 24> HTH = Matrix<double, 24, 24>::Zero(); 	// 矩阵 H^T*H 初始化为零矩阵
				HTH.block<12, 12>(0, 0) = H.transpose() * H;	//* HTH是24*24维度，它的左上角的12*12就是 H^T*H

				auto K_front = (HTH / R + P_.inverse()).inverse();	// 卡尔曼增益前半部(括号里面的部分)
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
				K = K_front.block<24, 12>(0, 0) * H.transpose() / R; //* 卡尔曼增益  注意这里的R，我们认为各个维度上的R相等，所以它是个对角矩阵，求逆就是倒数

				Eigen::Matrix<double, 24, 24> KH = Matrix<double, 24, 24>::Zero(); // K * H
				KH.block<24, 12>(0, 0) = K * H;
				Matrix<double, 24, 1> dx_ = K * dyn_share.h + (KH - Matrix<double, 24, 24>::Identity()) * dx_new; // 公式(18)，这里求的就是广义加后面的部分，其实就是x的变化量dx
				// std::cout << "dx_: " << dx_.transpose() << std::endl;
				x_ = boxplus(x_, dx_); //公式(18) //* 更新后的 x_

				//* 检查是否收敛，如果 dyn_share.converge 置为了 true，说明此次迭代更新已经收敛了，那么在 h_share_model 就会重新寻找近邻点，更新近邻点
				dyn_share.converge = true;
				for (int j = 0; j < 24; j++)
				{
					// 检查此次迭代的dx_中的每个量，如果有任意一个大于 epsi 就认为还没有收敛，继续迭代
					if (std::fabs(dx_[j]) > epsi)
					{
						dyn_share.converge = false;
						break;
					}
				}

				if (dyn_share.converge)
					t++;	// t其实就是重新寻找近邻点的次数

				if (!t && i == maximum_iter - 2) //如果迭代了3次还没收敛 强制令成true，这时h_share_model函数中会重新寻找近邻点
				{
					dyn_share.converge = true;
				}

				if (t > 1 || i == maximum_iter - 1)	// 如果迭代了4次或者重新寻找了一次近邻点，就认为结束了
				{
					P_ = (Matrix<double, 24, 24>::Identity() - KH) * P_; //* 公式(19) 更新协方差矩阵
					return;
				}
			}
		}

	private:
		state_ikfom x_;				// 状态量 24*1
		cov P_ = cov::Identity();	// 协方差矩阵 24*24
	};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP1
