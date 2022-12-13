#include "lp_plugin_motiontracking.h"

#include "lp_renderercam.h"
#include "lp_openmesh.h"
#include "renderer/lp_glselector.h"
#include "renderer/lp_glrenderer.h"

#include <lp_openmesh.h>

#include <QPainter>
#include <QStringList>
#include <QFileDialog>

bool s = false;

using namespace std::chrono_literals;

struct InputSettings {
  k4a_fps_t FPS = K4A_FRAMES_PER_SECOND_30;
  k4a_image_format_t ColorFormat = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  k4a_color_resolution_t ColorResolution = K4A_COLOR_RESOLUTION_720P;
  k4a_depth_mode_t DepthCameraMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
  bool SynchronizedImagesOnly = true;

#ifdef _WIN32
  k4abt_tracker_processing_mode_t processingMode =
      K4ABT_TRACKER_PROCESSING_MODE_GPU_DIRECTML;
#else
  k4abt_tracker_processing_mode_t processingMode =
      K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;
#endif
  bool Offline = false;
  std::string FileName;
  std::string ModelPath;
};
InputSettings inputSettings;

struct LP_Plugin_MotionTracking::Model_member {
    bool mUse_Model = false, mTest_Animation = false;
    bool mInitialized_R_DM = false;
    void initializeGL_R_DM(QOpenGLContext *ctx, QSurface *surf);

    float model_rotate0 = 90.0, model_rotate1 = 60.0;

    const float mModelScale = 1000.0;
    float mModel_Nearplane = 1000.0, mModel_Farplane = 3000.0;

    QVector3D mModelPos, mCam;
    QVector3D mModelBB_min, mModelBB_max;
    bool init_BB = false, init_modelpos = false;

    std::vector<std::array<float, 6>> mModelVAO;
    std::vector<uint> mModelF;

    int frame = 0;
    std::vector<int> mFrames_time;
    QString mAnimation_path = "/home/cpii/projects/build-Smart_Fashion-Desktop_Qt_5_15_1_GCC_64bit-Release/App/3D_models/benchmark/animation/";

    QOpenGLContext *mCB_Context = nullptr;
    QSurface *mCB_Surface = nullptr;

    float DecodeFloatRGBA (QVector4D rgba) {
        return QVector4D::dotProduct(rgba, QVector4D(1.0f, 1.0/255.0f, 1.0/65025.0f, 1.0/16581375.0f));
    }

    cv::Mat mDBackground, mIrBackground;
    QImage mDepth_Map_color0, mDepth_Map_color1, mNormal_Map0, mNormal_Map1;
    cv::Mat mDepth_Map_dis0, mDepth_Map_dis1, mIr_Map0, mIr_Map1;

    QOpenGLShaderProgram *mProgram_R_DM = nullptr;
    QOpenGLFramebufferObject *mR_DM_FBO = nullptr;
};

LP_Plugin_MotionTracking::r_t LP_Plugin_MotionTracking::rigid_transform_3D(cv::Mat A, cv::Mat B){
    // Find R and t (R * A + t = B)

    // find mean column wise
    cv::Mat centroid_A;
    cv::Mat centroid_B;
    cv::reduce(A, centroid_A, 1, cv::REDUCE_AVG);
    cv::reduce(B, centroid_B, 1, cv::REDUCE_AVG);

    // subtract mean
    for(int col=0; col<A.cols; col++){
        A.col(col) = A.col(col) - centroid_A;
    }
    for(int col=0; col<B.cols; col++){
        B.col(col) = B.col(col) - centroid_B;
    }

    cv::Mat H;
    H = A * B.clone().t();

    // find rotation
    cv::Mat U, S, Vt;
    cv::SVD::compute(H, S, U, Vt);

    cv::Mat R = Vt.clone().t() * U.clone().t();

    // special reflection case
    if(cv::determinant(R) < 0.0){
        qDebug() << "det(R) < 0.0, reflection detected!, correcting for it ...";
        Vt.row(2) = Vt.row(2) * -1.0;
    }

    cv::Mat t = centroid_B - R * centroid_A;

    r_t output;
    output.R = R;
    output.t = t;

    return output;
}

void LP_Plugin_MotionTracking::rigid_transform_3D_test(){
    cv::Mat R(3, 3, CV_64F);
    cv::randu(R, cv::Scalar(-3), cv::Scalar(3));

    cv::Mat t(3, 1, CV_64F);
    cv::randu(t, cv::Scalar(-500), cv::Scalar(500));

    // make R a proper rotation matrix, force orthonormal
    cv::Mat U, S, Vt;
    cv::SVD::compute(R, S, U, Vt);

    R = U * Vt;

    // special reflection case
    if(cv::determinant(R) < 0.0){
        Vt.row(2) = Vt.row(2) * -1.0;
        R = U*Vt;
    }

    // number of points
    int n = 32;

    cv::Mat A(3, n, CV_64F);
    cv::randu(A, cv::Scalar(-500), cv::Scalar(500));

    cv::Mat B = R*A;
    for(int col=0; col<B.cols; col++){
        B.col(col) = B.col(col) + t;
    }

    // Recover R and t
    r_t ret;
    ret = rigid_transform_3D(A.clone(), B.clone());

    // Compare the recovered R and t with the original
    cv::Mat B2 = ret.R * A;

    for(int col=0; col<B.cols; col++){
        B2.col(col) = B2.col(col) + ret.t;
    }

    // Find the root mean squared error
    cv::Mat err = B2 - B;
    cv::pow(err, 2, err);
    double _err = cv::sum(err)[0];
    double rmse = sqrt(_err/double(n));

    std::cout << "Points A:\n" << A << std::endl;

    std::cout << "Points B:\n" << B << std::endl;

    std::cout << "Recovered points B2:\n" << B2 << std::endl;

    std::cout << "Ground truth rotation:\n" << R << std::endl;

    std::cout << "Recovered rotation:\n" << ret.R << std::endl;

    std::cout << "Ground truth translation:\n" << t << std::endl;

    std::cout << "Recovered translation:\n" << ret.t << std::endl;

    std::cout << "RMSE:\n" << rmse << std::endl;

    if (rmse < 1e-5){
        qDebug() << "Everything looks good!";
    } else {
        qDebug() << "Hmm something doesn't look right ...";
    }

    QThread::msleep(-1);
}

LP_Plugin_MotionTracking::poly_solu_xyz LP_Plugin_MotionTracking::solve_polynomial_trajectory(k4a_float3_t po0, k4a_float3_t po1,
                                                                                              k4a_float3_t vel0, k4a_float3_t vel1,
                                                                                              float times0, float times1){
    // Joint Space Trajectory Cubic
    // solve Ax = b

    poly_solu_xyz solu;

    cv::Mat A = (cv::Mat_<double>(4, 4) << 1.0, times0, times0*times0, times0*times0*times0,
                                           1.0, times1, times1*times1, times1*times1*times1,
                                           0.0, 1.0, 2.0*times0, 3.0*times0*times0,
                                           0.0, 1.0, 2.0*times1, 3.0*times1*times1);

    cv::Mat bx = (cv::Mat_<double>(4, 1) << po0.xyz.x,
                                            po1.xyz.x,
                                            vel0.xyz.x,
                                            vel1.xyz.x);
    cv::Mat by = (cv::Mat_<double>(4, 1) << po0.xyz.y,
                                            po1.xyz.y,
                                            vel0.xyz.y,
                                            vel1.xyz.y);
    cv::Mat bz = (cv::Mat_<double>(4, 1) << po0.xyz.z,
                                            po1.xyz.z,
                                            vel0.xyz.z,
                                            vel1.xyz.z);

    cv::Mat solutionx = A.inv() * bx;
    cv::Mat solutiony = A.inv() * by;
    cv::Mat solutionz = A.inv() * bz;

//    qDebug() << "times0: " << times0;
//    qDebug() << "times1: " << times1;
//    std::cout << "A:" << A << std::endl
//              << "bx: " << bx << std::endl
//              << "by: " << by << std::endl
//              << "bz: " << bz << std::endl
//              << "solutionx: " << solutionx << std::endl
//              << "solutiony: " << solutiony << std::endl
//              << "solutionz: " << solutionz << std::endl;

    solu.x.a0 = solutionx.at<double>(0, 0);
    solu.x.a1 = solutionx.at<double>(1, 0);
    solu.x.a2 = solutionx.at<double>(2, 0);
    solu.x.a3 = solutionx.at<double>(3, 0);
    solu.y.a0 = solutiony.at<double>(0, 0);
    solu.y.a1 = solutiony.at<double>(1, 0);
    solu.y.a2 = solutiony.at<double>(2, 0);
    solu.y.a3 = solutiony.at<double>(3, 0);
    solu.z.a0 = solutionz.at<double>(0, 0);
    solu.z.a1 = solutionz.at<double>(1, 0);
    solu.z.a2 = solutionz.at<double>(2, 0);
    solu.z.a3 = solutionz.at<double>(3, 0);

    return solu;
}

void LP_Plugin_MotionTracking::cal_xyz(poly_solu_xyz solu, float time, int joint){
    //qDebug() << "Calculating xyz";
    tmpbody_b.joints[joint].position.xyz.x = solu.x.a0 + solu.x.a1*time + solu.x.a2*time*time + solu.x.a3*time*time*time;
    tmpbody_b.joints[joint].position.xyz.y = solu.y.a0 + solu.y.a1*time + solu.y.a2*time*time + solu.y.a3*time*time*time;
    tmpbody_b.joints[joint].position.xyz.z = solu.z.a0 + solu.z.a1*time + solu.z.a2*time*time + solu.z.a3*time*time*time;
}

void LP_Plugin_MotionTracking::Process_JointTraj(){
    //qDebug() << "Processing joint trajectory";

    k4a_float3_t tmp;
    std::vector<float> times;
    uint64_t Start_t = mJoint_Traj.Traj_timestampsA[0] < mJoint_Traj.Traj_timestampsB[0] ? mJoint_Traj.Traj_timestampsA[0] : mJoint_Traj.Traj_timestampsB[0];

    for(int f=0; f<5; f++){
        times.push_back(float(mJoint_Traj.Traj_timestampsB[f] - mJoint_Traj.Traj_timestampsB[0])*0.001);
    }

    if(mJoint_Traj.Traj_velB.empty()){
        for(int f=0; f<5; f++){
            std::vector<k4a_float3_t> tmp_vel;
            for(int joint=0; joint<K4ABT_JOINT_COUNT; joint++){
                if(f==0){
                    tmp.xyz.x = 0.0;
                    tmp.xyz.y = 0.0;
                    tmp.xyz.z = 0.0;
                    tmp_vel.push_back(tmp);
                } else if(f==4){
                    tmp.xyz.x = mJoint_Traj.Traj_velB[3][joint].xyz.x * 0.5;
                    tmp.xyz.y = mJoint_Traj.Traj_velB[3][joint].xyz.y * 0.5;
                    tmp.xyz.z = mJoint_Traj.Traj_velB[3][joint].xyz.z * 0.5;
                    tmp_vel.push_back(tmp);
                } else {
                    k4a_float3_t tmp_j0 = mJoint_Traj.Traj_skesB[f-1].joints[joint].position;
                    k4a_float3_t tmp_j1 = mJoint_Traj.Traj_skesB[f].joints[joint].position;
                    k4a_float3_t tmp_j2 = mJoint_Traj.Traj_skesB[f+1].joints[joint].position;
                    float t0 = times[f] - times[f-1];
                    float t1 = times[f+1] - times[f];
                    float v1_x = (tmp_j1.xyz.x - tmp_j0.xyz.x) / t0;
                    float v2_x = (tmp_j2.xyz.x - tmp_j1.xyz.x) / t1;
                    float v1_y = (tmp_j1.xyz.y - tmp_j0.xyz.y) / t0;
                    float v2_y = (tmp_j2.xyz.y - tmp_j1.xyz.y) / t1;
                    float v1_z = (tmp_j1.xyz.z - tmp_j0.xyz.z) / t0;
                    float v2_z = (tmp_j2.xyz.z - tmp_j1.xyz.z) / t1;
                    if(v1_x * v2_x > 0){
                        tmp.xyz.x = v1_x * t0 + v2_x * t1;
                    } else {
                        tmp.xyz.x = 0.0;
                    }
                    if(v1_y * v2_y > 0){
                        tmp.xyz.y = v1_y * t0 + v2_y * t1;
                    } else {
                        tmp.xyz.y = 0.0;
                    }
                    if(v1_z * v2_z > 0){
                        tmp.xyz.z = v1_z * t0 + v2_z * t1;
                    } else {
                        tmp.xyz.z = 0.0;
                    }
                    tmp_vel.push_back(tmp);
                }
            }
            mJoint_Traj.Traj_velB.push_back(tmp_vel);
        }
    } else {
        for(int f=3; f<5; f++){
            std::vector<k4a_float3_t> tmp_vel;
            for(int joint=0; joint<K4ABT_JOINT_COUNT; joint++){
                if(f==3){
                    k4a_float3_t tmp_j0 = mJoint_Traj.Traj_skesB[f-1].joints[joint].position;
                    k4a_float3_t tmp_j1 = mJoint_Traj.Traj_skesB[f].joints[joint].position;
                    k4a_float3_t tmp_j2 = mJoint_Traj.Traj_skesB[f+1].joints[joint].position;
                    float t0 = times[f] - times[f-1];
                    float t1 = times[f+1] - times[f];
                    float v1_x = (tmp_j1.xyz.x - tmp_j0.xyz.x) / t0;
                    float v2_x = (tmp_j2.xyz.x - tmp_j1.xyz.x) / t1;
                    float v1_y = (tmp_j1.xyz.y - tmp_j0.xyz.y) / t0;
                    float v2_y = (tmp_j2.xyz.y - tmp_j1.xyz.y) / t1;
                    float v1_z = (tmp_j1.xyz.z - tmp_j0.xyz.z) / t0;
                    float v2_z = (tmp_j2.xyz.z - tmp_j1.xyz.z) / t1;
                    if(v1_x * v2_x > 0){
                        tmp.xyz.x = v1_x * t0 + v2_x * t1;
                    } else {
                        tmp.xyz.x = 0.0;
                    }
                    if(v1_y * v2_y > 0){
                        tmp.xyz.y = v1_y * t0 + v2_y * t1;
                    } else {
                        tmp.xyz.y = 0.0;
                    }
                    if(v1_z * v2_z > 0){
                        tmp.xyz.z = v1_z * t0 + v2_z * t1;
                    } else {
                        tmp.xyz.z = 0.0;
                    }
                    tmp_vel.push_back(tmp);
                } else {
                    tmp.xyz.x = mJoint_Traj.Traj_velB[3][joint].xyz.x * 0.5;
                    tmp.xyz.y = mJoint_Traj.Traj_velB[3][joint].xyz.y * 0.5;
                    tmp.xyz.z = mJoint_Traj.Traj_velB[3][joint].xyz.z * 0.5;
                    tmp_vel.push_back(tmp);
                }
            }
            mJoint_Traj.Traj_velB.push_back(tmp_vel);
        }
    }

//    std::string p = "/home/cpii/Desktop/test_graph/Joint_Traj/" + std::to_string(mJoint_Traj.tmpf) + ".csv";
//    std::ofstream myFile(p);
//    for(int f=0; f<4; f++){
//        int joint = K4ABT_JOINT_HAND_LEFT;
//        k4a_float3_t tmp_j0 = mJoint_Traj.Traj_skesB[f].joints[joint].position;
//        k4a_float3_t tmp_j1 = mJoint_Traj.Traj_skesB[f+1].joints[joint].position;
//        k4a_float3_t tmp_v0 = mJoint_Traj.Traj_velB[f][joint];
//        k4a_float3_t tmp_v1 = mJoint_Traj.Traj_velB[f+1][joint];
//        poly_solu_xyz solu = solve_polynomial_trajectory(tmp_j0, tmp_j1,
//                                                         tmp_v0, tmp_v1,
//                                                         times[f], times[f+1]);
//        for(int t=0; t<int((times[f+1]-times[f])*1000); t++){
//            float time = float(t)*0.001+times[f];
//            float x = solu.x.a0 + solu.x.a1*time + solu.x.a2*time*time + solu.x.a3*time*time*time;
//            float y = solu.y.a0 + solu.y.a1*time + solu.y.a2*time*time + solu.y.a3*time*time*time;
//            float z = solu.z.a0 + solu.z.a1*time + solu.z.a2*time*time + solu.z.a3*time*time*time;

//            myFile << time;
//            myFile << " ";
//            myFile << x;
//            myFile << " ";
//            myFile << y;
//            myFile << " ";
//            myFile << z;
//            myFile << "\n";
//        }
//    }
//    myFile.close();
//    mJoint_Traj.tmpf += 2;

    // Calculate the solution for the third frame
    for(int joint=0; joint<K4ABT_JOINT_COUNT; joint++){
        k4a_float3_t tmp_j0 = mJoint_Traj.Traj_skesB[2].joints[joint].position;
        k4a_float3_t tmp_j1 = mJoint_Traj.Traj_skesB[3].joints[joint].position;
        k4a_float3_t tmp_v0 = mJoint_Traj.Traj_velB[2][joint];
        k4a_float3_t tmp_v1 = mJoint_Traj.Traj_velB[3][joint];
        poly_solu_xyz solu = solve_polynomial_trajectory(tmp_j0, tmp_j1,
                                                         tmp_v0, tmp_v1,
                                                         times[2], times[3]);

        float time = (float(mJoint_Traj.Traj_timestampsA[2] + mJoint_Traj.Traj_timestampsA[3]) * 0.5 - float(Start_t))*0.001;
        cal_xyz(solu, time, joint);
    }
}

void LP_Plugin_MotionTracking::erase_traj(){
    mJoint_Traj.Traj_skesB.erase(mJoint_Traj.Traj_skesB.begin());
    mJoint_Traj.Traj_timestampsA.erase(mJoint_Traj.Traj_timestampsA.begin());
    mJoint_Traj.Traj_timestampsB.erase(mJoint_Traj.Traj_timestampsB.begin());
    mJoint_Traj.Traj_velB.erase(mJoint_Traj.Traj_velB.begin());
    mJoint_Traj.Traj_velB.pop_back();
}

void LP_Plugin_MotionTracking::Process_AllJointTraj(){
    qDebug() << "Processing all joint trajetory";

    k4a_float3_t tmp;
    std::vector<float> times;
    std::vector<std::vector<k4a_float3_t>> traj_velB;
    uint64_t Start_t = mSkeData.cam0_t_ori[0] < mSkeData.cam1_t_ori[0] ? mSkeData.cam0_t_ori[0] : mSkeData.cam1_t_ori[0];

    for(int f=0; f<mSkeData.cam1_t_ori.size(); f++){
        times.push_back(float(mSkeData.cam1_t_ori[f] - mSkeData.cam1_t_ori[0])*0.001);
    }

    for(int f=0; f<mSkeData.cam1_t_ori.size(); f++){
        std::vector<k4a_float3_t> tmp_vel;
        for(int joint=0; joint<K4ABT_JOINT_COUNT; joint++){
            if(f==0 || f==mSkeData.cam1_t_ori.size()-1){
                tmp.xyz.x = 0.0;
                tmp.xyz.y = 0.0;
                tmp.xyz.z = 0.0;
                tmp_vel.push_back(tmp);
            } else {
                k4a_float3_t tmp_j0 = mSkeData.skes1[f-1].joints[joint].position;
                k4a_float3_t tmp_j1 = mSkeData.skes1[f].joints[joint].position;
                k4a_float3_t tmp_j2 = mSkeData.skes1[f+1].joints[joint].position;
                float t0 = times[f] - times[f-1];
                float t1 = times[f+1] - times[f];
                float v1_x = (tmp_j1.xyz.x - tmp_j0.xyz.x) / t0;
                float v2_x = (tmp_j2.xyz.x - tmp_j1.xyz.x) / t1;
                float v1_y = (tmp_j1.xyz.y - tmp_j0.xyz.y) / t0;
                float v2_y = (tmp_j2.xyz.y - tmp_j1.xyz.y) / t1;
                float v1_z = (tmp_j1.xyz.z - tmp_j0.xyz.z) / t0;
                float v2_z = (tmp_j2.xyz.z - tmp_j1.xyz.z) / t1;
                if(v1_x * v2_x > 0){
                    tmp.xyz.x = v1_x * t0 + v2_x * t1;
                } else {
                    tmp.xyz.x = 0.0;
                }
                if(v1_y * v2_y > 0){
                    tmp.xyz.y = v1_y * t0 + v2_y * t1;
                } else {
                    tmp.xyz.y = 0.0;
                }
                if(v1_z * v2_z > 0){
                    tmp.xyz.z = v1_z * t0 + v2_z * t1;
                } else {
                    tmp.xyz.z = 0.0;
                }
                tmp_vel.push_back(tmp);
            }
        }
        traj_velB.push_back(tmp_vel);
    }

    for(int f=0; f<mSkeData.cam1_t_ori.size()-1; f++){
        k4abt_skeleton_t tmpb;
        for(int joint=0; joint<K4ABT_JOINT_COUNT; joint++){
            k4a_float3_t tmp_j0 = mSkeData.skes1[f].joints[joint].position;
            k4a_float3_t tmp_j1 = mSkeData.skes1[f+1].joints[joint].position;
            k4a_float3_t tmp_v0 = traj_velB[f][joint];
            k4a_float3_t tmp_v1 = traj_velB[f+1][joint];
            poly_solu_xyz solu = solve_polynomial_trajectory(tmp_j0, tmp_j1,
                                                             tmp_v0, tmp_v1,
                                                             times[f], times[f+1]);

            float time = (float(mSkeData.cam0_t_ori[f] + mSkeData.cam0_t_ori[f+1]) * 0.5 - float(Start_t))*0.001;
            tmpb.joints[joint].position.xyz.x = solu.x.a0 + solu.x.a1*time + solu.x.a2*time*time + solu.x.a3*time*time*time;
            tmpb.joints[joint].position.xyz.y = solu.y.a0 + solu.y.a1*time + solu.y.a2*time*time + solu.y.a3*time*time*time;
            tmpb.joints[joint].position.xyz.z = solu.z.a0 + solu.z.a1*time + solu.z.a2*time*time + solu.z.a3*time*time*time;
        }
        bodyb_all.push_back(tmpb);
    }
    bodyb_all.push_back(mSkeData.skes1.back());

    QString path("/home/cpii/Desktop/test_img/camall/");
    if(!QDir(path).exists()){
        QDir().mkdir(path);
    }
    std::string p = "/home/cpii/Desktop/test_graph/Joint_AllTraj/Joint_AllTraj.csv";
    std::ofstream myFile(p);
    mJoint_tmp_num = 32; joint_a_num = 32; joint_b_num = 32; mBone_tmp_num = 31; bone_a_num = 31; bone_b_num = 31;
    for(int f=0; f<mSkeData.cam1_t_ori.size(); f++){
        if(f==0){
            SkeletonoperationRunoptimizationall(0, f);
            myFile << mRest_skeleton.joints[K4ABT_JOINT_HAND_LEFT].position.xyz.x;
            myFile << " ";
            myFile << mRest_skeleton.joints[K4ABT_JOINT_HAND_LEFT].position.xyz.y;
            myFile << " ";
            myFile << mRest_skeleton.joints[K4ABT_JOINT_HAND_LEFT].position.xyz.z;
            myFile << "\n";
        } else {
            SkeletonoperationRunoptimizationall(1, f);
            myFile << mRest_skeleton.joints[K4ABT_JOINT_HAND_LEFT].position.xyz.x;
            myFile << " ";
            myFile << mRest_skeleton.joints[K4ABT_JOINT_HAND_LEFT].position.xyz.y;
            myFile << " ";
            myFile << mRest_skeleton.joints[K4ABT_JOINT_HAND_LEFT].position.xyz.z;
            myFile << "\n";
            SkeletonoperationRunoptimizationall(0, f);
            myFile << mRest_skeleton.joints[K4ABT_JOINT_HAND_LEFT].position.xyz.x;
            myFile << " ";
            myFile << mRest_skeleton.joints[K4ABT_JOINT_HAND_LEFT].position.xyz.y;
            myFile << " ";
            myFile << mRest_skeleton.joints[K4ABT_JOINT_HAND_LEFT].position.xyz.z;
            myFile << "\n";
        }
    }
    myFile.close();
}

QWidget *LP_Plugin_MotionTracking::DockUi()
{
    mWidget = std::make_shared<QWidget>();
    QVBoxLayout *layout = new QVBoxLayout(mWidget.get());

    mLabel = new QLabel("Click 'Run camera' to run the camera \n"
                        "or 'Use model' first if you wish to use 3D model");
    //mLabel2 = new QLabel(" ");
    mButton0 = new QPushButton("Run camera");
    mButton1 = new QPushButton("Get the rest pose of the candidate");
    mButton2 = new QPushButton("Use opt");
    mButton3 = new QPushButton("Use model");
    mButton4 = new QPushButton("Test animation");

    layout->addWidget(mLabel);
    //layout->addWidget(mLabel2);
    layout->addWidget(mButton0);
    layout->addWidget(mButton1);
    layout->addWidget(mButton2);
    layout->addWidget(mButton3);
    layout->addWidget(mButton4);

    mWidget->setLayout(layout);

    // Button
    connect(mButton0, &QPushButton::clicked, [this]() {
        s_isRunning = !s_isRunning;
        QString boolText = s_isRunning ? "true" : "false";
        qDebug() << "Run camera: " << boolText;
        if(s_isRunning){
            RunCamera();
            if(D->mUse_Model){
                mLabel->setText("If use skeleton optimization,\n"
                                "click 'Get the rest pose of the candidate'\n");
            } else {
                mLabel->setText("If use skeleton optimization,\n"
                                "make sure both cameras can capture the whole body,\n"
                                "then click 'Get the rest pose of the candidate'");
            }
        } else {
            gFuture.waitForFinished();
            mImage0 = QImage();
            mImage1 = QImage();
            mImageske = QImage();
            mDImage0 = QImage();
            mDImage1 = QImage();
            mIrImage0 = QImage();
            mIrImage1 = QImage();
        }
    });

    connect(mButton1, &QPushButton::clicked, [this]() {
        if(mAvg_count<15){
            mGetRest = true;
        } else {
            qDebug() << "Already got the rest pose!";
        }
    });

    connect(mButton2, &QPushButton::clicked, [this]() {
        if(mAvg_count<15){
            qDebug() << "Get the rest pose of the candidate first!";
        } else {
            mUse_Opt = !mUse_Opt;
            QString boolText = mUse_Opt ? "true" : "false";
            qDebug() << "Use Opt: " << boolText;
            mLabel->setText("Using skeleton optimization, click again to stop\n"
                            "If test animation,\n"
                            "click 'Test animation'\n");
        }
    });

    connect(mButton3, &QPushButton::clicked, [this]() {
        D->mUse_Model = !D->mUse_Model;
        QString boolText = D->mUse_Model ? "true" : "false";
        qDebug() << "Use Mode: " << boolText;

        if(D->mUse_Model){
            QString filename = QFileDialog::getOpenFileName(
                               nullptr,
                               QObject::tr("Open Obj"),
                               QDir::currentPath(),
                               QObject::tr("Obj files (*.obj)"));
            //qDebug() << filename;
            loadObj(filename.toLocal8Bit().data(), D->mModelVAO, D->mModelF);
        } else {
            D->mInitialized_R_DM = false;

            D->mModel_Nearplane = 1000.0;
            D->mModel_Farplane = 3000.0;

            D->mModelBB_min = QVector3D(0.f, 0.f, 0.f),
            D->mModelBB_max = QVector3D(0.f, 0.f, 0.f);

            D->init_BB = false;
            D->init_modelpos = false;

            D->mModelVAO.clear();
            D->mModelF.clear();

            D->mDBackground = cv::Mat();
            D->mIrBackground = cv::Mat();
            D->mDepth_Map_color0 = QImage();
            D->mDepth_Map_color1 = QImage();
            D->mNormal_Map0 = QImage();
            D->mNormal_Map1 = QImage();
            D->mDepth_Map_dis0 = cv::Mat();
            D->mDepth_Map_dis1 = cv::Mat();
            D->mIr_Map0 = cv::Mat();
            D->mIr_Map1 = cv::Mat();

            D->mProgram_R_DM = nullptr;
            D->mR_DM_FBO = nullptr;
        }
        emit glUpdateRequest();
    });

    connect(mButton4, &QPushButton::clicked, [this]() {
        if(D->mUse_Model && s_isRunning){
            D->mTest_Animation = true;
            QString boolText = D->mTest_Animation ? "true" : "false";
            qDebug() << "Test animation: " << boolText;

            // load time stamps
            QString qs;
            QFile fin("timestamps.csv");
            if(!fin.open(QIODevice::ReadOnly)) {
                qDebug() << "Failed to open file";
                return;
            }
            QTextStream in(&fin);
            bool tmp_f = true;
            int tmp_ft = 0;
            while(!in.atEnd())
            {
                qs = in.readLine();
                auto l = qs.split(',');
                if(tmp_f){
                    tmp_ft = l[0].toInt();
                    tmp_f = false;
                }
                int t_0 = l[0].toInt()-tmp_ft;
                int t_1 = l[1].toInt()-tmp_ft;
                if(t_0>3000){
                    break;
                }
                D->mFrames_time.push_back(t_0);
                if(t_1>3000){
                    break;
                }
                D->mFrames_time.push_back(t_0);
            }
            qDebug() << "Got timestamps! Total frames: " << D->mFrames_time.size();

            mSave_samples = true;
            boolText = mSave_samples ? "true" : "false";
            qDebug() << "Save samples: " << boolText;
            create_paths();
        } else {
            qDebug() << "Cannot test animation, use model and start camera first!";
        }
    });

    return mWidget.get();
}

bool LP_Plugin_MotionTracking::Run()
{
    D = std::make_shared<Model_member>();

    //KAfunctions = std::make_shared<Kinect_Azure_Functions>();

    // rigid_transform_3D_test();

    // qDebug() << "Main thread : " << QThread::currentThreadId();

    return false;
}

LP_Plugin_MotionTracking::~LP_Plugin_MotionTracking()
{
    Q_EMIT glContextRequest([this](){
        delete mProgram_R;
        mProgram_R = nullptr;
    });

    if ( D && D->mCB_Context ) {
        D->mCB_Context->makeCurrent(D->mCB_Surface);

        delete D->mProgram_R_DM;
        D->mProgram_R_DM = nullptr;

        delete D->mR_DM_FBO;
        D->mR_DM_FBO = nullptr;

        D->mCB_Context->doneCurrent();
        delete D->mCB_Context;
        D->mCB_Context = nullptr;
    }

    s_isRunning = false;
    mUse_Opt = false;
    mGetRest = false;
    mInitSKO = false;
    gFuture.waitForFinished();
}

void LP_Plugin_MotionTracking::RunCamera()
{
    gFuture = QtConcurrent::run([this]() {
      uint32_t device_count = k4a_device_get_installed_count();
      qDebug() << "Found " << device_count << "connected devices:";

      std::vector<k4a_device_t> devices(device_count);
      std::vector<k4abt_tracker_t> trackers(device_count);
      sensorCalibrations.resize(device_count);
      std::vector<k4a_transformation_t> k4a_transformations(device_count);

      for (uint8_t dev_ind = 0; dev_ind < device_count; dev_ind++) {
        char* serial_number = NULL;
        size_t serial_number_length = 0;
        k4a_device_t device = nullptr;

        VERIFY(k4a_device_open(dev_ind, &device), "Open K4A Device failed!");

        k4a_device_get_serialnum(device, NULL, &serial_number_length);
        serial_number = (char*)malloc(serial_number_length);
        k4a_device_get_serialnum(device, serial_number, &serial_number_length);

        qDebug() << "dev_ind: " << dev_ind << "\n"
                 << "serialnum: " << serial_number << "\n";

        delete serial_number;

        k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        deviceConfig.camera_fps = inputSettings.FPS;
        deviceConfig.depth_mode = inputSettings.DepthCameraMode;
        deviceConfig.color_format = inputSettings.ColorFormat;
        deviceConfig.color_resolution = inputSettings.ColorResolution;
        deviceConfig.synchronized_images_only = inputSettings.SynchronizedImagesOnly;
        if (dev_ind == 0) {
          //deviceConfig.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
          deviceConfig.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
        } else {
          //deviceConfig.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
          deviceConfig.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
        }

        k4a_calibration_t sensorCalibration;
        VERIFY(k4a_device_start_cameras(device, &deviceConfig),
               "Start K4A cameras failed!");
        // Get calibration information
        VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode,
                                          deviceConfig.color_resolution,
                                          &sensorCalibration),
                                          "Get depth camera calibration failed!");

        mCam_rows = sensorCalibration.color_camera_calibration.resolution_height;
        mCam_cols = sensorCalibration.color_camera_calibration.resolution_width;
        mDepthh = sensorCalibration.depth_camera_calibration.resolution_height;
        mDepthw = sensorCalibration.depth_camera_calibration.resolution_width;

        k4a_transformation_t transformation = k4a_transformation_create(&sensorCalibration);
        if(transformation==nullptr){
            qDebug() << "Create transformation failed!";
        }

        // Create Body Tracker
        k4abt_tracker_t tracker = nullptr;
        k4abt_tracker_configuration_t trackerConfig = K4ABT_TRACKER_CONFIG_DEFAULT;
        trackerConfig.processing_mode = inputSettings.processingMode;
        //trackerConfig.sensor_orientation = K4ABT_SENSOR_ORIENTATION_CLOCKWISE90;

        // trackerConfig.model_path = inputSettings.ModelPath.c_str();

        VERIFY(k4abt_tracker_create(&sensorCalibration, trackerConfig, &tracker),
               "Body tracker initialization failed!");

        devices[dev_ind] = std::move(device);
        trackers[dev_ind] = std::move(tracker);
        sensorCalibrations[dev_ind] = std::move(sensorCalibration);
        k4a_transformations[dev_ind] = std::move(transformation);

        //QThread::msleep(20);
      }

      std::vector<k4a_capture_t> sensorCaptures(device_count);
      std::vector<k4abt_frame_t> bodyFrames(device_count);

      // auto prev0 = std::chrono::steady_clock::now();
      // auto prev1 = std::chrono::steady_clock::now();
      uint32_t fps = 0, fps_counter = 0;
      uint32_t fps2 = 0, fps_counter2 = 0;
      auto start_time = std::chrono::high_resolution_clock::now();
      auto start_time2 = std::chrono::high_resolution_clock::now();
      auto now = std::chrono::high_resolution_clock::now();
      auto now2 = std::chrono::high_resolution_clock::now();
      auto start_time_opt = std::chrono::high_resolution_clock::now();
      auto now_opt = std::chrono::high_resolution_clock::now();

      float _durr = 0.0;
      int opt_times = 0;

      if(mSave_samples){
          create_paths();
      }

//      QString qs;
//      int line = 0;
//      cv::Mat dmat = cv::Mat::zeros(576, 640, CV_16UC1);
//      cv::Mat irmat = cv::Mat::zeros(576, 640, CV_16UC1);
//      QFile fin("dimg.csv");
//      if(!fin.open(QIODevice::ReadOnly)) {
//          qDebug() << "Failed to open file";
//      }
//      QTextStream in(&fin);
//      while(!in.atEnd())
//      {
//          qs = in.readLine();
//          auto l = qs.split(' ');
//          for(int i=0; i<l.size()-1; i++){
//              dmat.at<uint16_t>(line, i) = l[i].toInt();
//          }
//          line++;
//      }
//      fin.close();
//      QFile fin2("irimg.csv");
//      if(!fin2.open(QIODevice::ReadOnly)) {
//          qDebug() << "Failed to open file";
//      }
//      QTextStream in2(&fin2);
//      line = 0;
//      while(!in2.atEnd())
//      {
//          qs = in2.readLine();
//          auto l = qs.split(' ');
//          for(int i=0; i<l.size()-1; i++){
//              irmat.at<uint16_t>(line, i) = l[i].toInt();
//          }
//          line++;
//      }
//      fin2.close();

      while (s_isRunning) {
        for (unsigned int dev_ind = 0; dev_ind < device_count; dev_ind++) {
          k4a_wait_result_t getCaptureResult =
              k4a_device_get_capture(devices[dev_ind], &sensorCaptures[dev_ind],
                                     0); // timeout_in_ms is set to 0
          if (getCaptureResult == K4A_WAIT_RESULT_SUCCEEDED) {
              if(D->mUse_Model){
                  k4a_image_t cimg = k4a_capture_get_color_image(sensorCaptures[dev_ind]);
                  k4a_image_t dimg = k4a_capture_get_depth_image(sensorCaptures[dev_ind]);
                  k4a_image_t irimg = k4a_capture_get_ir_image(sensorCaptures[dev_ind]);
                  if(dev_ind == 0){
                      if(D->mTest_Animation && (D->frame+1) < D->mFrames_time.size()){
                          // timestamps between objs are 2ms
                          int f = 0.5*(D->mFrames_time[D->frame]);
                          QString num;
                          if(f<10){
                              num = QString("00%1.obj").arg(f);
                          } else if(f<100){
                              num = QString("0%1.obj").arg(f);
                          } else {
                              num = QString("%1.obj").arg(f);
                          }
                          QString filename = QString(D->mAnimation_path + num);
                          loadObj(filename.toLocal8Bit().data(), D->mModelVAO, D->mModelF);
                      }

                      cv::Mat mat_dis, mat_ir;
                      mLock.lockForRead();
                      D->mDepth_Map_dis0.copyTo(mat_dis);
                      D->mIr_Map0.copyTo(mat_ir);
                      mLock.unlock();

                      //rotate cvmat
                      //cv::rotate(mat_dis, mat_dis, cv::ROTATE_90_CLOCKWISE);
                      //cv::rotate(mat_ir, mat_ir, cv::ROTATE_90_CLOCKWISE);

                      //conver the image fron CV::mat to k4a::image
                      memcpy(k4a_image_get_buffer(dimg), &mat_dis.ptr<int16_t>(0)[0], mat_dis.rows*mat_dis.cols*sizeof(int16_t));
                      memcpy(k4a_image_get_buffer(irimg), &mat_ir.ptr<int16_t>(0)[0], mat_ir.rows*mat_ir.cols*sizeof(int16_t));

                      //process on color k4a image
                      k4a_image_t transformed_depth_image = NULL;
                      VERIFY(k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                              mCam_cols,
                                              mCam_rows,
                                              mCam_cols * (int)sizeof(uint16_t),
                                              &transformed_depth_image),
                             "Failed to create transformed depth image");
                      VERIFY(k4a_transformation_depth_image_to_color_camera(k4a_transformations[dev_ind],
                                                                            dimg,
                                                                            transformed_depth_image),
                             "Failed to transform depth image to color image!");
                      uint8_t *buffer = k4a_image_get_buffer(transformed_depth_image);
                      cv::Mat transformed_depth_image_mat(mCam_rows, mCam_cols, CV_16UC1, (void *)buffer, cv::Mat::AUTO_STEP);
                      transformed_depth_image_mat = transformed_depth_image_mat/15.0;
                      cv::cvtColor(transformed_depth_image_mat, transformed_depth_image_mat, cv::COLOR_GRAY2RGBA);
                      transformed_depth_image_mat.convertTo(transformed_depth_image_mat, CV_8UC4);
                      memcpy(k4a_image_get_buffer(cimg),
                             &transformed_depth_image_mat.ptr<cv::Vec4b>(0)[0],
                             transformed_depth_image_mat.rows*transformed_depth_image_mat.cols*sizeof(cv::Vec4b));

                      k4a_image_release(transformed_depth_image);
                  } else {
                      if(D->mTest_Animation && (D->frame+1) < D->mFrames_time.size()){
                          // timestamps between objs are 2ms
                          int f = 0.5*(D->mFrames_time[D->frame+1]);
                          QString num;
                          if(f<10){
                              num = QString("00%1.obj").arg(f);
                          } else if(f<100){
                              num = QString("0%1.obj").arg(f);
                          } else {
                              num = QString("%1.obj").arg(f);
                          }
                          QString filename = QString(D->mAnimation_path + num);
                          loadObj(filename.toLocal8Bit().data(), D->mModelVAO, D->mModelF);

                          D->frame += 2;
                      }

                      cv::Mat mat_dis, mat_ir;
                      mLock.lockForRead();
                      D->mDepth_Map_dis1.copyTo(mat_dis);
                      D->mIr_Map1.copyTo(mat_ir);
                      mLock.unlock();

                      //rotate cvmat
                      //cv::rotate(mat_dis, mat_dis, cv::ROTATE_90_CLOCKWISE);
                      //cv::rotate(mat_ir, mat_ir, cv::ROTATE_90_CLOCKWISE);

                      //conver the image fron CV::mat to k4a::image
                      memcpy(k4a_image_get_buffer(dimg), &mat_dis.ptr<int16_t>(0)[0], mat_dis.rows*mat_dis.cols*sizeof(int16_t));
                      memcpy(k4a_image_get_buffer(irimg), &mat_ir.ptr<int16_t>(0)[0], mat_ir.rows*mat_ir.cols*sizeof(int16_t));

                      //process on color k4a image
                      k4a_image_t transformed_depth_image = NULL;
                      VERIFY(k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                              mCam_cols,
                                              mCam_rows,
                                              mCam_cols * (int)sizeof(uint16_t),
                                              &transformed_depth_image),
                             "Failed to create transformed depth image");
                      VERIFY(k4a_transformation_depth_image_to_color_camera(k4a_transformations[dev_ind],
                                                                            dimg,
                                                                            transformed_depth_image),
                             "Failed to transform depth image to color image!");
                      uint8_t *buffer = k4a_image_get_buffer(transformed_depth_image);
                      cv::Mat transformed_depth_image_mat(mCam_rows, mCam_cols, CV_16UC1, (void *)buffer, cv::Mat::AUTO_STEP);
                      transformed_depth_image_mat = transformed_depth_image_mat/15.0;
                      cv::cvtColor(transformed_depth_image_mat, transformed_depth_image_mat, cv::COLOR_GRAY2RGBA);
                      transformed_depth_image_mat.convertTo(transformed_depth_image_mat, CV_8UC4);
                      memcpy(k4a_image_get_buffer(cimg),
                             &transformed_depth_image_mat.ptr<cv::Vec4b>(0)[0],
                             transformed_depth_image_mat.rows*transformed_depth_image_mat.cols*sizeof(cv::Vec4b));

                      k4a_image_release(transformed_depth_image);
                  }
                  k4a_image_release(cimg);
                  k4a_image_release(dimg);
                  k4a_image_release(irimg);
              }

            // CV MAT TO K4A COLOR IMAGE ///////////////////////////////////////////////////////////////////////////////
  //          k4a_image_t cimg = k4a_capture_get_color_image(sensorCaptures[dev_ind]);
  //          int colorh = k4a_image_get_height_pixels(cimg);
  //          int colorw = k4a_image_get_width_pixels(cimg);
  //          qDebug() << colorh << " " << colorw;
  //          //convert the image from k4a::image to CV ::mat
  //          cv::Mat cImg = cv::Mat(k4a_image_get_height_pixels(k4aimg), k4a_image_get_width_pixels(k4aimg), CV_8UC4, k4a_image_get_buffer(k4aimg));
  //          cv::Mat cImg = cv::Mat::zeros(colorh, colorw, CV_8UC4);
  //          //convert the image from CV ::mat to k4a::image
  //          memcpy(k4a_image_get_buffer(cimg), &cImg.ptr<cv::Vec4b>(0)[0], cImg.rows*cImg.cols  *sizeof(cv::Vec4b));
            // CV MAT TO K4A DEPTH IMAGE ///////////////////////////////////////////////////////////////////////////////
  //          k4a_image_t dimg = k4a_capture_get_depth_image(sensorCaptures[dev_ind]);
  //          int depthh = k4a_image_get_height_pixels(dimg);
  //          int depthw = k4a_image_get_width_pixels(dimg);
  //          qDebug() << depthh << " " << depthw;
  //          //convert the image from k4a::image to CV ::mat
  //          //cv::Mat dImg = cv::Mat(k4a_image_get_height_pixels(dimg), k4a_image_get_width_pixels(dimg), CV_16UC1, reinterpret_cast<uint16_t*>(k4a_image_get_buffer(dimg)));
  //          cv::Mat dImg = cv::Mat::zeros(depthh, depthw, CV_16UC1);
  //          //convert the image from CV ::mat to k4a::image
  //          memcpy(k4a_image_get_buffer(dimg), &dImg.ptr<cv::Vec4b>(0)[0], dImg.rows*dImg.cols  *sizeof(cv::Vec4b));
            // /////////////////////////////////////////////////////////////////////////////////////////////////////////

            k4a_wait_result_t queueCaptureResult = k4abt_tracker_enqueue_capture(
                trackers[dev_ind], sensorCaptures[dev_ind], 0);

            // Release the sensor capture once it is no longer needed.
            k4a_capture_release(sensorCaptures[dev_ind]);

            if (queueCaptureResult == K4A_WAIT_RESULT_FAILED) {
              qDebug() << "Error! Add capture to tracker process queue failed!";
              continue;
            }
          } else if (getCaptureResult != K4A_WAIT_RESULT_TIMEOUT) {
            qDebug() << "Get depth capture returned error: " << getCaptureResult;
            continue;
          }

          // Pop Result from Body Tracker
          k4abt_frame_t bodyFrame = nullptr;
          k4a_wait_result_t popFrameResult = k4abt_tracker_pop_result(
              trackers[dev_ind], &bodyFrame, 0); // timeout_in_ms is set to 0

          if (popFrameResult == K4A_WAIT_RESULT_SUCCEEDED) {
            if (bodyFrames[dev_ind] != nullptr) {
              // Release the bodyFrame
              k4abt_frame_release(bodyFrames[dev_ind]);
            }
            bodyFrames[dev_ind] = std::move(bodyFrame);
          }

          if(dev_ind==0){
              QThread::msleep(5);
          }
        }

        if(bodyFrames[0] == nullptr || bodyFrames[1] == nullptr){
            if(bodyFrames[0] != nullptr){
                bodyFrames[0] == nullptr;
                //qDebug() << "Null body frame 0!";
            }
            if(bodyFrames[1] != nullptr){
                bodyFrames[1] == nullptr;
                //qDebug() << "Null body frame 1!";
            }
            continue;
        } else {
            // Skip captures without a color image inside the while loop
            k4a_capture_t originalCapture0 = k4abt_frame_get_capture(bodyFrames[0]);
            k4a_capture_t originalCapture1 = k4abt_frame_get_capture(bodyFrames[1]);
            k4a_image_t image_color0 = k4a_capture_get_color_image(originalCapture0);
            k4a_image_t image_color1 = k4a_capture_get_color_image(originalCapture1);
            if(image_color0 == nullptr || image_color1 == nullptr){
                qDebug() << "Null color images!";
                k4a_capture_release(originalCapture0);
                k4a_capture_release(originalCapture1);
                k4a_image_release(image_color0);
                k4a_image_release(image_color1);
                originalCapture0 = nullptr;
                originalCapture1 = nullptr;
                image_color0 = nullptr;
                image_color1 = nullptr;
                bodyFrames[0] == nullptr;
                bodyFrames[1] == nullptr;
                continue;
            }
        }

        for (unsigned int dev_ind = 0; dev_ind < device_count; dev_ind++) {
            if (mGetRest && dev_ind == 0) {
              mJoint_tmp_num = 0;
              mBone_tmp_num = 0;
            } else {
              joint_b_num = 0;
              bone_b_num = 0;
            }

            if (mUse_Opt) {
              if (dev_ind == 0) {
                joint_a_num = 0;
                bone_a_num = 0;
                got_skea = false;
                //KAfunctions->clear();
              } else {
                joint_b_num = 0;
                bone_b_num = 0;
                got_skeb = false;
              }
            }

            // Visualize
            VisualizeResult(bodyFrames[dev_ind], sensorCalibrations[dev_ind], k4a_transformations[dev_ind], dev_ind);
        }
        //qDebug() << "================================================";

        if (mUse_Opt && got_skea && got_skeb) {
            auto _start = std::chrono::high_resolution_clock::now();
            if (!mInitSKO) {
              SkeletonoperationInit();
              start_time_opt = std::chrono::high_resolution_clock::now();
            } else if(mJoint_Traj.Traj_skesB.size()==mJoint_Traj.TrajFrame_Size) {
              if(mSkeData.cam1_t.size()>1){
                SkeletonoperationRunoptimization(1);
                auto _end = std::chrono::high_resolution_clock::now();
                _durr += std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
                opt_times++;
                //qDebug() << "optimization total time (1): " << _durr;
              }
              SkeletonoperationRunoptimization(0);
              auto _end = std::chrono::high_resolution_clock::now();
              _durr += std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
              opt_times++;
              //qDebug() << "optimization total time (0): " << _durr;
              lastbody_b = tmpbody_b;
              erase_traj();
            }

            now_opt = std::chrono::high_resolution_clock::now();
            float optdurr = std::chrono::duration_cast<std::chrono::milliseconds>(now_opt - start_time_opt).count();
            if (optdurr > 1000) {
              start_time_opt = now_opt;
              qDebug() << "=====================================================\n"
                       << "--SKEOPT FPS: " << optfps_count << "--\n"
                       << "--Average SKEOPT Durration: " << _durr / float(opt_times) << "--\n"
                       << "=====================================================";
              optfps_count = 0;
              _durr = 0.0;
              opt_times = 0;
            }
        } else if(D->mTest_Animation){
            qDebug() << "Skip skeopt.";
            QString boolText = got_skea ? "true" : "false";
            qDebug() << "Got skea: " << got_skea;
            boolText = got_skeb ? "true" : "false";
            qDebug() << "Got skeb: " << got_skeb;
        }

        if(!mSave_samples){
            if(mSkeData.cam0_t.size() > 3){
                mSkeData.cam0_t.erase(mSkeData.cam0_t.begin());
            }
            if(mSkeData.cam1_t.size() > 3){
                mSkeData.cam1_t.erase(mSkeData.cam1_t.begin());
            }
        } else if ((mSkeData.cam0_t.size() > 150 && mSkeData.cam1_t.size() > 150 && !D->mTest_Animation)) {
            Process_AllJointTraj();

            save_samples();

            qDebug() << "GG";
            QThread::msleep(-1);
        }

        if (bodyFrames[0] != nullptr) {
          // for main camera count the FPS and number of capture/enqueue and
          // dequeu in the previous second.
          now = std::chrono::high_resolution_clock::now();
          float durr = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now - start_time)
                           .count();
          if (durr > 1000) {
            start_time = now;
            fps = fps_counter;
            qDebug() << "--FPS: " << fps << "--";
            fps_counter = 0;
          }
          fps_counter++;
          //Release the bodyFrame
          k4abt_frame_release(bodyFrames[0]);
          bodyFrames[0] = nullptr;
        }

        if(device_count>1 && bodyFrames[1] != nullptr){
          // for main camera count the FPS and number of capture/enqueue and
          // dequeu in the previous second.
          now2 = std::chrono::high_resolution_clock::now();
          float durr = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now2 - start_time2)
                           .count();
          if (durr > 1000) {
            start_time2 = now2;
            fps2 = fps_counter2;
            qDebug() << "--FPS2: " << fps2 << "--\n"
                     << "=====================================================";
            fps_counter2 = 0;
          }
          fps_counter2++;

          //Release the bodyFrame
          k4abt_frame_release(bodyFrames[1]);
          bodyFrames[1] = nullptr;
        }
      }


      for (unsigned int dev_ind = 0; dev_ind < device_count; dev_ind++) {
        k4abt_tracker_shutdown(trackers[dev_ind]);
        k4abt_tracker_destroy(trackers[dev_ind]);
        k4a_transformation_destroy(k4a_transformations[dev_ind]);
        k4a_device_stop_cameras(devices[dev_ind]);
        k4a_device_close(devices[dev_ind]);
      }
      return;
    });
}

void LP_Plugin_MotionTracking::VisualizeResult(k4abt_frame_t bodyFrame,
                                               k4a_calibration_t sensorCalibration,
                                               k4a_transformation_t transformation,
                                               int dev_ind) {
  // Obtain original capture that generates the body tracking result
  k4a_capture_t originalCapture = k4abt_frame_get_capture(bodyFrame);
  k4a_image_t colorImage = k4a_capture_get_color_image(originalCapture);
  k4a_image_t depthImage = k4a_capture_get_depth_image(originalCapture);
  k4a_image_t irImage = k4a_capture_get_ir_image(originalCapture);
  //mDepthh = k4a_image_get_height_pixels(depthImage);
  //mDepthw = k4a_image_get_width_pixels(depthImage);

  uint64_t frametime = k4a_image_get_system_timestamp_nsec(colorImage);
  frametime *= 1e-6;

  //qDebug() << "camera " << dev_ind << " frame_time: " << frametime;

  uint8_t *buffer = k4a_image_get_buffer(colorImage);
  uint8_t *Dbuffer = k4a_image_get_buffer(depthImage);
  uint8_t *IRbuffer = k4a_image_get_buffer(irImage);
  cv::Mat tmpColorMat(mCam_rows, mCam_cols, CV_8UC4, (void *)buffer, cv::Mat::AUTO_STEP);
  cv::Mat tmpDImg(mDepthh, mDepthw, CV_16UC1, (void *)Dbuffer, cv::Mat::AUTO_STEP);
  cv::Mat tmpIrImg(mDepthh, mDepthw, CV_16UC1, (void *)IRbuffer, cv::Mat::AUTO_STEP);
  cv::Mat colorMat = tmpColorMat.clone();
  cv::Mat dImg = tmpDImg.clone();
  cv::Mat irImg = tmpIrImg.clone();
  cv::Mat colorMat_ori = colorMat.clone();

//    if(!s && dev_ind==0){
//        std::ofstream myFile("dimg.csv");
//        for(int row=0; row<dImg.rows; row++){
//            for(int col=0; col<dImg.cols; col++){
//                myFile << dImg.at<uint16_t>(row, col);
//                if(col!=dImg.cols-1){
//                  myFile << " ";
//                }
//            }
//            myFile << "\n";
//        }
//        myFile.close();

//        std::ofstream myFile2("irimg.csv");
//        for(int row=0; row<irImg.rows; row++){
//            for(int col=0; col<irImg.cols; col++){
//                myFile2 << irImg.at<uint16_t>(row, col);
//                if(col!=irImg.cols-1){
//                  myFile2 << " ";
//                }
//            }
//            myFile2 << "\n";
//        }
//        myFile2.close();
//        s = true;
//        qDebug() << "Saved background!";
//    }

//  double minVal, maxVal;
//  cv::Point minLoc, maxLoc;
//  cv::minMaxLoc(dImg, &minVal, &maxVal, &minLoc, &maxLoc);
//  dImg = dImg*int((65536.0/maxVal));
//  cv::minMaxLoc(irImg, &minVal, &maxVal, &minLoc, &maxLoc);
//  irImg = irImg*int((65536.0/maxVal));

  int greyscaled_d = 20, greyscaled_ir = 50;
  dImg = dImg*greyscaled_d;
  irImg = irImg*greyscaled_ir;

  cv::putText(colorMat, //target image
              std::to_string(frametime), //text
              cv::Point(10, 80), //top-left position
              cv::FONT_HERSHEY_SIMPLEX,
              3.0,
              cv::Scalar(0, 0, 255, 255), //font color
              4.0);

  //qDebug() << colorMat.rows << " " << colorMat.cols;

  std::vector<Color> pointCloudColors(mDepthw * mDepthh,
                                      {1.f, 1.f, 1.f, 1.f});

  // Read body index map and assign colors
  k4a_image_t bodyIndexMap = k4abt_frame_get_body_index_map(bodyFrame);
  const uint8_t *bodyIndexMapBuffer = k4a_image_get_buffer(bodyIndexMap);
  for (int i = 0; i < mDepthw * mDepthh; i++) {
    uint8_t bodyIndex = bodyIndexMapBuffer[i];
    if (bodyIndex != K4ABT_BODY_INDEX_MAP_BACKGROUND) {
      uint32_t bodyId = k4abt_frame_get_body_id(bodyFrame, bodyIndex);
      pointCloudColors[i] = g_bodyColors[bodyId % g_bodyColors.size()];
    }
  }
  k4a_image_release(bodyIndexMap);

  // Visualize the skeleton data
  uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
  for (uint32_t i = 0; i < numBodies; i++) {
    k4abt_body_t body;
    VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton),
           "Get skeleton from body frame failed!");
    body.id = k4abt_frame_get_body_id(bodyFrame, i);

    // Assign the correct color based on the body id
    Color color = g_bodyColors[body.id % g_bodyColors.size()];
    color.a = 0.4f;
    Color lowConfidenceColor = color;
    lowConfidenceColor.a = 0.1f;

    // Visualize joints
    for (int joint = 0; joint < static_cast<int>(K4ABT_JOINT_COUNT); joint++) {
      // 13 	ELBOW_RIGHT
      // 14 	WRIST_RIGHT
      // 15 	HAND_RIGHT
      // 16 	HANDTIP_RIGHT
      // 17 	THUMB_RIGHT
      if (body.skeleton.joints[joint].confidence_level >=
          K4ABT_JOINT_CONFIDENCE_LOW) {
          //K4ABT_JOINT_CONFIDENCE_MEDIUM) {
        const k4a_float3_t &jointPosition =
            body.skeleton.joints[joint].position;
        // const k4a_quaternion_t &jointOrientation =
        // body.skeleton.joints[joint].orientation;
        if (dev_ind == 0) {
          if (mGetRest) {
            mJoint_tmp_num++;
          }
          if (mUse_Opt) {
            joint_a_num++;
          }
        } else {
          if (mGetRest || mUse_Opt) {
            joint_b_num++;
          }
        }

        k4a_float2_t joint2d;
        int valid;
        k4a_calibration_3d_to_2d(&sensorCalibration, &jointPosition,
                                 K4A_CALIBRATION_TYPE_DEPTH,
                                 K4A_CALIBRATION_TYPE_COLOR, &joint2d, &valid);
        if (valid) {
          Color joint_color = body.skeleton.joints[joint].confidence_level >=
                                      K4ABT_JOINT_CONFIDENCE_MEDIUM
                                  ? color
                                  : lowConfidenceColor;
          cv::circle(colorMat, cv::Point(int(joint2d.xy.x), int(joint2d.xy.y)),
                     6,
                     cv::Scalar(joint_color.b * 255, joint_color.g * 255,
                                joint_color.r * 255, joint_color.a * 255),
                     cv::FILLED, cv::LINE_8);
        }
      }
    }

    // Visualize bones
    for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++) {
      k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
      k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;

      if (body.skeleton.joints[joint1].confidence_level >=
              K4ABT_JOINT_CONFIDENCE_LOW &&
              //K4ABT_JOINT_CONFIDENCE_MEDIUM &&
          body.skeleton.joints[joint2].confidence_level >=
              K4ABT_JOINT_CONFIDENCE_LOW) {
              //K4ABT_JOINT_CONFIDENCE_MEDIUM) {
        bool confidentBone = body.skeleton.joints[joint1].confidence_level >=
                                 K4ABT_JOINT_CONFIDENCE_LOW &&
                                 //K4ABT_JOINT_CONFIDENCE_MEDIUM &&
                             body.skeleton.joints[joint2].confidence_level >=
                                 K4ABT_JOINT_CONFIDENCE_LOW;
                                 //K4ABT_JOINT_CONFIDENCE_MEDIUM;
        const k4a_float3_t &joint1Position =
            body.skeleton.joints[joint1].position;
        const k4a_float3_t &joint2Position =
            body.skeleton.joints[joint2].position;
        k4a_float2_t joint2d_1, joint2d_2;
        int valid1, valid2;
        k4a_calibration_3d_to_2d(
            &sensorCalibration, &joint1Position, K4A_CALIBRATION_TYPE_DEPTH,
            K4A_CALIBRATION_TYPE_COLOR, &joint2d_1, &valid1);
        k4a_calibration_3d_to_2d(
            &sensorCalibration, &joint2Position, K4A_CALIBRATION_TYPE_DEPTH,
            K4A_CALIBRATION_TYPE_COLOR, &joint2d_2, &valid2);
        if (valid1 && valid2) {
          Color bone_color = confidentBone ? color : lowConfidenceColor;
          cv::line(colorMat,
                   cv::Point(int(joint2d_1.xy.x), int(joint2d_1.xy.y)),
                   cv::Point(int(joint2d_2.xy.x), int(joint2d_2.xy.y)),
                   cv::Scalar(bone_color.b * 255, bone_color.g * 255,
                              bone_color.r * 255, bone_color.a * 255),
                   5, cv::LINE_4);
        }

        if (dev_ind == 0) {
          if (mGetRest) {
            mBone_tmp_num++;
          }
          if (mUse_Opt) {
            bone_a_num++;
          }
        } else {
          if (mGetRest || mUse_Opt) {
            bone_b_num++;
          }
        }
      }
    }

    if (mGetRest && mJoint_tmp_num == 32 && mBone_tmp_num == 31) {
        if(dev_ind == 0){
            static bool once = [this](){
                std::cout << "Doing rigid transformation, please do not move" << std::endl;
                mKinectA_pts = cv::Mat::zeros(3, 32, CV_64F);
                mKinectB_pts = cv::Mat::zeros(3, 32, CV_64F);
                return true;
            }();
            mRest_skeleton = body.skeleton;
        } else if (joint_b_num == 32 && bone_b_num == 31){
            for(int col=0; col<32; col++){
                const k4a_float3_t &jointPositiona = mRest_skeleton.joints[col].position;
                const k4a_float3_t &jointPositionb = body.skeleton.joints[col].position;
                cv::Mat tmpa = (cv::Mat_<double>(3, 1) << jointPositiona.xyz.x,
                                                          jointPositiona.xyz.y,
                                                          jointPositiona.xyz.z);
                cv::Mat tmpb = (cv::Mat_<double>(3, 1) << jointPositionb.xyz.x,
                                                          jointPositionb.xyz.y,
                                                          jointPositionb.xyz.z);
                mKinectA_pts.col(col) = mKinectA_pts.col(col) + tmpa;
                mKinectB_pts.col(col) = mKinectB_pts.col(col) + tmpb;
            }
            mAvg_count++;
            qDebug() << "[ " << mAvg_count << " / 15 ]";
            if(mAvg_count == 15){
                mKinectA_pts = mKinectA_pts / double(mAvg_count);
                mKinectB_pts = mKinectB_pts / double(mAvg_count);
                mRet = rigid_transform_3D(mKinectB_pts.clone(), mKinectA_pts.clone());
                for(int joint=0; joint<mJoint_tmp_num; joint++){
                    mRest_skeleton.joints[joint].position.xyz.x = mKinectA_pts.at<double>(0, joint);
                    mRest_skeleton.joints[joint].position.xyz.y = mKinectA_pts.at<double>(1, joint);
                    mRest_skeleton.joints[joint].position.xyz.z = mKinectA_pts.at<double>(2, joint);
                }
                mGetRest = false;
                qDebug() << "Got rest pose!";
                mLabel->setText("Click 'Use opt' to use skeleton optimization");

                // Find the root mean squared error
                cv::Mat Aligned_skeb =  mRet.R * mKinectB_pts;
                for(int col=0; col<Aligned_skeb.cols; col++){
                    Aligned_skeb.col(col) = Aligned_skeb.col(col) + mRet.t;
                }
                cv::Mat err = Aligned_skeb - mKinectA_pts;
                cv::pow(err, 2, err);
                double _err = cv::sum(err)[0];
                double rmse = sqrt(_err/double(mJoint_tmp_num));
                std::cout << "RMSE:\n" << rmse << std::endl;

                if(mSave_samples || D->mUse_Model){
                    save_Rt(Aligned_skeb);
                }
            }
        }
    }

    if (mUse_Opt) {
      int n = D->frame/2;
      if (dev_ind == 0) {
//          if(!KAfunctions->point_cloud_color_to_depth(transformation, depthImage, colorImage, dev_ind)){
//              qDebug() << "Failed to get point cloud!";
//          }

          if(joint_a_num == 32 && bone_a_num == 31){
              mTmpFrametime = frametime;
              tmpbody_a = body.skeleton;
              got_skea = true;
              //qDebug() << "Got a pose!";
          }
      } else if(got_skea && joint_b_num == 32 && bone_b_num == 31) {
          cv::Mat tmp_mat = cv::Mat::zeros(3, 32, CV_64F);
            for(int col=0; col<32; col++){
              const k4a_float3_t &jointPosition = body.skeleton.joints[col].position;
              cv::Mat tmp = (cv::Mat_<double>(3, 1) << jointPosition.xyz.x,
                                                       jointPosition.xyz.y,
                                                       jointPosition.xyz.z);
              tmp_mat.col(col) = tmp_mat.col(col) + tmp;
          }
          cv::Mat Aligned_skeb =  mRet.R * tmp_mat;
          for(int col=0; col<Aligned_skeb.cols; col++){
              Aligned_skeb.col(col) = Aligned_skeb.col(col) + mRet.t;
          }

          for(int joint=0; joint<joint_b_num; joint++){
              tmpbody_b.joints[joint].position.xyz.x = Aligned_skeb.at<double>(0, joint);
              tmpbody_b.joints[joint].position.xyz.y = Aligned_skeb.at<double>(1, joint);
              tmpbody_b.joints[joint].position.xyz.z = Aligned_skeb.at<double>(2, joint);
          }
          got_skeb = true;
          //qDebug() << "Got b pose!";

          mJoint_Traj.Traj_timestampsA.push_back(mTmpFrametime);
          mJoint_Traj.Traj_timestampsB.push_back(frametime);
          mJoint_Traj.Traj_skesB.push_back(tmpbody_b);
          mSkeData.skes1ori.push_back(tmpbody_b);

          if(mInitSKO && mJoint_Traj.Traj_skesB.size()==mJoint_Traj.TrajFrame_Size){
              Process_JointTraj();
              uint64_t adjusted_tB = 0.5 * (mJoint_Traj.Traj_timestampsA[2] + mJoint_Traj.Traj_timestampsA[3]);
              //qDebug() << "adjusted_tB: " << adjusted_tB;

              // Use the third point of trajectory
              // 1 2 "3" 4 5
              static bool once = [this](){
                  mSkeData.StartTime = mJoint_Traj.Traj_timestampsA[2];
                  return true;
              }();
              mSkeData.cam0_t_ori.push_back(mJoint_Traj.Traj_timestampsA[2]);
              mSkeData.cam1_t_ori.push_back(mJoint_Traj.Traj_timestampsB[2]);
              mSkeData.cam0_t.push_back(mJoint_Traj.Traj_timestampsA[2] - mSkeData.StartTime);
              mSkeData.cam1_t.push_back(adjusted_tB - mSkeData.StartTime);
              //qDebug() << "cam0_t: " << mSkeData.cam0_t.back();
              //qDebug() << "cam1_t: " << mSkeData.cam1_t.back();

              if(mSave_samples){
                  int n0 = mSkeData.cam0_t.size()-1;
                  int n1 = mSkeData.cam1_t.size()-1;
                  if(D->mTest_Animation){
                      n0 = n;
                      n1 = n;
                  }
                  mSkeData.colorMats0.push_back(mTmpMatA.clone());
                  mSkeData.colorMats1.push_back(mTmpMatB.clone());
                  mSkeData.colorMatoris0.push_back(mTmpMatOriA.clone());
                  mSkeData.colorMatoris1.push_back(mTmpMatOriB.clone());
                  mSkeData.irMats0.push_back(mTmpirMatA.clone()/greyscaled_ir);
                  mSkeData.irMats1.push_back(mTmpirMatB.clone()/greyscaled_ir);
                  mSkeData.skes0.push_back(tmpbody_a);
                  mSkeData.skes1.push_back(tmpbody_b);

//                  if(!D->mUse_Model) {
//                      // Get point cloud
//                      if(!KAfunctions->point_cloud_color_to_depth(transformation, depthImage, colorImage, dev_ind)){
//                          qDebug() << "Failed to get point cloud!";
//                      }
//                      KAfunctions->point_cloud_image0[n0] = std::move(KAfunctions->tmpPoint_cloud_image0);
//                      KAfunctions->point_cloud_image1[n1] = std::move(KAfunctions->tmpPoint_cloud_image1);
//                      KAfunctions->transformed_color_image0[n0] = std::move(KAfunctions->tmpTransformed_color_image0);
//                      KAfunctions->transformed_color_image1[n1] = std::move(KAfunctions->tmpTransformed_color_image1);
//                  }
              }
          }
      } else if (mInitSKO && D->mTest_Animation){
          qDebug() << "joint_b_num: " << joint_b_num << "\n"
                   << "bone_b_num: " << bone_b_num;

          mSkeData.colorMats0.push_back(mTmpMatA.clone());
          mSkeData.colorMats1.push_back(mTmpMatB.clone());
          mSkeData.colorMatoris0.push_back(mTmpMatOriA.clone());
          mSkeData.colorMatoris1.push_back(mTmpMatOriB.clone());
          mSkeData.irMats0.push_back(mTmpirMatA.clone()/greyscaled_ir);
          mSkeData.irMats1.push_back(mTmpirMatB.clone()/greyscaled_ir);
      } else if (!got_skea) {
          qDebug() << "Can not get skeleton A!\n"
                   << "joint_a_num: " << joint_a_num << "\n"
                   << "bone_a_num: " << bone_a_num;
      } else if (got_skea) {
          qDebug() << "Can not get skeleton B!\n"
                   << "joint_b_num: " << joint_b_num << "\n"
                   << "bone_b_num: " << bone_b_num;
      }
    }
  }

  if(mUse_Opt){
      if(!init_tmpmat){
          if(dev_ind == 0 && joint_a_num == 32 && bone_a_num == 31){
              mTmpMatA_Last = colorMat.clone();
              mTmpMatOriA_Last = colorMat_ori.clone();
              mTmpirMatA_Last = irImg.clone();
          } else if (got_skea && joint_b_num == 32 && bone_b_num == 31) {
              mTmpMatB_Last = colorMat.clone();
              mTmpMatOriB_Last = colorMat_ori.clone();
              mTmpirMatB_Last = irImg.clone();
              init_tmpmat = true;
          }
      } else {
          if(dev_ind == 0 && joint_a_num == 32 && bone_a_num == 31){
              mTmpMatA = mTmpMatA_Last.clone();
              mTmpMatOriA = mTmpMatOriA_Last.clone();
              mTmpirMatA = mTmpirMatA_Last.clone();
              mTmpMatA_Last = colorMat.clone();
              mTmpMatOriA_Last = colorMat_ori.clone();
              mTmpirMatA_Last = irImg.clone();
          } else if (got_skea && joint_b_num == 32 && bone_b_num == 31) {
              mTmpMatB = mTmpMatB_Last.clone();
              mTmpMatOriB = mTmpMatOriB_Last.clone();
              mTmpirMatB = mTmpirMatB_Last.clone();
              mTmpMatB_Last = colorMat.clone();
              mTmpMatOriB_Last = colorMat_ori.clone();
              mTmpirMatB_Last = irImg.clone();
          }
      }
  }

  if(dev_ind == 0){
      mImage0 = QImage((uchar *)colorMat.data, colorMat.cols, colorMat.rows, colorMat.step, QImage::Format_ARGB32).copy();
      mDImage0 = QImage((uchar *)dImg.data, dImg.cols, dImg.rows, dImg.step, QImage::Format_Grayscale16).copy();
      mIrImage0 = QImage((uchar *)irImg.data, irImg.cols, irImg.rows, irImg.step, QImage::Format_Grayscale16).copy();
  } else {
      mImage1 = QImage((uchar *)colorMat.data, colorMat.cols, colorMat.rows, colorMat.step, QImage::Format_ARGB32).copy();
      mDImage1 = QImage((uchar *)dImg.data, dImg.cols, dImg.rows, dImg.step, QImage::Format_Grayscale16).copy();
      mIrImage1 = QImage((uchar *)irImg.data, irImg.cols, irImg.rows, irImg.step, QImage::Format_Grayscale16).copy();
  }
  emit glUpdateRequest();

  colorMat.release();
  dImg.release();
  irImg.release();

  k4a_capture_release(originalCapture);
  k4a_image_release(colorImage);
  k4a_image_release(depthImage);
  k4a_image_release(irImage);
}

void LP_Plugin_MotionTracking::SkeletonoperationInit() {
  k4abt_skeleton_t *templateModel = NULL, *model1 = NULL,
                   *model2 = NULL; // Objects storing skeleton data
  templateModel = &mRest_skeleton; // a template storing the Rest pose of the candidate
  model1 = &tmpbody_a; // model1: skeleton data from Kinect A
  model2 = &tmpbody_b; // model2: skeleton data from Kinect B
  // check the validity of skeletons captured from
  // Kinect///////////////////////////////////////
  if (!templateModel || !model1 || !model2) {
    qDebug() << "models are not inputted enough!";
    return;
  }
  if (mJoint_tmp_num != joint_a_num || mJoint_tmp_num != joint_b_num) {
    qDebug() << "joint number are not the same!";
    return;
  }
  if (mBone_tmp_num != bone_a_num || mBone_tmp_num != bone_b_num) {
    qDebug() << "bone number are not the same!";
    return;
  }
  /////////////////////////////////////////////////////////////////////////////////
  int jointNum = mJoint_tmp_num; // number of joint in skeleton
  int boneNum = mBone_tmp_num;   // number of bone in skeleton
  int modelNum = 2;              // number of skeletons or number of cameras
  //qDebug() << jointNum << " " << boneNum << " " << modelNum;

  if (skeOpt == NULL) {
      skeOpt = std::make_shared<SkeletonOpt>(modelNum, jointNum, boneNum);
      //skeOpt = new SkeletonOpt(modelNum, jointNum, boneNum); // create the object
  }

  ////initialize array containing points
  /// information////////////////////////////////////
  for (int joint = 0; joint < jointNum; joint++) {
    const k4a_float3_t &jointPosition1 = model1->joints[joint].position;
    const k4a_float3_t &jointPosition2 = model2->joints[joint].position;
    double x1, y1, z1, x2, y2, z2;
    x1 = jointPosition1.xyz.x;
    y1 = jointPosition1.xyz.y;
    z1 = jointPosition1.xyz.z;
    x2 = jointPosition2.xyz.x;
    y2 = jointPosition2.xyz.y;
    z2 = jointPosition2.xyz.z;
    if (!skeOpt->InsertNode(0, joint, x1, y1, z1)) {
      qDebug() << "insert failed!";
      return;
    }
    if (!skeOpt->InsertNode(1, joint, x2, y2, z2)) {
      qDebug() << "insert failed!";
      return;
    }
  }

  // set edges and length
  // constrain/////////////////////////////////////////////////////////////
  for (int bone = 0; bone < boneNum; bone++) {
    k4abt_joint_id_t n1 = g_boneList[bone].first;
    k4abt_joint_id_t n2 = g_boneList[bone].second;

    double edg_length = sqrt(pow(templateModel->joints[n2].position.xyz.x -
                                     templateModel->joints[n1].position.xyz.x,
                                 2) +
                             pow(templateModel->joints[n2].position.xyz.y -
                                     templateModel->joints[n1].position.xyz.y,
                                 2) +
                             pow(templateModel->joints[n2].position.xyz.z -
                                     templateModel->joints[n1].position.xyz.z,
                                 2));

    if (!skeOpt->InsertEdge(bone, n1, n2)) {
      qDebug() << "edge insertion failed!\n";
      return;
    }
    if (!skeOpt->SetLengthConstrain(bone, edg_length)) {
      qDebug() << "set length constrain failed!\n";
      return;
    }
  }

  // set initial value from
  // template////////////////////////////////////////////////////////////
  for (int joint = 0; joint < jointNum; joint++) {
    const k4a_float3_t &jointPosition = templateModel->joints[joint].position;
    double x, y, z;
    x = jointPosition.xyz.x;
    y = jointPosition.xyz.y;
    z = jointPosition.xyz.z;

    if (!skeOpt->SetInitialVertexPosition(joint, x, y, z)) {
      qDebug() << "set initial position failed!\n";
      return;
    }
  }

  // set
  // weighting////////////////////////////////////////////////////////////////////////////
  for (int m = 0; m < modelNum; m++) {
    for (int vdx = 0; vdx < jointNum; vdx++) {
      double weighting = 1.0 / modelNum;
      for (int i = 0; i < 3; i++) {
        if (!skeOpt->SetModelVertexWeight(m, vdx, i, weighting)) {
          qDebug() << "set model weighting failed!\n";
          return;
        }
      }
    }
  }

  // Run first time to move point/////////////////////////
  skeOpt->Run(1, false);
  /////////////////////////////////////////////////////

  // get back the
  // result////////////////////////////////////////////////////////////////////
  for (int joint = 0; joint < jointNum; joint++) {
    double x, y, z;
    if (!skeOpt->GetVertexPosition(joint, x, y, z)) {
      qDebug() << "get vertex position failed!";
      return;
    }

    templateModel->joints[joint].position.xyz.x = x;
    templateModel->joints[joint].position.xyz.y = y;
    templateModel->joints[joint].position.xyz.z = z;
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  weighting_a.resize(mJoint_tmp_num);
  weighting_b.resize(mJoint_tmp_num);
  mInitSKO = true;
  qDebug() << "SkeletonOpt init";
}

void LP_Plugin_MotionTracking::SkeletonoperationRunoptimization(int use_cam) {
  k4abt_skeleton_t *templateModel = NULL, *model1 = NULL,
                   *model2 = NULL; // Objects storing skeleton data
  templateModel = &mRest_skeleton; // a template storing the Rest pose of the candidate
//  model1 = &tmpbody_a; // model1: skeleton data from Kinect A
//  if(use_cam == 0){
//      model2 = &tmpbody_b; // model2: skeleton data from Kinect B
//  } else {
//      model2 = &lastbody_b;// model2: last skeleton data from Kinect B
//  }
  if(use_cam == 0){
      model1 = &tmpbody_a; // model1: skeleton data from Kinect A
      model2 = &tmpbody_b; // model2: skeleton data from Kinect B
  } else {
      model1 = &lastbody_b;// model1: last skeleton data from Kinect B
      model2 = &tmpbody_a; // model2: skeleton data from Kinect A
  }

  // check the validity of skeletons captured from
  // Kinect///////////////////////////////////////
  if (!templateModel || !model1 || !model2) {
    qDebug() << "models are not inputted enough!";
    return;
  }
  if (mJoint_tmp_num != joint_a_num || mJoint_tmp_num != joint_b_num) {
    qDebug() << "joint number are not the same!";
    return;
  }
  if (mBone_tmp_num != bone_a_num || mBone_tmp_num != bone_b_num) {
    qDebug() << "bone number are not the same!";
    return;
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////
  int jointNum = mJoint_tmp_num; // number of joint in skeleton
  //int boneNum = mBone_tmp_num;   // number of bone in skeleton
  int modelNum = 2; // number of skeletons or number of cameras

  ////initialize array containing points
  /// information////////////////////////////////////
  //k4abt_joint_t **tmparray = new k4abt_joint_t *[jointNum];
  k4abt_joint_t **nodearray1 = new k4abt_joint_t *[jointNum];
  k4abt_joint_t **nodearray2 = new k4abt_joint_t *[jointNum];

  for (int joint = 0; joint < jointNum; joint++) {
    k4abt_joint_t *node1 = &model1->joints[joint];
    nodearray1[joint] = node1; // put all the data into the array

    k4abt_joint_t *node2 = &model2->joints[joint];
    nodearray2[joint] = node2; // put all the data into the array
  }
  //////////////////////////////////////////////////////////////////////////////////////
  // function solving inconsistency of skeletons : Referring to Chapter 3 of
  // paper
  checking_estpoint_new(nodearray1, nodearray2, use_cam);
  //////////////////////////////////////////////////////////////////////////////////////

  // input the data of points in 2 skeletons into SkeletonOpt
  // Object//////////////////////
  // set initial value from
  // template////////////////////////////////////////////////////////////
  for (int joint = 0; joint < jointNum; joint++) {
    k4abt_joint_t *joint1 = &model1->joints[joint];
    k4abt_joint_t *joint2 = &model2->joints[joint];
    k4abt_joint_t *node = &templateModel->joints[joint];
    //tmparray[joint] = node;

    double x1, y1, z1, x2, y2, z2, x3, y3, z3;
    x1 = joint1->position.xyz.x;
    y1 = joint1->position.xyz.y;
    z1 = joint1->position.xyz.z;
    x2 = joint2->position.xyz.x;
    y2 = joint2->position.xyz.y;
    z2 = joint2->position.xyz.z;
    x3 = node->position.xyz.x;
    y3 = node->position.xyz.y;
    z3 = node->position.xyz.z;
    if (!skeOpt->InsertNode(0, joint, x1, y1, z1)) {
      qDebug() << "insert failed!\n";
      return;
    }
    if (!skeOpt->InsertNode(1, joint, x2, y2, z2)) {
      qDebug() << "insert failed!\n";
      return;
    }
    if (!skeOpt->SetInitialVertexPosition(joint, x3, y3, z3)) {
      qDebug() << "set initial position failed!\n";
      return;
    }
  }
  //////////////////////////////////////////////////////////////////////////////////

  // set weighting;
  for (int m = 0; m < modelNum; m++) {
    for (int vdx = 0; vdx < jointNum; vdx++) {
      double weighting = 1.0 / modelNum;

      if (m == 0) {
        weighting = weighting_a[vdx];
      } else if (m == 1) {
        weighting = weighting_b[vdx];
      }

      for (int i = 0; i < 3; i++) {
          if (!skeOpt->SetModelVertexWeight(m, vdx, i, weighting)) {
            qDebug() << "set model weighting failed!\n";
            return;
          }
      }
    }
  }
  delete[] nodearray1;
  delete[] nodearray2;
  //delete[] tmparray;
  // ready to Run ///////////////////////////////////////////

  skeOpt->Run(50, true); // Run the program

  // get back the result/////////////////////////////////////////////////////////////////////////////
  for (int joint = 0; joint < jointNum; joint++) {
    double x, y, z;

    if (!skeOpt->GetVertexPosition(joint, x, y, z)) {
      qDebug() << "get vertex position failed!\n";
      return;
    }

    templateModel->joints[joint].position.xyz.x = x;
    templateModel->joints[joint].position.xyz.y = y;
    templateModel->joints[joint].position.xyz.z = z;
  }

  cv::Mat colorMat;
  k4abt_skeleton_t draw_ske = *templateModel;
  colorMat = mTmpMatOriA.clone();

  // Assign the correct color based on the body id
  Color color = g_bodyColors[0];
  color.a = 0.4f;

  // Visualize joints
  for (int joint = 0; joint < jointNum; joint++) {
    const k4a_float3_t &jointPosition = draw_ske.joints[joint].position;

    k4a_float2_t joint2d;
    int valid;
    k4a_calibration_3d_to_2d(&sensorCalibrations[0], &jointPosition,
                             K4A_CALIBRATION_TYPE_DEPTH,
                             K4A_CALIBRATION_TYPE_COLOR, &joint2d, &valid);
    cv::Point Point2d = cv::Point(int(joint2d.xy.x), int(joint2d.xy.y));
    if (valid) {
      Color joint_color = color;
//      if(use_cam==0){
//          if(joint==6){
//              mSkeData.ELBOW_LEFT0.push_back(Point2d);
//          } else if(joint==7){
//              mSkeData.WRIST_LEFT0.push_back(Point2d);
//          }
//      } else {
//          if(joint==6){
//              mSkeData.ELBOW_LEFT1.push_back(Point2d);
//          } else if(joint==7){
//              mSkeData.WRIST_LEFT1.push_back(Point2d);
//          }
//      }

      cv::circle(colorMat, Point2d, 6,
                 cv::Scalar(joint_color.b * 255, joint_color.g * 255,
                            joint_color.r * 255, joint_color.a * 255),
                 cv::FILLED, cv::LINE_8);
    }
  }

  // Visualize bones
  for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++) {
    k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
    k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;

    const k4a_float3_t &joint1Position = draw_ske.joints[joint1].position;
    const k4a_float3_t &joint2Position = draw_ske.joints[joint2].position;
    k4a_float2_t joint2d_1, joint2d_2;
    int valid1, valid2;
    k4a_calibration_3d_to_2d(&sensorCalibrations[0], &joint1Position,
                             K4A_CALIBRATION_TYPE_DEPTH,
                             K4A_CALIBRATION_TYPE_COLOR, &joint2d_1, &valid1);
    k4a_calibration_3d_to_2d(&sensorCalibrations[0], &joint2Position,
                             K4A_CALIBRATION_TYPE_DEPTH,
                             K4A_CALIBRATION_TYPE_COLOR, &joint2d_2, &valid2);
    if (valid1 && valid2) {
      Color bone_color = color;
      cv::line(colorMat, cv::Point(int(joint2d_1.xy.x), int(joint2d_1.xy.y)),
               cv::Point(int(joint2d_2.xy.x), int(joint2d_2.xy.y)),
               cv::Scalar(bone_color.b * 255, bone_color.g * 255,
                          bone_color.r * 255, bone_color.a * 255),
               5, cv::LINE_4);
    }
  }

  std::string timetext;
  if(use_cam == 0){
      timetext = std::to_string(mSkeData.cam0_t.back());
  } else {
      timetext = std::to_string(mSkeData.cam1_t[mSkeData.cam1_t.size()-2]);
  }

  cv::putText(colorMat, //target image
              timetext, //text
              cv::Point(10, 80), //top-left position
              cv::FONT_HERSHEY_SIMPLEX,
              3.0,
              cv::Scalar(0, 0, 255, 255), //font color
              4.0);

  if(mSave_samples){
      std::vector<k4a_float3_t> tmp_jointposi;
      for(int j=0; j<jointNum;j++){
          tmp_jointposi.push_back(templateModel->joints[j].position);
      }
      mSkeData.Joint_posis.push_back(tmp_jointposi);

      if(mSkeData.Timestamps.size()==0){
          mSkeData.Timestamps.push_back(0.0);
      } else {
          std::pair<uint64_t, uint64_t> m =
                  use_cam ? std::minmax(mSkeData.cam0_t[mSkeData.cam0_t.size()-2], mSkeData.cam1_t[mSkeData.cam1_t.size()-2]) : std::minmax(mSkeData.cam0_t.back(), mSkeData.cam1_t[mSkeData.cam1_t.size()-2]);
          mSkeData.Timestamps.push_back(float(m.second-m.first)*0.001);
      }

      int n0 = mSkeData.cam0_t.size()-1;
      int n1 = mSkeData.cam1_t.size()-2;
      if(D->mTest_Animation){
          n0 = D->frame/2;
          n1 = D->frame/2;
      }
      if(use_cam == 0){
          mSkeData.optskeMats0.push_back(colorMat);
          mSkeData.optskes0.push_back(*templateModel);
      } else {
          mSkeData.optskeMats1.push_back(colorMat);
          mSkeData.optskes1.push_back(*templateModel);
      }
  }

  mImageske = QImage((uchar *)colorMat.data, colorMat.cols, colorMat.rows, colorMat.step, QImage::Format_ARGB32).copy();
  emit glUpdateRequest();

  colorMat.release();
  optfps_count++;
}

void LP_Plugin_MotionTracking::SkeletonoperationRunoptimizationall(int use_cam, int frame) {
  k4abt_skeleton_t *templateModel = NULL, *model1 = NULL,
                   *model2 = NULL; // Objects storing skeleton data

  templateModel = &mRest_skeleton; // a template storing the Rest pose of the candidate
  if(use_cam == 0){
      model1 = &mSkeData.skes0[frame]; // model1: skeleton data from Kinect A
      model2 = &bodyb_all[frame]; // model2: skeleton data from Kinect B
  } else {
      model1 = &bodyb_all[frame-1];// model1: last skeleton data from Kinect B
      model2 = &mSkeData.skes0[frame]; // model2: skeleton data from Kinect A
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  int jointNum = mJoint_tmp_num; // number of joint in skeleton
  int modelNum = 2; // number of skeletons or number of cameras

  ////initialize array containing points
  /// information////////////////////////////////////
  k4abt_joint_t **nodearray1 = new k4abt_joint_t *[jointNum];
  k4abt_joint_t **nodearray2 = new k4abt_joint_t *[jointNum];

  for (int joint = 0; joint < jointNum; joint++) {
    k4abt_joint_t *node1 = &model1->joints[joint];
    nodearray1[joint] = node1; // put all the data into the array

    k4abt_joint_t *node2 = &model2->joints[joint];
    nodearray2[joint] = node2; // put all the data into the array
  }
  //////////////////////////////////////////////////////////////////////////////////////
  // function solving inconsistency of skeletons : Referring to Chapter 3 of
  // paper
  checking_estpoint_new(nodearray1, nodearray2, use_cam);
  //////////////////////////////////////////////////////////////////////////////////////

  // input the data of points in 2 skeletons into SkeletonOpt
  // Object//////////////////////
  // set initial value from
  // template////////////////////////////////////////////////////////////
  for (int joint = 0; joint < jointNum; joint++) {
    k4abt_joint_t *joint1 = &model1->joints[joint];
    k4abt_joint_t *joint2 = &model2->joints[joint];
    k4abt_joint_t *node = &templateModel->joints[joint];
    //tmparray[joint] = node;

    double x1, y1, z1, x2, y2, z2, x3, y3, z3;
    x1 = joint1->position.xyz.x;
    y1 = joint1->position.xyz.y;
    z1 = joint1->position.xyz.z;
    x2 = joint2->position.xyz.x;
    y2 = joint2->position.xyz.y;
    z2 = joint2->position.xyz.z;
    x3 = node->position.xyz.x;
    y3 = node->position.xyz.y;
    z3 = node->position.xyz.z;
    if (!skeOpt->InsertNode(0, joint, x1, y1, z1)) {
      qDebug() << "insert failed!\n";
      return;
    }
    if (!skeOpt->InsertNode(1, joint, x2, y2, z2)) {
      qDebug() << "insert failed!\n";
      return;
    }
    if (!skeOpt->SetInitialVertexPosition(joint, x3, y3, z3)) {
      qDebug() << "set initial position failed!\n";
      return;
    }
  }
  //////////////////////////////////////////////////////////////////////////////////

  // set weighting;
  for (int m = 0; m < modelNum; m++) {
    for (int vdx = 0; vdx < jointNum; vdx++) {
      double weighting = 1.0 / modelNum;

      if (m == 0) {
        weighting = weighting_a[vdx];
      } else if (m == 1) {
        weighting = weighting_b[vdx];
      }

      for (int i = 0; i < 3; i++) {
          if (!skeOpt->SetModelVertexWeight(m, vdx, i, weighting)) {
            qDebug() << "set model weighting failed!\n";
            return;
          }
      }
    }
  }
  delete[] nodearray1;
  delete[] nodearray2;
  //delete[] tmparray;
  // ready to Run ///////////////////////////////////////////

  skeOpt->Run(50, true); // Run the program

  // get back the result/////////////////////////////////////////////////////////////////////////////
  for (int joint = 0; joint < jointNum; joint++) {
    double x, y, z;

    if (!skeOpt->GetVertexPosition(joint, x, y, z)) {
      qDebug() << "get vertex position failed!\n";
      return;
    }

    templateModel->joints[joint].position.xyz.x = x;
    templateModel->joints[joint].position.xyz.y = y;
    templateModel->joints[joint].position.xyz.z = z;
  }

  cv::Mat colorMat;
  k4abt_skeleton_t draw_ske = *templateModel;
  colorMat = mSkeData.colorMatoris0[frame].clone();

  // Assign the correct color based on the body id
  Color color = g_bodyColors[0];
  color.a = 0.4f;

  // Visualize joints
  for (int joint = 0; joint < jointNum; joint++) {
    const k4a_float3_t &jointPosition = draw_ske.joints[joint].position;

    k4a_float2_t joint2d;
    int valid;
    k4a_calibration_3d_to_2d(&sensorCalibrations[0], &jointPosition,
                             K4A_CALIBRATION_TYPE_DEPTH,
                             K4A_CALIBRATION_TYPE_COLOR, &joint2d, &valid);
    cv::Point Point2d = cv::Point(int(joint2d.xy.x), int(joint2d.xy.y));
    if (valid) {
      Color joint_color = color;
      cv::circle(colorMat, Point2d, 6,
                 cv::Scalar(joint_color.b * 255, joint_color.g * 255,
                            joint_color.r * 255, joint_color.a * 255),
                 cv::FILLED, cv::LINE_8);
    }
  }

  // Visualize bones
  for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++) {
    k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
    k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;

    const k4a_float3_t &joint1Position = draw_ske.joints[joint1].position;
    const k4a_float3_t &joint2Position = draw_ske.joints[joint2].position;
    k4a_float2_t joint2d_1, joint2d_2;
    int valid1, valid2;
    k4a_calibration_3d_to_2d(&sensorCalibrations[0], &joint1Position,
                             K4A_CALIBRATION_TYPE_DEPTH,
                             K4A_CALIBRATION_TYPE_COLOR, &joint2d_1, &valid1);
    k4a_calibration_3d_to_2d(&sensorCalibrations[0], &joint2Position,
                             K4A_CALIBRATION_TYPE_DEPTH,
                             K4A_CALIBRATION_TYPE_COLOR, &joint2d_2, &valid2);
    if (valid1 && valid2) {
      Color bone_color = color;
      cv::line(colorMat, cv::Point(int(joint2d_1.xy.x), int(joint2d_1.xy.y)),
               cv::Point(int(joint2d_2.xy.x), int(joint2d_2.xy.y)),
               cv::Scalar(bone_color.b * 255, bone_color.g * 255,
                          bone_color.r * 255, bone_color.a * 255),
               5, cv::LINE_4);
    }
  }

  std::string timetext;
  if(use_cam == 0){
      timetext = std::to_string(mSkeData.cam0_t[frame]);
  } else {
      timetext = std::to_string(mSkeData.cam1_t[frame-1]);
  }

  cv::putText(colorMat, //target image
              timetext, //text
              cv::Point(10, 80), //top-left position
              cv::FONT_HERSHEY_SIMPLEX,
              3.0,
              cv::Scalar(0, 0, 255, 255), //font color
              4.0);

  int id = 0;
  if(use_cam==0){
      id = frame*2;
  } else {
      id = (frame-1)*2+1;
  }
  std::string path = "/home/cpii/Desktop/test_img/camall/" + std::to_string(id) + ".png";
  cv::imwrite(path, colorMat);

  colorMat.release();
}

void LP_Plugin_MotionTracking::checking_estpoint_new(k4abt_joint_t **skea,
                                                     k4abt_joint_t **skeb,
                                                     int use_cam) {
  /////////////////////////////////////////////////////////////
  for (int i = 0; i < mJoint_tmp_num; i++) {
    k4abt_joint_t *nodea = skea[i];
    k4abt_joint_t *nodeb = skeb[i];
    /*
    double ax, ay, az, bx, by, bz;
    ax = nodea->position.xyz.x;
    ay = nodea->position.xyz.y;
    az = nodea->position.xyz.z;
    bx = nodeb->position.xyz.x;
    by = nodeb->position.xyz.y;
    bz = nodeb->position.xyz.z;
    */

    // estimation////////////////////////////////////////////////////////////////
    // printf("joint point: %d\n", i + 1);

    // set the weighting
    // //////////////////////////////////////////////////////////////////////////////
    if ((nodea->confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM &&
        nodeb->confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM) ||
        (nodea->confidence_level < K4ABT_JOINT_CONFIDENCE_MEDIUM &&
        nodeb->confidence_level < K4ABT_JOINT_CONFIDENCE_MEDIUM)) {
      // "infer" containing "inferred
      // point" information returned from
      // Kinect
      weighting_a[i] = 0.5;
      weighting_b[i] = 0.5;

      cam_prior_new(skea, skeb, i, use_cam);
    } else {
      cam_prior_new(skea, skeb, i, use_cam);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // printf("final ratio: %lf %lf\n", weighting_a[i], weighting_b[i]);
  }
}

void LP_Plugin_MotionTracking::cam_prior_new(k4abt_joint_t **skea,
                                             k4abt_joint_t **skeb,
                                             int jointpoint,
                                             int use_cam) {
  // initialize value////////////////////////////////////////////////////////
  int k = 4;
  // double infer_ratio = 0.3;

  // OR/////////////////////////////////////////////
  int corres_jpoint = choosing_dis(
      skea, skeb, jointpoint); // check the corresponding joints of 2 skeletons.
                               // e.g.right hand and left hand
  if (corres_jpoint == -1) {
    return;
  }

  /////////////////////////////////////////////////

  // calclate distance/////////////////////////////////////////////
  double x1, y1, z1, x2, y2, z2;

  // Kinect A///////////////////////////////////////////////////////
  x1 = skea[jointpoint]->position.xyz.x; // coordinate of points without affine
                                         // transformation
  y1 = skea[jointpoint]->position.xyz.y;
  z1 = skea[jointpoint]->position.xyz.z;
  x2 = skea[corres_jpoint]->position.xyz.x;
  y2 = skea[corres_jpoint]->position.xyz.y;
  z2 = skea[corres_jpoint]->position.xyz.z;
  //double dis_a = sqrt((x1 - x2) * (x1 - x2));
  double dis_a = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

  // Kinect B///////////////////////////////////////////////////////
  x1 = skeb[jointpoint]->position.xyz.x; // coordinate of points without affine
                                         // transformation
  y1 = skeb[jointpoint]->position.xyz.y;
  z1 = skeb[jointpoint]->position.xyz.z;
  x2 = skeb[corres_jpoint]->position.xyz.x;
  y2 = skeb[corres_jpoint]->position.xyz.y;
  z2 = skeb[corres_jpoint]->position.xyz.z;
  //double dis_b = sqrt((x1 - x2) * (x1 - x2));
  double dis_b = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
  // printf("dis_a: %lf\n dis_b: %lf\n", dis_a, dis_b);

  ////////////////////////////////////////////////////
  double weighting_factora = 0.0, weighting_factorb = 0.0;
  ////////////////////////////////////////////////
  double weight_ha = 0.5, weight_hb = 0.5;
  double tt = 0.4;
  if (skea[jointpoint]->confidence_level < K4ABT_JOINT_CONFIDENCE_MEDIUM &&
      skeb[jointpoint]->confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM) {
    weight_hb = smooth_step(tt);
    weight_ha = 1.0 - smooth_step(tt);
  } else if (skea[jointpoint]->confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM &&
             skeb[jointpoint]->confidence_level < K4ABT_JOINT_CONFIDENCE_MEDIUM) {
    weight_ha = smooth_step(tt);
    weight_hb = 1.0 - smooth_step(tt);
  }
  // printf("infer ratio: %lf %lf\n", weight_ha, weight_hb);
  // Kinect A///////////////////////////////////////
  double tmp_ratioa;
  if ((dis_a + dis_b) == 0.0) {
    tmp_ratioa = dis_a;
  } else {
    tmp_ratioa = dis_a / (dis_a + dis_b);
  }
  tmp_ratioa *= weight_ha;
  for (int i = 1; i < k; i++) {
    tmp_ratioa *= tmp_ratioa;
  }
  // Kinect B///////////////////////////////////////
  double tmp_ratiob;
  if ((dis_a + dis_b) == 0.0) {
    tmp_ratiob = dis_b;
  } else {
    tmp_ratiob = dis_b / (dis_a + dis_b);
  }
  tmp_ratiob *= weight_hb;
  for (int i = 1; i < k; i++) {
    tmp_ratiob *= tmp_ratiob;
  }
  ////////////////////////////////////////////////
  double ratioa = 0.0;
  if ((tmp_ratioa + tmp_ratiob) == 0.0) {
    ratioa = weighting_factora;
  } else {
    ratioa = tmp_ratioa / (tmp_ratioa + tmp_ratiob);
  }
  double ratiob = 0.0;
  if ((tmp_ratioa + tmp_ratiob) == 0.0) {
    ratiob = weighting_factorb;
  } else {
    ratiob = tmp_ratiob / (tmp_ratioa + tmp_ratiob);
  }
  weighting_a[jointpoint] = ratioa;
  weighting_b[jointpoint] = ratiob;
}

double LP_Plugin_MotionTracking::smooth_step(double t) {
  double e = 1.0 / M_PI;
  double h = (1.0 + (2.0 / M_PI) * atan(t / e)) / 2.0;
  return h;
}

int LP_Plugin_MotionTracking::choosing_dis(k4abt_joint_t **skea, k4abt_joint_t **skeb,
                             int jointpoint) {
  // nearest A
  // neibour////////////////////////////////////////////////////////////
  int _edgeNum = 31;
  k4abt_joint_t *node_i = skea[jointpoint];
  std::vector<double> dis_nei(_edgeNum);
  int nei_a_no = -1;
  double min_dis_a = 1e6;

  for (int i = 0; i < _edgeNum; i++) {
    dis_nei[i] = 1e6;
  }
  for (int i = 0; i < _edgeNum; i++) {
    if (i == jointpoint) {
      continue;
    }
    k4abt_joint_t *nei_node = skea[i];
    bool edge_check = false;

    // Edge check
    for (int j = 0; j < _edgeNum; j++) {
      if (g_boneList[j].first != jointpoint &&
          g_boneList[j].second != jointpoint) {
        continue;
      }
      if (g_boneList[j].first == i || g_boneList[j].second == i) {
        edge_check = true;
      }
    }
    // OR// two ring /////////////////////////
    /*
        for (int j = 0; j < _edgeNum && !edge_check; j++) {
          if (g_boneList[j].first != jointpoint &&
              g_boneList[j].second != jointpoint) {
            continue;
          }
          k4abt_joint_t *start_j = skea[g_boneList[j].first];
          int ednode_ind;
          if (start_j == node_i) {
            ednode_ind = g_boneList[j].first;
          } else {
            ednode_ind = g_boneList[j].second;
          }
          for (int k = 0; k < _edgeNum && !edge_check; k++) {
            if (g_boneList[k].first != ednode_ind &&
                g_boneList[k].second != ednode_ind) {
              continue;
            }
            k4abt_joint_t *nedge_f = skea[g_boneList[k].first];
            k4abt_joint_t *nedge_s = skea[g_boneList[k].second];
            if (nedge_f == nei_node || nedge_s == nei_node) {
              edge_check = true;
            }
          }
        }
    */
    ///////////////////////////////
    if (edge_check) {
      continue;
    }
    double x1, y1, z1, x2, y2, z2;
    x1 = node_i->position.xyz.x;
    y1 = node_i->position.xyz.y;
    z1 = node_i->position.xyz.z;
    x2 = nei_node->position.xyz.x;
    y2 = nei_node->position.xyz.y;
    z2 = nei_node->position.xyz.z;
    dis_nei[i] = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    //dis_nei[i] = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    //dis_nei[i] = sqrt((x1-x2)*(x1-x2));
    if (dis_nei[i] < min_dis_a) {
      min_dis_a = dis_nei[i];
      nei_a_no = i;
    }
  }
  // printf("nei from Kinect A: %d\n", nei_a_no);
  /////////////////////////////////////////////////////////////////////////////
  // nearest B
  // neibour////////////////////////////////////////////////////////////
  k4abt_joint_t *node_j = skeb[jointpoint];
  // double dis_nei[_edgeNum];
  int nei_b_no = -1;
  double min_dis_b = 1e6;
  for (int i = 0; i < _edgeNum; i++) {
    dis_nei[i] = 1e6;
  }
  for (int i = 0; i < _edgeNum; i++) {
    if (i == jointpoint) {
      continue;
    }
    k4abt_joint_t *nei_node = skeb[i];
    bool edge_check = false;

    // Edge check
    for (int j = 0; j < _edgeNum; j++) {
      if (g_boneList[j].first != jointpoint &&
          g_boneList[j].second != jointpoint) {
        continue;
      }
      if (g_boneList[j].first == i || g_boneList[j].second == i) {
        edge_check = true;
      }
    }

    // OR// two ring /////////////////////////
    /*
        for (int j = 0; j < _edgeNum && !edge_check; j++) {
          if (g_boneList[j].first != jointpoint &&
              g_boneList[j].second != jointpoint) {
            continue;
          }
          k4abt_joint_t *start_j = skeb[g_boneList[j].first];
          int ednode_ind;
          if (start_j == node_j) {
            ednode_ind = g_boneList[j].first;
          } else {
            ednode_ind = g_boneList[j].second;
          }
          for (int k = 0; k < _edgeNum && !edge_check; k++) {
            if (g_boneList[k].first != ednode_ind &&
                g_boneList[k].second != ednode_ind) {
              continue;
            }
            k4abt_joint_t *nedge_f = skeb[g_boneList[k].first];
            k4abt_joint_t *nedge_s = skeb[g_boneList[k].second];
            if (nedge_f == nei_node || nedge_s == nei_node) {
              edge_check = true;
            }
          }
        }
    */
    ///////////////////////////////
    if (edge_check) {
      continue;
    }
    double x1, y1, z1, x2, y2, z2;
    x1 = node_j->position.xyz.x;
    y1 = node_j->position.xyz.y;
    z1 = node_j->position.xyz.z;
    x2 = nei_node->position.xyz.x;
    y2 = nei_node->position.xyz.y;
    z2 = nei_node->position.xyz.z;
    dis_nei[i] = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    //dis_nei[i] = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    //dis_nei[i] = sqrt((x1-x2)*(x1-x2));
    if (dis_nei[i] < min_dis_b) {
      min_dis_b = dis_nei[i];
      nei_b_no = i;
    }
  }
  // printf("nei from Kinect B: %d\n", nei_b_no);
  /////////////////////////////////////////////////////////////////////////////
  // compare a&b////////////////////////////////////////////////////////////////
  int final_nei_no = -1;
  if (min_dis_a < min_dis_b){
    final_nei_no = nei_a_no;
  } else {
    final_nei_no = nei_b_no;
  }
  // printf("nei final: %d\n", final_nei_no);

  /////////////////////////////////////////////////////////////////////////////
  // free
  // delete []dis_nei;

  return final_nei_no;
}

void LP_Plugin_MotionTracking::loadObj(const char *filename,
                                       std::vector<std::array<float, 6>> &vao,
                                       std::vector<uint> &f)
{
    QString qs;
    QFile fin(filename);
    if(!fin.open(QIODevice::ReadOnly)) {
        qDebug() << "Failed to open file";
        return;
    }
    QTextStream in(&fin);

    std::vector<std::array<int,3>> fids;
    std::vector<QVector3D> verts, norms;

    float x_min = 0.0, x_max = 0.0,
          y_min = 0.0, y_max = 0.0,
          z_min = 0.0, z_max = 0.0;

    while(!in.atEnd())
    {
        qs = in.readLine();
        auto l = qs.split(' ');
        if ( "v" == l[0] ) {
            float l1_scaled = l[1].toFloat()*D->mModelScale;
            float l2_scaled = l[2].toFloat()*D->mModelScale;
            float l3_scaled = l[3].toFloat()*D->mModelScale;
            verts.emplace_back(l1_scaled, l2_scaled, l3_scaled);
            if(!D->init_modelpos){
                if(!D->init_BB){
                    x_min = l1_scaled;
                    x_max = l1_scaled;
                    y_min = l2_scaled;
                    y_max = l2_scaled;
                    z_min = l3_scaled;
                    z_max = l3_scaled;
                    D->init_BB = true;
                }
                if(l1_scaled < x_min){
                    x_min = l1_scaled;
                } else if(l1_scaled > x_max){
                    x_max = l1_scaled;
                }
                if(l2_scaled < y_min){
                    y_min = l2_scaled;
                } else if(l2_scaled > y_max){
                    y_max = l2_scaled;
                }
                if(l3_scaled < z_min){
                    z_min = l3_scaled;
                } else if(l3_scaled > z_max){
                    z_max = l3_scaled;
                }
            }
        } else if ( "vn" == l[0] ) {
            norms.emplace_back(l[1].toFloat(), l[2].toFloat(), l[3].toFloat());
        } else if ("f" == l[0] ){
            auto fid = l[1].split('/');
            fids.push_back({fid[0].toInt(), fid[1].toInt(), fid[2].toInt()});

            fid = l[2].split('/');
            fids.push_back({fid[0].toInt(), fid[1].toInt(), fid[2].toInt()});

            fid = l[3].split('/');
            fids.push_back({fid[0].toInt(), fid[1].toInt(), fid[2].toInt()});
        } else if ("vt" == l[0] ){

        } else {
            if(!D->init_modelpos){
                qWarning() << "Unknown element : " << qs;
            }
        }
    }
    fin.close();

    if(!D->init_modelpos){
        qDebug() << "Done tangest";
    }
    std::transform(fids.begin(), fids.end(), fids.begin(), [](std::array<int, 3> &i){
        --i[0]; --i[1]; --i[2];
        return i;
    });

    const int nFs = int(fids.size());

    vao.clear();
    f.clear();
    vao.resize(nFs);
    f.resize(nFs);
    for ( int i=0; i<nFs; i += 3 ) {
        const auto &fv0 = fids.at(i),
                   &fv1 = fids.at(i+1),
                   &fv2 = fids.at(i+2);
        std::array<float, 6> va;

        const auto *v = &verts.at(fv0[0]);
        const auto *n = &norms.at(fv0[2]);

        //Setting VAO
        va[0] = v->x(); va[1] = v->y(); va[2] = v->z();
        va[3] = n->x(); va[4] = n->y(); va[5] = n->z();
        vao[i] = va;
        f[i] = i;

        v = &verts.at(fv1[0]);
        n = &norms.at(fv1[2]);
        va[0] = v->x(); va[1] = v->y(); va[2] = v->z();
        va[3] = n->x(); va[4] = n->y(); va[5] = n->z();
        vao[i+1] = va;
        f[i+1] = i+1;

        v = &verts.at(fv2[0]);
        n = &norms.at(fv2[2]);
        va[0] = v->x(); va[1] = v->y(); va[2] = v->z();
        va[3] = n->x(); va[4] = n->y(); va[5] = n->z();
        vao[i+2] = va;
        f[i+2] = i+2;
    }

    if(!D->init_modelpos){
        D->mModelBB_min = QVector3D(x_min-0.1*(x_max-x_min), y_min-0.1*(y_max-y_min), z_min-0.1*(z_max-z_min));
        D->mModelBB_max = QVector3D(x_max+0.1*(x_max-x_min), y_max+0.1*(y_max-y_min), z_max+0.1*(z_max-z_min));

        QVector3D posi = QVector3D((D->mModelBB_min.x()+D->mModelBB_max.x())*0.5,
                                   (D->mModelBB_min.y()+D->mModelBB_max.y())*0.5,
                                   (D->mModelBB_min.z()+D->mModelBB_max.z())*0.5);
        D->mModelPos = posi;
        D->mCam = QVector3D(posi.x(), posi.y(), D->mModelBB_max.z()+D->mModel_Nearplane);
        if(D->mModel_Farplane - D->mModel_Nearplane < D->mModelBB_max.z() - D->mModelBB_min.z()){
            D->mModel_Farplane = D->mModelBB_max.z() - D->mModelBB_min.z() + D->mModel_Nearplane;
        }

        qDebug() << "Model position: " << D->mModelPos << "\n"
                 << "Camera position: " << D->mCam << "\n"
                 << "Bounding box min: " << D->mModelBB_min << "\n"
                 << "Bounding box max: " << D->mModelBB_max << "\n"
                 << "Near plane: " << D->mModel_Nearplane << "\n"
                 << "Far plane: " << D->mModel_Farplane << "\n";

        qDebug() << filename << " loaded";

        D->init_modelpos = true;
    }
}

void LP_Plugin_MotionTracking::create_paths(){
    if(!QDir(QString::fromStdString(mCam0_path)).exists()){
        QDir().mkdir(QString::fromStdString(mCam0_path));
    }
    if(!QDir(QString::fromStdString(mCam0ori_path)).exists()){
        QDir().mkdir(QString::fromStdString(mCam0ori_path));
    }
    if(!QDir(QString::fromStdString(mCam0ske_path)).exists()){
        QDir().mkdir(QString::fromStdString(mCam0ske_path));
    }
    if(!QDir(QString::fromStdString(mCam1_path)).exists()){
        QDir().mkdir(QString::fromStdString(mCam1_path));
    }
    if(!QDir(QString::fromStdString(mCam1ori_path)).exists()){
        QDir().mkdir(QString::fromStdString(mCam1ori_path));
    }
    if(!QDir(QString::fromStdString(mCam1ske_path)).exists()){
        QDir().mkdir(QString::fromStdString(mCam1ske_path));
    }
    if(!QDir(QString::fromStdString(mCam1skeori_path)).exists()){
        QDir().mkdir(QString::fromStdString(mCam1skeori_path));
    }
    if(!QDir(QString::fromStdString(mSke0_path)).exists()){
        QDir().mkdir(QString::fromStdString(mSke0_path));
    }
    if(!QDir(QString::fromStdString(mSke0obj_path)).exists()){
        QDir().mkdir(QString::fromStdString(mSke0obj_path));
    }
    if(!QDir(QString::fromStdString(mSke_path)).exists()){
        QDir().mkdir(QString::fromStdString(mSke_path));
    }
    if(!QDir(QString::fromStdString(mSke1_path)).exists()){
        QDir().mkdir(QString::fromStdString(mSke1_path));
    }
    if(!QDir(QString::fromStdString(mSke1obj_path)).exists()){
        QDir().mkdir(QString::fromStdString(mSke1obj_path));
    }
    if(!QDir(QString::fromStdString(mCam0ir_path)).exists()){
        QDir().mkdir(QString::fromStdString(mCam0ir_path));
    }
    if(!QDir(QString::fromStdString(mCam1ir_path)).exists()){
        QDir().mkdir(QString::fromStdString(mCam1ir_path));
    }
//    if(!QDir(QString::fromStdString(mPointcloud_path0)).exists()){
//        QDir().mkdir(QString::fromStdString(mPointcloud_path0));
//    }
//    if(!QDir(QString::fromStdString(mPointcloud_path1)).exists()){
//        QDir().mkdir(QString::fromStdString(mPointcloud_path1));
//    }
}

void LP_Plugin_MotionTracking::save_Rt(cv::Mat Aligned_skeb){
    std::string path0mesh = "/home/cpii/Desktop/test_img/RT_A.obj";
    std::string path1mesh = "/home/cpii/Desktop/test_img/RT_B.obj";
    std::string pathkmesh = "/home/cpii/Desktop/test_img/RT_K.obj";
    std::string pathRt = "/home/cpii/Desktop/test_img/Rt.csv";

    save_obj(path0mesh, mKinectA_pts);
    save_obj(path1mesh, mKinectB_pts);
    save_obj(pathkmesh, Aligned_skeb);
    std::ofstream myFile(pathRt);
    for (size_t i = 0; i < mRet.R.rows; ++i) {
        for (size_t j = 0; j < mRet.R.cols; ++j) {
            myFile << mRet.R.at<double>(i, j);
            if(i!=(mRet.R.rows-1) || j!=(mRet.R.cols-1)){
                myFile << " ";
            }
        }
    }
    myFile << "\n";
    for (size_t i = 0; i < mRet.t.rows; ++i) {
        myFile << mRet.t.at<double>(i, 0);
        if(i!=(mRet.t.rows-1)){
            myFile << " ";
        }
    }
    myFile.close();
}

void LP_Plugin_MotionTracking::save_obj(std::string path, cv::Mat mat){
    std::ofstream myFile(path);
    for (int repeat = 0; repeat < 2; repeat++){
        for (size_t j = 0; j < mat.cols; ++j) {
          myFile << "v ";
          myFile << mat.at<double>(0, j);
          myFile << " ";
          myFile << mat.at<double>(1, j);
          myFile << " ";
          myFile << mat.at<double>(2, j);
          myFile << "\n";
        }
    }
    for (size_t i = 0; i < g_boneList.size(); ++i) {
      myFile << "f ";
      myFile << g_boneList[i].first + 1;
      myFile << " ";
      myFile << g_boneList[i].second + 1;
      myFile << " ";
      myFile << g_boneList[i].second + 1 + mJoint_tmp_num;
      myFile << "\n";
    }
    myFile.close();
}

void LP_Plugin_MotionTracking::save_obj(std::string path, k4abt_skeleton_t *ske){
    std::ofstream myFile(path);
    for (int repeat = 0; repeat < 2; repeat++){
        for (size_t j = 0; j < static_cast<int>(K4ABT_JOINT_COUNT); ++j) {
          myFile << "v ";
          myFile << ske->joints[j].position.xyz.x;
          myFile << " ";
          myFile << ske->joints[j].position.xyz.y;
          myFile << " ";
          myFile << ske->joints[j].position.xyz.z;
          myFile << "\n";
        }
    }
    for (size_t i = 0; i < g_boneList.size(); ++i) {
      myFile << "f ";
      myFile << g_boneList[i].first + 1;
      myFile << " ";
      myFile << g_boneList[i].second + 1;
      myFile << " ";
      myFile << g_boneList[i].second + 1 + mJoint_tmp_num;
      myFile << "\n";
    }
    myFile.close();
}

void LP_Plugin_MotionTracking::save_samples(){
    qDebug() << "Saving samples";
    // save images and obj
    for(int i=0; i<mSkeData.colorMats0.size(); i++){
        std::string path = mCam0_path + std::to_string(i*2) + ".png";
        //cv::rotate(mTmpMat, mTmpMat, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imwrite(path, mSkeData.colorMats0[i]);
        //cv::rotate(mTmpMat, mTmpMat, cv::ROTATE_90_CLOCKWISE);

        path = mCam1_path + std::to_string(i*2+1) + ".png";
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imwrite(path, mSkeData.colorMats1[i]);
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_CLOCKWISE);

        path = mCam0ori_path + std::to_string(i*2) + ".png";
        cv::imwrite(path, mSkeData.colorMatoris0[i]);

        path = mCam1ori_path + std::to_string(i*2+1) + ".png";
        cv::imwrite(path, mSkeData.colorMatoris1[i]);

        path = mCam0ir_path + std::to_string(i*2) + ".jpg";
        //cv::rotate(save_mat, save_mat, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imwrite(path, mSkeData.irMats0[i]);

        path = mCam1ir_path + std::to_string(i*2+1) + ".jpg";
        //cv::rotate(save_mat, save_mat, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imwrite(path, mSkeData.irMats1[i]);

        path = mSke_path + std::to_string(i*2) + ".png";
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imwrite(path, mSkeData.optskeMats0[i]);
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_CLOCKWISE);

        path = mSke0_path + std::to_string(i*2) + ".png";
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imwrite(path, mSkeData.optskeMats0[i]);
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_CLOCKWISE);

        path = mSke0obj_path + std::to_string(i*2) + ".obj";
        save_obj(path, &mSkeData.optskes0[i]);

        if(!D->mUse_Model){
            std::string path0mesh = mCam0ske_path + std::to_string(i*2) + ".obj";
            std::string path1mesh = mCam1ske_path + std::to_string(i*2+1) + ".obj";
            std::string path1orimesh = mCam1skeori_path + std::to_string(i*2+1) + ".obj";
            save_obj(path0mesh, &mSkeData.skes0[i]);
            save_obj(path1mesh, &mSkeData.skes1[i]);
            save_obj(path1orimesh, &mSkeData.skes1ori[i]);
        }
    }
    for(int i=0; i<mSkeData.optskeMats1.size(); i++){
        std::string path = mSke_path + std::to_string(i*2+1) + ".png";
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imwrite(path, mSkeData.optskeMats1[i]);
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_CLOCKWISE);

        path = mSke1_path + std::to_string(i*2+1) + ".png";
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imwrite(path, mSkeData.optskeMats1[i]);
        //cv::rotate(colorMat, colorMat, cv::ROTATE_90_CLOCKWISE);

        path = mSke1obj_path + std::to_string(i*2+1) + ".obj";
        save_obj(path, &mSkeData.optskes1[i]);
    }

    // save cam time
    std::ofstream myFile(mCamtime_path);
    for (size_t i = 0; i < mSkeData.cam0_t.size(); i++) {
      myFile << mSkeData.cam0_t[i];
      myFile << " ";
      myFile << mSkeData.cam1_t[i];
      myFile << "\n";
    }
    myFile.close();

    // save original cam time
    std::ofstream myFileo(mCamtimeori_path);
    for (size_t i = 0; i < mSkeData.cam0_t_ori.size(); i++) {
      myFileo << mSkeData.cam0_t_ori[i];
      myFileo << " ";
      myFileo << mSkeData.cam1_t_ori[i];
      myFileo << "\n";
    }
    myFileo.close();

    // save time stamps
    std::ofstream myFile2(mTimestamps_path);
    for (size_t i = 0; i < mSkeData.Timestamps.size(); i++) {
      myFile2 << mSkeData.Timestamps[i];
      myFile2 << "\n";
    }
    myFile2.close();

    // save joint positions
    std::ofstream myFile3(mJointposi_path);
    for (size_t f = 0; f < mSkeData.Joint_posis.size(); f++){
        for (size_t j = 0; j < mSkeData.Joint_posis[f].size(); j++) {
          myFile3 << mSkeData.Joint_posis[f][j].xyz.x;
          myFile3 << " ";
          myFile3 << mSkeData.Joint_posis[f][j].xyz.y;
          myFile3 << " ";
          myFile3 << mSkeData.Joint_posis[f][j].xyz.z;
          myFile3 << " ";
        }
        myFile3 << "\n";
    }
    myFile3.close();

    // save joint velocity
    std::ofstream myFile4(mJointvelo_path);
    for (size_t f = 0; f < mSkeData.Joint_posis.size(); f++){
        float time = mSkeData.Timestamps[f];
        std::vector<k4a_float3_t> velo;
        for (size_t j = 0; j < mSkeData.Joint_posis[f].size(); j++) {
            k4a_float3_t tmpv;
            if(f==mSkeData.Joint_posis.size()-1){
                tmpv.xyz.x = 0.0;
                tmpv.xyz.y = 0.0;
                tmpv.xyz.z = 0.0;
                velo.push_back(tmpv);

                myFile4 << 0.0;
                myFile4 << " ";
                myFile4 << 0.0;
                myFile4 << " ";
                myFile4 << 0.0;
                myFile4 << " ";
            } else {
                // s=1/2(u+v)t -> v=2s/t-u
//                float vx = 2.0*(mJoint_posis[f][j].xyz.x-mJoint_posis[f-1][j].xyz.x)/time - mJoint_velo[f-1][j].xyz.x;
//                float vy = 2.0*(mJoint_posis[f][j].xyz.y-mJoint_posis[f-1][j].xyz.y)/time - mJoint_velo[f-1][j].xyz.y;
//                float vz = 2.0*(mJoint_posis[f][j].xyz.z-mJoint_posis[f-1][j].xyz.z)/time - mJoint_velo[f-1][j].xyz.z;

                // s=vt -> v=s/t
                float vx = (mSkeData.Joint_posis[f+1][j].xyz.x-mSkeData.Joint_posis[f][j].xyz.x)/time;
                float vy = (mSkeData.Joint_posis[f+1][j].xyz.y-mSkeData.Joint_posis[f][j].xyz.y)/time;
                float vz = (mSkeData.Joint_posis[f+1][j].xyz.z-mSkeData.Joint_posis[f][j].xyz.z)/time;

                tmpv.xyz.x = vx;
                tmpv.xyz.y = vy;
                tmpv.xyz.z = vz;
                velo.push_back(tmpv);

                myFile4 << vx;
                myFile4 << " ";
                myFile4 << vy;
                myFile4 << " ";
                myFile4 << vz;
                myFile4 << " ";
            }
        }
        myFile4 << "\n";
        mSkeData.Joint_velo.push_back(velo);
    }
    myFile4.close();

    // save joint acceleration
    std::ofstream myFile5(mJointacce_path);
    for (size_t f = 0; f < mSkeData.Joint_velo.size(); f++){
        float time = mSkeData.Timestamps[f+1];
        for (size_t j = 0; j < mSkeData.Joint_velo[f].size(); j++) {
            if(f==mSkeData.Joint_velo.size()-1){
                myFile5 << 0.0;
                myFile5 << " ";
                myFile5 << 0.0;
                myFile5 << " ";
                myFile5 << 0.0;
                myFile5 << " ";
            } else {
                // v=u+at -> a=(v-u)/t
                myFile5 << (mSkeData.Joint_velo[f+1][j].xyz.x-mSkeData.Joint_velo[f][j].xyz.x)/time;
                myFile5 << " ";
                myFile5 << (mSkeData.Joint_velo[f+1][j].xyz.y-mSkeData.Joint_velo[f][j].xyz.y)/time;
                myFile5 << " ";
                myFile5 << (mSkeData.Joint_velo[f+1][j].xyz.z-mSkeData.Joint_velo[f][j].xyz.z)/time;
                myFile5 << " ";
            }
        }
        myFile5 << "\n";
    }
    myFile5.close();

//    // save point cloud
//    KAfunctions->write_point_cloud(mPointcloud_path0, mPointcloud_path1);

    // save 2d Points
//    std::ofstream myFile6(m2dPoint_path);
//    for (size_t i = 0; i < mSkeData.ELBOW_LEFT0.size(); i++){
//        myFile6 << mSkeData.ELBOW_LEFT0[i].x;
//        myFile6 << " ";
//        myFile6 << mSkeData.ELBOW_LEFT0[i].y;
//        myFile6 << " ";
//        myFile6 << mSkeData.WRIST_LEFT0[i].x;
//        myFile6 << " ";
//        myFile6 << mSkeData.WRIST_LEFT0[i].y;
//        myFile6 << "\n";
//        if(i<mSkeData.ELBOW_LEFT0.size()-1){
//            myFile6 << mSkeData.ELBOW_LEFT1[i].x;
//            myFile6 << " ";
//            myFile6 << mSkeData.ELBOW_LEFT1[i].y;
//            myFile6 << " ";
//            myFile6 << mSkeData.WRIST_LEFT1[i].x;
//            myFile6 << " ";
//            myFile6 << mSkeData.WRIST_LEFT1[i].y;
//            myFile6 << "\n";
//        }
//    }
//    myFile6.close();
}

void LP_Plugin_MotionTracking::PainterDraw(QWidget *glW)
{
    QPainter painter(glW);
    if ( "window_Shade" == glW->objectName()) {
        int hw = 0.5*glW->width();
        int img_h = 0;
        //float aspect = float(mDImage0.width())/mDImage0.height();
        if(!mUse_Opt){
            //auto img = mImage0.copy().transformed(QMatrix().rotate(-90.0));
            //img = img.scaledToWidth(0.5*hw);
            auto img = mImage0.copy().scaledToWidth(hw);
            painter.drawImage(0, 0, img);

            //auto img2 = mImage1.copy().transformed(QMatrix().rotate(-90.0));
            //img2 = img2.scaledToWidth(0.5*hw);
            auto img2 = mImage1.copy().scaledToWidth(hw);
            img_h = img2.height();
            painter.drawImage(hw, 0, img2);
        } else {
            int w3 = 0.333*glW->width();

            //auto img = mImage0.copy().transformed(QMatrix().rotate(-90.0));
            //img = img.scaledToWidth(w3);
            auto img = mImage0.copy().scaledToWidth(w3);
            img_h = img.height();
            painter.drawImage(0, 0, img);

            //auto img2 = mImage1.copy().transformed(QMatrix().rotate(-90.0));
            //img2 = img2.scaledToWidth(w3);
            auto img2 = mImage1.copy().scaledToWidth(w3);
            painter.drawImage(w3, 0, img2);

            if(!mImageske.isNull()){
                //auto imgske = mImageske.copy().transformed(QMatrix().rotate(-90.0));
                //imgske = imgske.scaledToWidth(w3);
                auto imgske = mImageske.copy().scaledToWidth(w3);
                painter.drawImage(2*w3, 0, imgske);
            }
        }
        //auto dimg = mDImage0.copy().transformed(QMatrix().rotate(-90.0).scale((float)hw*0.5/mDImage0.width(), aspect*hw*0.5/mDImage0.width()));
        QImage dimg = mDImage0.copy().scaledToWidth(0.5*hw, Qt::SmoothTransformation);
        painter.drawImage(0, img_h, dimg);

        //auto irimg = mIrImage0.copy().transformed(QMatrix().rotate(-90.0).scale((float)hw*0.5/mIrImage0.width(), aspect*hw*0.5/mIrImage0.width()));
        auto irimg = mIrImage0.copy().scaledToWidth(0.5*hw, Qt::SmoothTransformation);
        painter.drawImage(0.5*hw, img_h, irimg);

        //auto dimg2 = mDImage1.copy().transformed(QMatrix().rotate(-90.0).scale((float)hw*0.5/mDImage0.width(), aspect*hw*0.5/mDImage0.width()));
        auto dimg2 = mDImage1.copy().scaledToWidth(0.5*hw, Qt::SmoothTransformation);
        painter.drawImage(hw, img_h, dimg2);

        //auto irimg2 = mIrImage1.copy().transformed(QMatrix().rotate(-90.0).scale((float)hw*0.5/mIrImage1.width(), aspect*hw*0.5/mIrImage1.width()));
        auto irimg2 = mIrImage1.copy().scaledToWidth(0.5*hw, Qt::SmoothTransformation);
        painter.drawImage(1.5*hw, img_h, irimg2);
    } else {
        if(D->mUse_Model){
            int qw = 0.5*glW->width();

            Get_depthmap();

            mLock.lockForRead();
            QImage dm0 = D->mDepth_Map_color0;
            QImage dm1 = D->mDepth_Map_color1;
            mLock.unlock();

            if(!dm0.isNull()){
                dm0 = dm0.scaledToWidth(qw);
                painter.drawImage(0, 0, dm0);
            }
            if(!dm1.isNull()){
                dm1 = dm1.scaledToWidth(qw);
                painter.drawImage(qw, 0, dm1);
            }
        }
    }
}

QString LP_Plugin_MotionTracking::MenuName()
{
    return tr("menuPlugins");
}

QAction *LP_Plugin_MotionTracking::Trigger()
{
    if ( !mAction ){
        mAction = new QAction("Motion Tracking");
    }
    return mAction;
}

void LP_Plugin_MotionTracking::initializeGL_R()
{
    std::string vsh, fsh;

    vsh =
        "attribute vec3 a_pos;\n"       //The position of a point in 3D that used in FunctionRender()
        "attribute vec3 a_norm;\n"
        "uniform mat3 m3_normal;\n"
        "uniform mat4 m4_mvp;\n"        //The Model-View-Matrix
        "uniform mat4 m4_view;\n"
        "uniform float f_pointSize;\n"  //Point size determined in FunctionRender()
        "varying vec3 normal;\n"
        "varying vec3 pos;\n"
        "void main(){\n"
        "   pos = vec3( m4_view * vec4(a_pos, 1.0));\n"
        "   gl_Position = m4_mvp * vec4(a_pos, 1.0);\n" //Output the OpenGL position
        "   gl_PointSize = f_pointSize; \n"
        "   normal = m3_normal * a_norm;\n"
        "}";
    fsh =
        "uniform vec4 v4_color;\n"
        "varying vec3 pos;\n"
        "varying vec3 normal;\n"
        "void main(){\n"
        "   vec3 lightPos = vec3(0.0, 1000.0, 0.0);\n"
        "   vec3 viewDir = normalize( - pos);\n"
        "   vec3 lightDir = normalize(lightPos - pos);\n"
        "   vec3 H = normalize(viewDir + lightDir);\n"
        "   vec3 N = normalize(normal);\n"
        "   vec3 ambi = v4_color.rgb;\n"
        "   float Kd = max(dot(H, N), 0.0);\n"
        "   vec3 diff = Kd * vec3(0.2, 0.2, 0.2);\n"
        "   vec3 color = ambi + diff;\n"
        "   float Ks = pow( Kd, 80.0 );\n"
        "   vec3 spec = Ks * vec3(0.5, 0.5, 0.5);\n"
        "   color += spec;\n"
        "   gl_FragColor = vec4(color,1.0);\n" //Output the fragment color;
        "   gl_FragColor.a = v4_color.a;\n"
        "   "
        "}";

    auto prog = new QOpenGLShaderProgram;   //Intialize the Shader with the above GLSL codes
    prog->addShaderFromSourceCode(QOpenGLShader::Vertex,vsh.c_str());
    prog->addShaderFromSourceCode(QOpenGLShader::Fragment,fsh.data());
    if (!prog->create() || !prog->link()){  //Check whether the GLSL codes are valid
        qDebug() << prog->log();
        return;
    }
    mProgram_R = prog;            //If everything is fine, assign to the member variable

    mInitialized_R = true;
}

void LP_Plugin_MotionTracking::Model_member::initializeGL_R_DM(QOpenGLContext *ctx, QSurface *surf){
    ctx->makeCurrent(surf);

    std::string vsh_dm, fsh_dm;

    vsh_dm =
        "#version 450\n"
        "in vec3 a_pos;\n"
        "uniform mat4 m4_mvp;\n"
        "in vec3 a_norm;\n"
        "out vec3 norm;\n"
        "out float depth;\n"
        "void main(){\n"
        "   gl_Position = m4_mvp * vec4(a_pos, 1.0);\n"
        "   depth = gl_Position.z / gl_Position.w;\n"
        "   depth = (depth + 1.0) * 0.5;\n"
        "   norm = a_norm;\n"
        "}";

//    fsh_dm =
//        "varying float depth;\n"
//        "vec4 EncodeFloatRGBA (float v) {\n"
//        "    vec4 enc = vec4(1.0, 255.0, 65025.0, 16581375.0) * v;\n"
//        "    enc = fract(enc);\n"
//        "    enc -= enc.yzww * vec4(1.0/255.0, 1.0/255.0, 1.0/255.0, 0.0);\n"
//        "    return enc;\n"
//        "}\n"
//        "float DecodeFloatRGBA (vec4 rgba) {\n"
//        "    return dot(rgba, vec4(1.0, 1.0/255.0, 1.0/65025.0, 1.0/16581375.0));\n"
//        "}\n"
//        "void main(){\n"
//        "   gl_FragColor = EncodeFloatRGBA(depth);\n"
//        "   gl_FragColor.a = 1.0;\n"
//        "}";


    fsh_dm =
        "#version 450\n"
        "in float depth;\n"
        "in vec3 norm;\n"
        "out vec4 color0;\n"
        "out vec4 color1;\n"
        "vec4 EncodeFloatRGBA (float v) {\n"
        "    vec4 enc = vec4(1.0, 255.0, 65025.0, 16581375.0) * v;\n"
        "    enc = fract(enc);\n"
        "    enc -= enc.yzww * vec4(1.0/255.0, 1.0/255.0, 1.0/255.0, 0.0);\n"
        "    return enc;\n"
        "}\n"
        "float DecodeFloatRGBA (vec4 rgba) {\n"
        "    return dot(rgba, vec4(1.0, 1.0/255.0, 1.0/65025.0, 1.0/16581375.0));\n"
        "}\n"
        "void main(){\n"
        "   color0 = EncodeFloatRGBA(depth);\n"
        //"   color0.a = 1.0;\n"
        "   vec3 n_norm = normalize(norm);\n"
        "   n_norm = (n_norm+1.0)*0.5;\n"
        "   color1.rgb = n_norm;\n"
        "   color1.a = 1.0;\n"
        "}";

    //The callback
    //mR_DM_FBO = new QOpenGLFramebufferObject(576, 640);
    mR_DM_FBO = new QOpenGLFramebufferObject(640, 576);
    mR_DM_FBO->setAttachment(QOpenGLFramebufferObject::Depth);
    //mR_DM_FBO->addColorAttachment(576, 640);
    mR_DM_FBO->addColorAttachment(640, 576);

    auto prog = new QOpenGLShaderProgram;   //Intialize the Shader with the above GLSL codes
    prog->addShaderFromSourceCode(QOpenGLShader::Vertex,vsh_dm.c_str());
    prog->addShaderFromSourceCode(QOpenGLShader::Fragment,fsh_dm.data());
    if (!prog->create() || !prog->link()){  //Check whether the GLSL codes are valid
        qDebug() << prog->log();
        return;
    }

    mProgram_R_DM = prog;

    ctx->doneCurrent();

    mInitialized_R_DM = true;
}

void LP_Plugin_MotionTracking::Get_depthmap()
{
    if ( !D->mInitialized_R_DM ) {
        D->initializeGL_R_DM(D->mCB_Context, D->mCB_Surface);

        D->mDepth_Map_dis0 = cv::Mat::zeros(D->mR_DM_FBO->height(), D->mR_DM_FBO->width(), CV_16UC1);
        D->mDepth_Map_dis1 = cv::Mat::zeros(D->mR_DM_FBO->height(), D->mR_DM_FBO->width(), CV_16UC1);
        D->mIr_Map0 = cv::Mat::zeros(D->mR_DM_FBO->height(), D->mR_DM_FBO->width(), CV_16UC1);
        D->mIr_Map1 = cv::Mat::zeros(D->mR_DM_FBO->height(), D->mR_DM_FBO->width(), CV_16UC1);
        D->mDBackground = cv::Mat::zeros(D->mR_DM_FBO->height(), D->mR_DM_FBO->width(), CV_16UC1);
        D->mIrBackground = cv::Mat::zeros(D->mR_DM_FBO->height(), D->mR_DM_FBO->width(), CV_16UC1);

        QString qs;

        QFile fin("dimg_bg.csv");
        if(!fin.open(QIODevice::ReadOnly)) {
            qDebug() << "Failed to open file";
        }
        QTextStream in(&fin);
        int line = 0;
        while(!in.atEnd())
        {
            qs = in.readLine();
            auto l = qs.split(' ');
            for(int i=0; i<l.size(); i++){
                D->mDBackground.at<uint16_t>(D->mR_DM_FBO->height()-1-i, line) = l[i].toInt();
            }
            line++;
        }
        fin.close();
        qDebug() << "Depth background loaded";

        QFile fin2("irimg_bg.csv");
        if(!fin2.open(QIODevice::ReadOnly)) {
            qDebug() << "Failed to open file";
        }
        QTextStream in2(&fin2);
        line = 0;
        while(!in2.atEnd())
        {
            qs = in2.readLine();
            auto l = qs.split(' ');
            for(int i=0; i<l.size(); i++){
                D->mIrBackground.at<uint16_t>(D->mR_DM_FBO->height()-1-i, line) = l[i].toInt();
            }
            line++;
        }
        fin2.close();
        qDebug() << "Infrared background loaded";
    }

    if ( !D->mCB_Context || !D->mCB_Surface || !D->mR_DM_FBO ) {
        if(!D->mCB_Context){
            qDebug() << "mCB_Context fail";
        }
        if(!D->mCB_Surface){
            qDebug() << "mCB_Surface fail";
        }
        if(!D->mR_DM_FBO){
            qDebug() << "mR_DM_FBO fail";
        }

        qWarning() << "NULL context";
        return;
    }

    D->mCB_Context->makeCurrent(D->mCB_Surface);
    auto f = D->mCB_Context->extraFunctions();         //Get the OpenGL functions container
    //qDebug() << f->hasOpenGLFeature(QOpenGLFunctions::MultipleRenderTargets);


    D->mR_DM_FBO->bind();
    //Clear the FBO
    f->glViewport(0,0,D->mR_DM_FBO->width(), D->mR_DM_FBO->height());
    f->glClearColor(0.f,0.f,0.f,0.f);
    f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    f->glEnable(GL_DEPTH_TEST);
    f->glDisable(GL_BLEND);
    f->glEnable(GL_CULL_FACE);
    f->glCullFace(GL_BACK);
    GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    f->glDrawBuffers(2, bufs);

    mLock.lockForRead();
    auto modelPos = D->mModelPos;
    auto eyePos = D->mCam;
    float near_plane = D->mModel_Nearplane;
    float far_plane = D->mModel_Farplane;
    //auto BB_min = D->mModelBB_min;
    //auto BB_max = D->mModelBB_max;
    mLock.unlock();

    QMatrix4x4 view, proj, model0, model1;

    model0.setToIdentity();
    model0.translate(-250.0, 0.0, -100.0);
    model0.rotate(D->model_rotate0, QVector3D(0.0f, 1.0f, 0.0f));

    view.setToIdentity();
    view.lookAt(eyePos,
                modelPos,
                QVector3D(0.f, 1.f, 0.f));

    proj.setToIdentity();
    float verticalAngle = 75.0;
    float aspectRatio = float(D->mR_DM_FBO->width())/D->mR_DM_FBO->height();
    proj.perspective(verticalAngle,
                     aspectRatio,
                     near_plane,
                     far_plane);

    D->mProgram_R_DM->bind();
    D->mProgram_R_DM->enableAttributeArray("a_pos");
    D->mProgram_R_DM->enableAttributeArray("a_norm");

    D->mProgram_R_DM->setUniformValue("m4_mvp", proj * view * model0);  //Set the Model-View-Projection matrix
    const float *n_offset = 3+&D->mModelVAO.data()->at(0);
    D->mProgram_R_DM->setAttributeArray("a_norm", GL_FLOAT, n_offset, 3, 24);
    D->mProgram_R_DM->setAttributeArray("a_pos", GL_FLOAT, D->mModelVAO.data(), 3, 24);

    f->glDrawElements(GL_TRIANGLES, GLsizei(D->mModelF.size()),
                      GL_UNSIGNED_INT, D->mModelF.data());    //Actually draw all the points

    mLock.lockForWrite();
    D->mDepth_Map_color0 = D->mR_DM_FBO->toImage();
    D->mNormal_Map0 = D->mR_DM_FBO->toImage(true, 1);

    //Clear the FBO and draw second depth map
    f->glViewport(0,0,D->mR_DM_FBO->width(), D->mR_DM_FBO->height());
    f->glClearColor(0.f,0.f,0.f,0.f);
    f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    f->glEnable(GL_DEPTH_TEST);
    f->glDisable(GL_BLEND);
    f->glEnable(GL_CULL_FACE);
    f->glCullFace(GL_BACK);
    f->glDrawBuffers(2, bufs);
    model1.setToIdentity();
    model1.translate(-250.0, 0.0, -100.0);
    model1.rotate(D->model_rotate1, QVector3D(0.0f, 1.0f, 0.0f));
    D->mProgram_R_DM->setUniformValue("m4_mvp", proj * view * model1);  //Set the Model-View-Projection matrix
    f->glDrawElements(GL_TRIANGLES, GLsizei(D->mModelF.size()),
                      GL_UNSIGNED_INT, D->mModelF.data());    //Actually draw all the points
    D->mDepth_Map_color1 = D->mR_DM_FBO->toImage();
    D->mNormal_Map1 = D->mR_DM_FBO->toImage(true, 1);

    //Add background
    //float min_depth0=10000, max_depth0=0, min_depth1=10000, max_depth1=0;
    float Kdpower = 3.0, tmpd = 2500.0, dpower = 2.0, tmpir = 10000.0;
    int height = D->mDepth_Map_color0.height();
    int width = D->mDepth_Map_color0.width();
    QMatrix4x4 vp;
    vp.viewport(0, 0, width, height);

    for ( int y = 0; y < height; y++){
        for ( int x = 0; x < width; x++){
            //qDebug() << "row col: " << row << " " << col;
            QColor clrCurrent0(D->mDepth_Map_color0.pixel( x, y ) );
            QColor clrCurrent1(D->mDepth_Map_color1.pixel( x, y ) );
            QVector4D v0 = QVector4D(clrCurrent0.redF(), clrCurrent0.greenF(), clrCurrent0.blueF(), clrCurrent0.alphaF());
            QVector4D v1 = QVector4D(clrCurrent1.redF(), clrCurrent1.greenF(), clrCurrent1.blueF(), clrCurrent1.alphaF());
            float depth0 = D->DecodeFloatRGBA(v0);
            float depth1 = D->DecodeFloatRGBA(v1);
            float dis0, dis1;
            float z_ndc0 = 2.0 * depth0 - 1.0;
            float z_ndc1 = 2.0 * depth1 - 1.0;
            dis0 = 2.0 * near_plane * far_plane / (far_plane + near_plane - z_ndc0 * (far_plane - near_plane));
            dis1 = 2.0 * near_plane * far_plane / (far_plane + near_plane - z_ndc1 * (far_plane - near_plane));
            D->mDepth_Map_color0.setPixelColor(x, y, QColor(clrCurrent0.redF(), clrCurrent0.green(), clrCurrent0.blue(), 255));
            D->mDepth_Map_color1.setPixelColor(x, y, QColor(clrCurrent1.redF(), clrCurrent1.green(), clrCurrent1.blue(), 255));

            if(depth0>0.001){
//                if(dis0<min_depth0){
//                    min_depth0=dis0;
//                } else if(dis0>max_depth0) {
//                    max_depth0=dis0;
//                }
            } else {
                dis0 = 0.0;
            }
            if(depth1>0.001){
//                if(dis1<min_depth1){
//                    min_depth1=dis1;
//                } else if(dis1>max_depth1) {
//                    max_depth1=dis1;
//                }
            } else {
                dis1 = 0.0;
            }

            if(dis0 == 0.0){
                D->mDepth_Map_color0.setPixelColor(x, y, QColor(255.0, 255.0, 255.0, 255.0));
                D->mDepth_Map_dis0.at<uint16_t>(y, x) = D->mDBackground.at<uint16_t>(y, x);
                D->mIr_Map0.at<uint16_t>(y, x) = D->mIrBackground.at<uint16_t>(y, x);
            } else {
                D->mDepth_Map_dis0.at<uint16_t>(y, x) = dis0;

                QVector4D gl_Position = QVector4D(float(x), float(y), depth0, 1.0);
                //qDebug() << "gl_Position: " << gl_Position;
                QVector4D pos = (vp * proj).inverted() * gl_Position;
                //qDebug() << "vp: " << vp;
                //qDebug() << "proj: " << proj;
                //qDebug() << "(vp * proj).inverted(): " << (vp * proj).inverted();
                //qDebug() << "pos: " << pos;
                QVector3D lightPos(-pos.x(), -pos.y(), -pos.z());
                //qDebug() << "lightPos: " << lightPos;
                QVector3D viewDir = lightPos.normalized();
                //qDebug() << "viewDir: " << viewDir;
                QVector3D lightDir = viewDir;
                //qDebug() << "lightDir: " << lightDir;
                QVector3D H = (viewDir + lightDir).normalized();
                //qDebug() << "H: " << H;
                QColor normalc(D->mNormal_Map0.pixel(x, y));
                //qDebug() << "normalc: " << normalc;
                QVector3D a_norm(normalc.redF()*2.0-1.0, normalc.greenF()*2.0-1.0, normalc.blueF()*2.0-1.0);
                //qDebug() << "a_norm: " << a_norm;
                QVector3D normal = QVector3D(QMatrix4x4((view*model0).normalMatrix()) * QVector4D(a_norm));
                //qDebug() << "view.normalMatrix(): " << view.normalMatrix();
                //qDebug() << "normal: " << normal;
                QVector3D N = normal.normalized();
                //qDebug() << "N: " << N;
                float Kd = std::max(QVector3D::dotProduct(H, N), 0.f);
                //qDebug() << "Kd: " << Kd;
                Kd = std::min(Kd, 1.f);
                //qDebug() << "Kd: " << Kd;
                float Ks = pow(Kd, Kdpower);
                //qDebug() << "Ks: " << Ks;
                D->mIr_Map0.at<uint16_t>(y, x) = Ks*pow((tmpd-dis0)/tmpd, dpower)*tmpir;
                //qDebug() << "D->mIr_Map0.at<uint16_t>(y, x): " << D->mIr_Map0.at<uint16_t>(y, x);
            }
            if(dis1 == 0.0){
                D->mDepth_Map_color1.setPixelColor(x, y, QColor(255.0, 255.0, 255.0, 255.0));
                D->mDepth_Map_dis1.at<uint16_t>(y, x) = D->mDBackground.at<uint16_t>(y, x);
                D->mIr_Map1.at<uint16_t>(y, x) = D->mIrBackground.at<uint16_t>(y, x);
            } else {
                D->mDepth_Map_dis1.at<uint16_t>(y, x) = dis1;

                QVector4D gl_Position = QVector4D(float(x), float(y), depth1, 1.0);
                QVector4D pos = (vp * proj).inverted() * gl_Position;
                QVector3D lightPos(-pos.x(), -pos.y(), -pos.z());
                QVector3D viewDir = lightPos.normalized();
                QVector3D lightDir = viewDir;
                QVector3D H = (viewDir + lightDir).normalized();
                QColor normalc(D->mNormal_Map1.pixel(x, y));
                QVector3D a_norm(normalc.redF()*2.0-1.0, normalc.greenF()*2.0-1.0, normalc.blueF()*2.0-1.0);
                QVector3D normal = QVector3D(QMatrix4x4((view*model1).normalMatrix()) * QVector4D(a_norm));
                QVector3D N = normal.normalized();
                float Kd = std::max(QVector3D::dotProduct(H, N), 0.f);
                Kd = std::min(Kd, 1.f);
                float Ks = pow(Kd, Kdpower);
                D->mIr_Map1.at<uint16_t>(y, x) = Ks*pow((tmpd-dis1)/tmpd, dpower)*tmpir;
            }
        }
    }
//    qDebug() << "min_depth0: " << min_depth0 << "max_depth0: " << max_depth0;
//    qDebug() << "min_depth1: " << min_depth1 << "max_depth1: " << max_depth1;
    mLock.unlock();

    D->mProgram_R_DM->disableAttributeArray("a_norm");
    D->mProgram_R_DM->disableAttributeArray("a_pos");
    D->mProgram_R_DM->release();
    D->mR_DM_FBO->release();

    f->glDisable(GL_CULL_FACE);
    f->glDisable(GL_DEPTH_TEST);
    f->glDisable(GL_BLEND);
    D->mCB_Context->doneCurrent();
}

void LP_Plugin_MotionTracking::FunctionalRender_R(QOpenGLContext *ctx, QSurface *surf, QOpenGLFramebufferObject *fbo, const LP_RendererCam &cam, const QVariant &options)
{
    Q_UNUSED(surf)  //Mostly not used within a Functional.
//    Q_UNUSED(options)   //Not used in this functional.

    if ( !mInitialized_R ){   //The OpenGL resources, e.g. Shader, not initilized
        initializeGL_R();     //Call the initialize member function.
    }

    if( !D->mInitialized_R_DM){
        static bool once = [this,
                            surf](){
            D->mCB_Surface = surf;
            D->mCB_Context = new QOpenGLContext();
            D->mCB_Context->moveToThread(this->thread());
            return true;
        }();
    }

    QMatrix4x4 view = cam->ViewMatrix(),
               proj = cam->ProjectionMatrix();

    auto f = ctx->extraFunctions();

    fbo->bind();

    mProgram_R->bind();
    mProgram_R->enableAttributeArray("a_pos");
    mProgram_R->enableAttributeArray("a_norm");
    mProgram_R->setUniformValue("m4_mvp", proj * view );
    mProgram_R->setUniformValue("v4_color", QVector4D(0.2f, 0.2f, 0.2f, 0.6f));
    mProgram_R->setUniformValue("m3_normal", view.normalMatrix());
    mProgram_R->setUniformValue("m4_view", view);

    if(D->mUse_Model){
        const float *n_offset = 3+&D->mModelVAO.data()->at(0);

        mProgram_R->setAttributeArray("a_pos", GL_FLOAT, D->mModelVAO.data(), 3, 24);
        mProgram_R->setAttributeArray("a_norm", GL_FLOAT, n_offset, 3, 24);
        f->glDrawElements(GL_TRIANGLES, GLsizei(D->mModelF.size()),
                          GL_UNSIGNED_INT, D->mModelF.data());
    }

    mProgram_R->disableAttributeArray("a_pos");
    mProgram_R->disableAttributeArray("a_norm");
    mProgram_R->release();

    fbo->release();
}
