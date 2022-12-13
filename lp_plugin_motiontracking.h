#ifndef LP_PLUGIN_MOTIONTRACKING_H
#define LP_PLUGIN_MOTIONTRACKING_H

#include "plugin/lp_actionplugin.h"

#include <QDebug>
#include <QImage>
#include <QMainWindow>
#include <QReadWriteLock>
#include <QVector4D>
#include <QtConcurrent/QtConcurrent>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMatrix4x4>
#include <QOpenGLExtraFunctions>

#include "opencv2/aruco.hpp"
#include "opencv2/aruco/charuco.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

#include <k4a/k4a.h>
#include <k4abt.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>

#include <BodyTrackingHelpers.h>
#include <Utilities.h>

#include <SkeletonOpt.h>
#include <kinect_azure_functions.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <thread>
#include <fstream>

#define LP_Plugin_MotionTracking_iid "cpii.rp5.SmartFashion.LP_Plugin_MotionTracking/0.1"

class LP_Plugin_MotionTracking : public LP_ActionPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID LP_Plugin_MotionTracking_iid)
    Q_INTERFACES(LP_ActionPlugin)
public:
    virtual ~LP_Plugin_MotionTracking();

    // LP_Functional interface
    QWidget *DockUi();

    // LP_ActionPlugin interface
    bool Run();
    QString MenuName();
    QAction *Trigger();

private:
    std::shared_ptr<QWidget> mWidget;
    QLabel *mLabel = nullptr;
    QLabel *mLabel2 = nullptr;
    QPushButton *mButton0 = nullptr;
    QPushButton *mButton1 = nullptr;
    QPushButton *mButton2 = nullptr;
    QPushButton *mButton3 = nullptr;
    QPushButton *mButton4 = nullptr;
    QOpenGLShaderProgram *mProgram_R = nullptr;

    bool mInitialized_R = false;
    void initializeGL_R();

    bool s_isRunning = false;
    QImage mImage0, mImage1, mImageske;
    QImage mDImage0, mDImage1;
    QImage mIrImage0, mIrImage1;
    QReadWriteLock mLock;

    struct r_t
    {
        cv::Mat R; // 3x3 rotation matrix
        cv::Mat t; // 3x1 column vector
    };
    r_t mRet;

    struct Model_member;
    std::shared_ptr<Model_member> D;

public slots:
    void FunctionalRender_R(QOpenGLContext *ctx, QSurface *surf, QOpenGLFramebufferObject *fbo, const LP_RendererCam &cam, const QVariant &options);
    void PainterDraw(QWidget *glW);
    void Get_depthmap();

protected:
    void RunCamera();
    void VisualizeResult(k4abt_frame_t bodyFrame,
                         k4a_calibration_t sensorCalibration,
                         k4a_transformation_t transformation,
                         int dev_ind);
    void keyReleaseEvent(QKeyEvent *event);
    r_t rigid_transform_3D(cv::Mat A, cv::Mat B);
    void rigid_transform_3D_test();

    void save_Rt(cv::Mat Aligned_skeb);
    void save_samples();

    void create_paths();
    void loadObj(const char *filename,
                 std::vector<std::array<float, 6>> &vao,
                 std::vector<uint> &f);
    void save_obj(std::string path, cv::Mat mat);
    void save_obj(std::string path, k4abt_skeleton_t *ske);
    struct SkeData
    {
        uint64_t StartTime = 0;
        std::vector<cv::Mat> colorMats0, colorMats1, colorMatoris0, colorMatoris1, irMats0, irMats1, optskeMats0, optskeMats1;
        //std::vector<cv::Point> ELBOW_LEFT0, WRIST_LEFT0, ELBOW_LEFT1, WRIST_LEFT1;
        std::vector<k4abt_skeleton_t> skes0, skes1, skes1ori, optskes0, optskes1;
        std::vector<uint64_t> cam0_t, cam1_t, cam0_t_ori, cam1_t_ori;
        std::vector<float> Timestamps;
        std::vector<std::vector<k4a_float3_t>> Joint_posis, Joint_velo;
    };
    SkeData mSkeData;

    bool mSave_samples = true;
    std::string mCamtime_path = "/home/cpii/Desktop/test_img/not_syn_time.csv";
    std::string mCamtimeori_path = "/home/cpii/Desktop/test_img/not_syn_time_ori.csv";
    std::string mTimestamps_path = "/home/cpii/Desktop/test_img/time_stamps.csv";
    std::string mJointposi_path = "/home/cpii/Desktop/test_img/joint_posi.csv";
    std::string mJointvelo_path = "/home/cpii/Desktop/test_img/joint_velo.csv";
    std::string mJointacce_path = "/home/cpii/Desktop/test_img/joint_acce.csv";
    std::string mCam0_path = "/home/cpii/Desktop/test_img/cam0/";
    std::string mCam0ori_path = "/home/cpii/Desktop/test_img/cam0ori/";
    std::string mCam0ske_path = "/home/cpii/Desktop/test_img/cam0ske/";
    std::string mCam1_path = "/home/cpii/Desktop/test_img/cam1/";
    std::string mCam1ori_path = "/home/cpii/Desktop/test_img/cam1ori/";
    std::string mCam1ske_path = "/home/cpii/Desktop/test_img/cam1ske/";
    std::string mCam1skeori_path = "/home/cpii/Desktop/test_img/cam1skeori/";
    std::string mSke_path = "/home/cpii/Desktop/test_img/ske/";
    std::string mSke0_path = "/home/cpii/Desktop/test_img/ske0/";
    std::string mSke0obj_path = "/home/cpii/Desktop/test_img/ske0obj/";
    std::string mSke1_path = "/home/cpii/Desktop/test_img/ske1/";
    std::string mSke1obj_path = "/home/cpii/Desktop/test_img/ske1obj/";
    std::string mCam0ir_path = "/home/cpii/Desktop/test_img/cam0ir/";
    std::string mCam1ir_path = "/home/cpii/Desktop/test_img/cam1ir/";
    //std::string mPointcloud_path0 = "/home/cpii/Desktop/test_img/Pointcloud0/";
    //std::string mPointcloud_path1 = "/home/cpii/Desktop/test_img/Pointcloud1/";
    //std::string m2dPoint_path = "/home/cpii/Desktop/test_img/2dPoint.csv";

    QFuture<void> gFuture;

    int mCam_rows = 0, mCam_cols = 0, mDepthh = 0, mDepthw = 0;
    std::vector<k4a_calibration_t> sensorCalibrations;

    uint64_t mTmpFrametime = 0;
    bool init_tmpmat = false;
    cv::Mat mTmpMatA, mTmpMatB, mTmpMatOriA, mTmpMatOriB, mTmpirMatA, mTmpirMatB;
    cv::Mat mTmpMatA_Last, mTmpMatB_Last, mTmpMatOriA_Last, mTmpMatOriB_Last, mTmpirMatA_Last, mTmpirMatB_Last;

    //std::shared_ptr<Kinect_Azure_Functions> KAfunctions;

    // SkeOpt
    void SkeletonoperationInit(); // Initialize the "SkeletonOpt" Object
    void SkeletonoperationRunoptimization(int use_cam);
    void SkeletonoperationRunoptimizationall(int use_cam, int frame);
    void checking_estpoint_new(k4abt_joint_t **skea, k4abt_joint_t **skeb, int use_cam);
    void cam_prior_new(k4abt_joint_t **skea, k4abt_joint_t **skeb, int jointpoint, int use_cam);
    double smooth_step(double t);
    int choosing_dis(k4abt_joint_t **skea, k4abt_joint_t **skeb, int jointpoint);

    bool mGetRest = false, mUse_Opt = false, mInitSKO = false, got_skea = false, got_skeb = false;
    int mAvg_count = 0, optfps_count = 0;
    cv::Mat mKinectA_pts, mKinectB_pts;
    k4abt_skeleton_t mRest_skeleton, tmpbody_a, tmpbody_b, lastbody_b;
    std::vector<k4abt_skeleton_t> bodyb_all;
    int mJoint_tmp_num = 0, joint_a_num = 0, joint_b_num = 0, mBone_tmp_num = 0, bone_a_num = 0, bone_b_num = 0;

    std::shared_ptr<SkeletonOpt> skeOpt = nullptr;
    // class for handing the optimization. Referring to attached SkeletonOpt.h and SkeletonOpt.cpp
    std::vector<double> weighting_a, weighting_b;

    // Joint Trajectory
    struct poly_solu
    {
        float a0;
        float a1;
        float a2;
        float a3;
    };
    struct poly_solu_xyz
    {
        poly_solu x;
        poly_solu y;
        poly_solu z;
    };
    struct Joint_Traj
    {
        int TrajFrame_Size = 5;
        uint64_t StartTime = 0;
        std::vector<k4abt_skeleton_t> Traj_skesB;
        std::vector<std::vector<k4a_float3_t>> Traj_velB;
        std::vector<uint64_t> Traj_timestampsA, Traj_timestampsB;
        int tmpf = 1;
    };
    Joint_Traj mJoint_Traj;
    void Process_JointTraj();
    void erase_traj();
    void Process_AllJointTraj();
    poly_solu_xyz solve_polynomial_trajectory(k4a_float3_t po0, k4a_float3_t po1,
                                              k4a_float3_t vel0, k4a_float3_t vel1,
                                              float times0, float times1);
    void cal_xyz(poly_solu_xyz solu, float time, int joint);
};

#endif // LP_PLUGIN_MOTIONTRACKING_H
