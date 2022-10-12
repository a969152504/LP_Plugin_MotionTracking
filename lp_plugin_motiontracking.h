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
                         k4a_calibration_t sensorCalibration, int dev_ind);
    void keyReleaseEvent(QKeyEvent *event);
    r_t rigid_transform_3D(cv::Mat A, cv::Mat B);
    void rigid_transform_3D_test();

    void save_samples();

    void loadObj(const char *filename,
                 std::vector<std::array<float, 6>> &vao,
                 std::vector<uint> &f);
    void save_obj(std::string path, cv::Mat mat);
    void save_obj(std::string path, k4abt_skeleton_t *ske);

    QFuture<void> gFuture;

    bool mSave_samples = true;
    std::string mCamtime_path = "/home/cpii/Desktop/test_img/not_syn_time.csv";
    std::string mTimestamps_path = "/home/cpii/Desktop/test_img/time_stamps.csv";
    std::string mJointposi_path = "/home/cpii/Desktop/test_img/joint_posi.csv";
    std::string mJointvelo_path = "/home/cpii/Desktop/test_img/joint_velo.csv";
    std::string mJointacce_path = "/home/cpii/Desktop/test_img/joint_acce.csv";
    std::string mCam0_path = "/home/cpii/Desktop/test_img/cam0/";
    std::string mCam0obj_path = "/home/cpii/Desktop/test_img/cam0obj/";
    std::string mCam1_path = "/home/cpii/Desktop/test_img/cam1/";
    std::string mCam1obj_path = "/home/cpii/Desktop/test_img/cam1obj/";
    std::string mSke0_path = "/home/cpii/Desktop/test_img/ske0/";
    std::string mSke0obj_path = "/home/cpii/Desktop/test_img/ske0obj/";
    std::string mSke1_path = "/home/cpii/Desktop/test_img/ske1/";
    std::string mSke1obj_path = "/home/cpii/Desktop/test_img/ske1obj/";
    std::string mCam0ir_path = "/home/cpii/Desktop/test_img/cam0ir/";
    std::string mCam1ir_path = "/home/cpii/Desktop/test_img/cam1ir/";

    int mCam_rows = 0, mCam_cols = 0, mDepthh = 0, mDepthw = 0;
    std::vector<k4a_calibration_t> sensorCalibrations;

    uint64_t mTmpFrametime = 0;
    cv::Mat mTmpMat, mTmpirMat;
    std::vector<uint64_t> cam0_t, cam1_t;

    // SkeOpt
    void SkeletonoperationInit(); // Initialize the "SkeletonOpt" Object
    void SkeletonoperationRunoptimization(int use_cam);
    void checking_estpoint_new(k4abt_joint_t **skea, k4abt_joint_t **skeb);
    void cam_prior_new(k4abt_joint_t **skea, k4abt_joint_t **skeb,
                       int jointpoint);
    double smooth_step(double t);
    int choosing_dis(k4abt_joint_t **skea, k4abt_joint_t **skeb, int jointpoint);

    bool mGetRest = false, mUse_Opt = false, mInitSKO = false, got_skea = false, got_skeb = false;
    int mAvg_count = 0, optfps_count = 0;
    cv::Mat mKinectA_pts, mKinectB_pts;
    k4abt_skeleton_t mRest_skeleton, tmpbody_a, tmpbody_b, lastbody_b;
    int mJoint_tmp_num = 0, joint_a_num = 0, joint_b_num = 0, mBone_tmp_num = 0, bone_a_num = 0, bone_b_num = 0;

    std::vector<std::vector<k4a_float3_t>> mJoint_posis, mJoint_velo;
    std::vector<float> mTimestamps;

    SkeletonOpt *skeOpt = nullptr; // class for handing the optimization. Referring to
                                   // attached SkeletonOpt.h and SkeletonOpt.cpp
    std::vector<double> length_const, weighting_a, weighting_b;
};

#endif // LP_PLUGIN_MOTIONTRACKING_H
