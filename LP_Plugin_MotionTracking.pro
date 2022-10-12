QT += gui

TEMPLATE = lib
DEFINES += LP_PLUGIN_MOTIONTRACKING_LIBRARY

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14

QMAKE_POST_LINK=$(MAKE) install

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    SkeletonOpt.cpp \
    lp_plugin_motiontracking.cpp

HEADERS += \
    LP_Plugin_MotionTracking_global.h \
    lp_plugin_motiontracking.h \
    BodyTrackingHelpers.h \
    SkeletonOpt.h \
    Utilities.h

# Default rules for deployment.
unix {
    target.path = /usr/lib
}

# Default rules for deployment.
target.path = $$OUT_PWD/../App/plugins/$$TARGET

!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../Model/release/ -lModel
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../Model/debug/ -lModel
else:unix:!macx: LIBS += -L$$OUT_PWD/../Model/ -lModel

INCLUDEPATH += $$PWD/../Model
DEPENDPATH += $$PWD/../Model

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../Functional/release/ -lFunctional
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../Functional/debug/ -lFunctional
else:unix:!macx: LIBS += -L$$OUT_PWD/../Functional/ -lFunctional

INCLUDEPATH += $$PWD/../Functional
DEPENDPATH += $$PWD/../Functional

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../OpenMesh/lib/ -lOpenMeshCore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../OpenMesh/lib/ -lOpenMeshCored
else:unix:!macx: LIBS += -L$$PWD/../../OpenMesh/lib/ -lOpenMeshCore

INCLUDEPATH += $$PWD/../../OpenMesh/include
DEPENDPATH += $$PWD/../../OpenMesh/include

INCLUDEPATH += $$PWD/home/cpii/Downloads/Azure-Kinect-Sensor-SDK/include
DEPENDPATH += $$PWD/home/cpii/Downloads/Azure-Kinect-Sensor-SDK/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../opencv/opencv-4.5.2/install/lib/release/ -lopencv_core
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../opencv/opencv-4.5.2/install/lib/debug/ -lopencv_core
else:unix: LIBS += -L$$PWD/../../opencv/opencv-4.5.2/install/lib/ \
    -lopencv_core \
    -lopencv_highgui \
    -lopencv_imgcodecs \
    -lopencv_imgproc

INCLUDEPATH += $$PWD/../../../opencv/opencv-4.5.2/install/include/opencv4
DEPENDPATH += $$PWD/../../../opencv/opencv-4.5.2/install/include/opencv4

LIBS += -lk4abt -lk4a
