#ifndef KINECT_AZURE_FUNCTIONS_H
#define KINECT_AZURE_FUNCTIONS_H

#include <iostream>
#include <vector>

#include <k4a/k4a.h>
#include <k4abt.h>

class Kinect_Azure_Functions
{
public:
    Kinect_Azure_Functions();
    ~Kinect_Azure_Functions(void);

    bool point_cloud_color_to_depth(k4a_transformation_t transformation_handle,
                                    const k4a_image_t depth_image,
                                    const k4a_image_t color_image,
                                    int dev_ind);
    void tranformation_helpers_write_point_cloud(const k4a_image_t point_cloud_image,
                                                 const k4a_image_t color_image,
                                                 const char *file_name);
    void write_point_cloud(std::string path0,
                           std::string path1);
    void clear();

    void q_generate_point_cloud(const k4a_image_t depth_image,
                                const k4a_image_t xy_table,
                                k4a_image_t point_cloud,
                                int *point_count);
    void q_create_xy_table(const k4a_calibration_t *calibration, k4a_image_t xy_table);
    void q_write_point_cloud(const char *file_name, const k4a_image_t point_cloud, int point_count);

    bool mInit = false;
    int depth_image_width_pixels = 0, depth_image_height_pixels = 0;
    k4a_image_t tmpTransformed_color_image0, tmpTransformed_color_image1;
    k4a_image_t tmpPoint_cloud_image0, tmpPoint_cloud_image1;
    std::vector<k4a_image_t> point_cloud_image0, point_cloud_image1;
    std::vector<k4a_image_t> transformed_color_image0, transformed_color_image1;
};

#endif // KINECT_AZURE_FUNCTIONS_H
