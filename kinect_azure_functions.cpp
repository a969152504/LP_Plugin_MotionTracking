#include "kinect_azure_functions.h"

#include <k4a/k4a.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <QDebug>

Kinect_Azure_Functions::Kinect_Azure_Functions()
{
    clear();
    point_cloud_image0.resize(150);
    point_cloud_image1.resize(150);
    transformed_color_image0.resize(150);
    transformed_color_image1.resize(150);
}

Kinect_Azure_Functions::~Kinect_Azure_Functions(void){
    clear();
    for(int i=0; i<point_cloud_image0.size(); i++){
        k4a_image_release(point_cloud_image0[i]);
        k4a_image_release(point_cloud_image1[i]);
        k4a_image_release(transformed_color_image0[i]);
        k4a_image_release(transformed_color_image1[i]);
    }
    point_cloud_image0.clear();
    point_cloud_image1.clear();
    transformed_color_image0.clear();
    transformed_color_image1.clear();
}

struct color_point_t
{
    int16_t xyz[3];
    uint8_t rgb[3];
};


void Kinect_Azure_Functions::tranformation_helpers_write_point_cloud(const k4a_image_t point_cloud_image,
                                                                     const k4a_image_t color_image,
                                                                     const char *file_name)
{
    std::vector<color_point_t> points;

    int width = k4a_image_get_width_pixels(point_cloud_image);
    int height = k4a_image_get_height_pixels(color_image);

    int16_t *point_cloud_image_data = (int16_t *)(void *)k4a_image_get_buffer(point_cloud_image);
    uint8_t *color_image_data = k4a_image_get_buffer(color_image);

    for (int i = 0; i < width * height; i++)
    {
        color_point_t point;
        point.xyz[0] = point_cloud_image_data[3 * i + 0];
        point.xyz[1] = point_cloud_image_data[3 * i + 1];
        point.xyz[2] = point_cloud_image_data[3 * i + 2];
        if (point.xyz[2] == 0)
        {
            continue;
        }

        point.rgb[0] = color_image_data[4 * i + 0];
        point.rgb[1] = color_image_data[4 * i + 1];
        point.rgb[2] = color_image_data[4 * i + 2];
        uint8_t alpha = color_image_data[4 * i + 3];

        if (point.rgb[0] == 0 && point.rgb[1] == 0 && point.rgb[2] == 0 && alpha == 0)
        {
            continue;
        }

        points.push_back(point);
    }

#define PLY_START_HEADER "ply"
#define PLY_END_HEADER "end_header"
#define PLY_ASCII "format ascii 1.0"
#define PLY_ELEMENT_VERTEX "element vertex"

    // save to the ply file
    std::ofstream ofs(file_name); // text mode first
    ofs << PLY_START_HEADER << std::endl;
    ofs << PLY_ASCII << std::endl;
    ofs << PLY_ELEMENT_VERTEX << " " << points.size() << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "property uchar red" << std::endl;
    ofs << "property uchar green" << std::endl;
    ofs << "property uchar blue" << std::endl;
    ofs << PLY_END_HEADER << std::endl;
    ofs.close();

    std::stringstream ss;
    for (size_t i = 0; i < points.size(); ++i)
    {
        // image data is BGR
        ss << (float)points[i].xyz[0] << " " << (float)points[i].xyz[1] << " " << (float)points[i].xyz[2];
        ss << " " << (float)points[i].rgb[2] << " " << (float)points[i].rgb[1] << " " << (float)points[i].rgb[0];
        ss << std::endl;
    }
    std::ofstream ofs_text(file_name, std::ios::out | std::ios::app);
    ofs_text.write(ss.str().c_str(), (std::streamsize)ss.str().length());
}

bool Kinect_Azure_Functions::point_cloud_color_to_depth(k4a_transformation_t transformation_handle,
                                                        const k4a_image_t depth_image,
                                                        const k4a_image_t color_image,
                                                        int dev_ind)
{
    if(!mInit){
        depth_image_width_pixels = k4a_image_get_width_pixels(depth_image);
        depth_image_height_pixels = k4a_image_get_height_pixels(depth_image);
        mInit = true;
    }

    if(dev_ind == 0){
        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                     depth_image_width_pixels,
                                                     depth_image_height_pixels,
                                                     depth_image_width_pixels * 4 * (int)sizeof(uint8_t),
                                                     &tmpTransformed_color_image0))
        {
            printf("Failed to create transformed color image0\n");
            return false;
        }
        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                                                     depth_image_width_pixels,
                                                     depth_image_height_pixels,
                                                     depth_image_width_pixels * 3 * (int)sizeof(int16_t),
                                                     &tmpPoint_cloud_image0))
        {
            printf("Failed to create point cloud image0\n");
            return false;
        }
        if (K4A_RESULT_SUCCEEDED != k4a_transformation_color_image_to_depth_camera(transformation_handle,
                                                                                   depth_image,
                                                                                   color_image,
                                                                                   tmpTransformed_color_image0))
        {
            printf("Failed to compute transformed color image0\n");
            return false;
        }

        if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(transformation_handle,
                                                                                  depth_image,
                                                                                  K4A_CALIBRATION_TYPE_DEPTH,
                                                                                  tmpPoint_cloud_image0))
        {
            printf("Failed to compute point cloud0\n");
            return false;
        }
    } else {
        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                     depth_image_width_pixels,
                                                     depth_image_height_pixels,
                                                     depth_image_width_pixels * 4 * (int)sizeof(uint8_t),
                                                     &tmpTransformed_color_image1))
        {
            printf("Failed to create transformed color image1\n");
            return false;
        }

        if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                                                     depth_image_width_pixels,
                                                     depth_image_height_pixels,
                                                     depth_image_width_pixels * 3 * (int)sizeof(int16_t),
                                                     &tmpPoint_cloud_image1))
        {
            printf("Failed to create point cloud image1\n");
            return false;
        }
        if (K4A_RESULT_SUCCEEDED != k4a_transformation_color_image_to_depth_camera(transformation_handle,
                                                                                   depth_image,
                                                                                   color_image,
                                                                                   tmpTransformed_color_image1))
        {
            printf("Failed to compute transformed color image1\n");
            return false;
        }

        if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(transformation_handle,
                                                                                  depth_image,
                                                                                  K4A_CALIBRATION_TYPE_DEPTH,
                                                                                  tmpPoint_cloud_image1))
        {
            printf("Failed to compute point cloud1\n");
            return false;
        }
    }
    return true;
}

void Kinect_Azure_Functions::write_point_cloud(std::string path0,
                                               std::string path1){
    qDebug() << "Saving point clouds";
    for(int i=0; i<point_cloud_image0.size(); i++){
        std::string file_name0 = path0 + std::to_string(i*2) + ".ply";
        std::string file_name1 = path1 + std::to_string(i*2+1) + ".ply";
        tranformation_helpers_write_point_cloud(point_cloud_image0[i], transformed_color_image0[i], file_name0.c_str());
        tranformation_helpers_write_point_cloud(point_cloud_image1[i], transformed_color_image1[i], file_name1.c_str());
        qDebug() << "[" << i+1 << "/" << point_cloud_image0.size() << "]";
    }
    for(int i=0; i<point_cloud_image0.size(); i++){
        k4a_image_release(point_cloud_image0[i]);
        k4a_image_release(point_cloud_image1[i]);
        k4a_image_release(transformed_color_image0[i]);
        k4a_image_release(transformed_color_image1[i]);
    }
    point_cloud_image0.clear();
    point_cloud_image1.clear();
    transformed_color_image0.clear();
    transformed_color_image1.clear();
}

void Kinect_Azure_Functions::clear(){
    tmpTransformed_color_image0 = NULL;
    tmpTransformed_color_image1 = NULL;
    tmpPoint_cloud_image0 = NULL;
    tmpPoint_cloud_image1 = NULL;
}

void Kinect_Azure_Functions::q_create_xy_table(const k4a_calibration_t *calibration, k4a_image_t xy_table)
{
    k4a_float2_t *table_data = (k4a_float2_t *)(void *)k4a_image_get_buffer(xy_table);

    int width = calibration->depth_camera_calibration.resolution_width;
    int height = calibration->depth_camera_calibration.resolution_height;

    k4a_float2_t p;
    k4a_float3_t ray;
    int valid;

    for (int y = 0, idx = 0; y < height; y++)
    {
        p.xy.y = (float)y;
        for (int x = 0; x < width; x++, idx++)
        {
            p.xy.x = (float)x;

            k4a_calibration_2d_to_3d(
                calibration, &p, 1.f, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &ray, &valid);

            if (valid)
            {
                table_data[idx].xy.x = ray.xyz.x;
                table_data[idx].xy.y = ray.xyz.y;
            }
            else
            {
                table_data[idx].xy.x = nanf("");
                table_data[idx].xy.y = nanf("");
            }
        }
    }
}

void Kinect_Azure_Functions::q_generate_point_cloud(const k4a_image_t depth_image,
                                                  const k4a_image_t xy_table,
                                                  k4a_image_t point_cloud,
                                                  int *point_count)
{
    int width = k4a_image_get_width_pixels(depth_image);
    int height = k4a_image_get_height_pixels(depth_image);

    uint16_t *depth_data = (uint16_t *)(void *)k4a_image_get_buffer(depth_image);
    k4a_float2_t *xy_table_data = (k4a_float2_t *)(void *)k4a_image_get_buffer(xy_table);
    k4a_float3_t *point_cloud_data = (k4a_float3_t *)(void *)k4a_image_get_buffer(point_cloud);

    *point_count = 0;
    for (int i = 0; i < width * height; i++)
    {
        if (depth_data[i] != 0 && !isnan(xy_table_data[i].xy.x) && !isnan(xy_table_data[i].xy.y))
        {
            point_cloud_data[i].xyz.x = xy_table_data[i].xy.x * (float)depth_data[i];
            point_cloud_data[i].xyz.y = xy_table_data[i].xy.y * (float)depth_data[i];
            point_cloud_data[i].xyz.z = (float)depth_data[i];
            (*point_count)++;
        }
        else
        {
            point_cloud_data[i].xyz.x = nanf("");
            point_cloud_data[i].xyz.y = nanf("");
            point_cloud_data[i].xyz.z = nanf("");
        }
    }
}

void Kinect_Azure_Functions::q_write_point_cloud(const char *file_name, const k4a_image_t point_cloud, int point_count)
{
    int width = k4a_image_get_width_pixels(point_cloud);
    int height = k4a_image_get_height_pixels(point_cloud);

    k4a_float3_t *point_cloud_data = (k4a_float3_t *)(void *)k4a_image_get_buffer(point_cloud);

    // save to the ply file
    std::ofstream ofs(file_name); // text mode first
    ofs << "ply" << std::endl;
    ofs << "format ascii 1.0" << std::endl;
    ofs << "element vertex"
        << " " << point_count << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "end_header" << std::endl;
    ofs.close();

    std::stringstream ss;
    for (int i = 0; i < width * height; i++)
    {
        if (isnan(point_cloud_data[i].xyz.x) || isnan(point_cloud_data[i].xyz.y) || isnan(point_cloud_data[i].xyz.z))
        {
            continue;
        }

        ss << (float)point_cloud_data[i].xyz.x << " " << (float)point_cloud_data[i].xyz.y << " "
           << (float)point_cloud_data[i].xyz.z << std::endl;
    }

    std::ofstream ofs_text(file_name, std::ios::out | std::ios::app);
    ofs_text.write(ss.str().c_str(), (std::streamsize)ss.str().length());
}
