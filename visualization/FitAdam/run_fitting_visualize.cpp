#include <GL/glew.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <gflags/gflags.h>
#include <vector>
#include <array>
#include <json/json.h>
#include <simple.h>
#include "totalmodel.h"
#include <FitToBody.h>
#include <VisualizedData.h>
#include <Renderer.h>
#include <KinematicModel.h>
#include <cassert>
#include <opencv2/highgui/highgui.hpp>
#include <GL/freeglut.h>
#include <pose_to_transforms.h>
#include "meshTrackingProj.h"
#include "SGSmooth.hpp"
#include "ModelFitter.h"
#include "utils.h"
#include <thread>
#include <boost/filesystem.hpp>

#define ROWS 1080
#define COLS 1920
#define FACE_VERIFY_THRESH 0.05
#define PI 3.14159265359
DEFINE_string(root_dirs, "", "Base root folder to access data");
DEFINE_string(tag, "", "Base root folder to access data");
DEFINE_string(seqName, "default", "Sequence Name to run");
DEFINE_string(speakerName, "default", "Sequence Name to run");

DEFINE_int32(sample_num, 1, "sample_num of output");
DEFINE_int32(start, 1, "Starting frame");
DEFINE_int32(end, 1000, "Ending frame");
DEFINE_bool(densepose, false, "Whether to fit onto result of densepose");
DEFINE_bool(OpenGLactive, false, "Whether to Stay in OpenGLWindow");
DEFINE_bool(euler, false, "True to use Euler angles, false to use angle axis representation");
DEFINE_int32(stage, 1, "Start from which stage.");
DEFINE_bool(imageOF, false, "If true, use image optical flow for the first tracking iteration; if false, always use texture optical flow.");
DEFINE_int32(freeze, 0, "If 1, do not use optical flow below hips; if 2, do not use optical for below chest.");
DEFINE_bool(singleStage, false, "If true, use single stage model fitter.");

TotalModel g_total_model;
double gResultJoint[21 * 3 + 2 * 21 * 3];
std::unique_ptr<Renderer> render = nullptr;
GLubyte ret_bytes[COLS * ROWS * 4];
float ret_depth[COLS * ROWS];

void emptyfunc() {}

void check_flags(int argc, char* argv[])
{
#ifdef GFLAGS_NAMESPACE
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
#else
    google::ParseCommandLineFlags(&argc, &argv, true);
#endif
    std::cout << "Root Directory: " << FLAGS_root_dirs << std::endl;
    std::cout << "Sequence Name: " << FLAGS_seqName << std::endl;
    if (FLAGS_seqName.compare("default") == 0)
    {
        std::cerr << "Error: Sequence Name must be set." << std::endl;
        exit(1);
    }
    if (FLAGS_start >= FLAGS_end)
    {
        std::cerr << "Error: Starting frame must be less than end frame." << std::endl;
        exit(1);
    }
}

void filter_hand_pose(const std::vector<smpl::SMPLParams>& params, std::vector<smpl::SMPLParams*>& batch_refit_params_ptr, uint start_frame)
{
    // run Savitzky-Golay filter on wrist and finger joints of params and copy to batch_refit_params_ptr
    assert(start_frame + batch_refit_params_ptr.size() <= params.size());
    if (batch_refit_params_ptr.size() < 2 * 11 + 2)
    {
        for (uint d = 60; d < TotalModel::NUM_POSE_PARAMETERS; d++)
        {
            for (uint t = 0; t < batch_refit_params_ptr.size(); t++)
                batch_refit_params_ptr[t]->m_adam_pose.data()[d] = params[start_frame + t].m_adam_pose.data()[d];
        }
        return;
    }
    for (uint d = 60; d < TotalModel::NUM_POSE_PARAMETERS; d++)
    {
        const int order = (d < 66) ? 3 : 5;
        std::vector<double> input(batch_refit_params_ptr.size());
        // Eigen::VectorXf input(batch_refit_params_ptr.size());
        for (uint t = 0; t < batch_refit_params_ptr.size(); t++)
            input.data()[t] = params[start_frame + t].m_adam_pose.data()[d];
        // Eigen::RowVectorXf output = savgolfilt(input, order, 21); // make sure the frame number is odd
        auto output = sg_smooth(input, 11, order);
        for (uint t = 0; t < batch_refit_params_ptr.size(); t++)
            batch_refit_params_ptr[t]->m_adam_pose.data()[d] = output.data()[t];
    }
}

int main(int argc, char* argv[])
{
    check_flags(argc, argv);
    bool render_on = true;
    bool save_mesh = false;

    if (render_on) {
        render.reset(new Renderer(&argc, argv));  // initialize the OpenGL renderer
        render->options.meshSolid = true;
        render->options.show_joint = false;
        Renderer::use_color_fbo = true;
    } 

    /*
    Stage 0: read in data
    */
    double calibK[9];  // K Matrix
    const std::string calib_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/calib.json";
    Json::Value json_root;
    std::ifstream f(calib_filename.c_str());
    if (!f.good())
    {
        std::cerr << "Error: Calib file " << calib_filename << " does not exists" << std::endl;
        exit(1);
    }
    f >> json_root;
    calibK[0] = json_root["K"][0u][0u].asDouble(); calibK[1] = json_root["K"][0u][1u].asDouble(); calibK[2] = json_root["K"][0u][2u].asDouble(); calibK[3] = json_root["K"][1u][0u].asDouble(); calibK[4] = json_root["K"][1u][1u].asDouble(); calibK[5] = json_root["K"][1u][2u].asDouble(); calibK[6] = json_root["K"][2u][0u].asDouble(); calibK[7] = json_root["K"][2u][1u].asDouble(); calibK[8] = json_root["K"][2u][2u].asDouble();
    f.close();

    std::vector<std::array<double, 2 * ModelFitter::NUM_KEYPOINTS_2D + 3 * ModelFitter::NUM_PAF_VEC + 2>> net_output;   // read in network output
    

    std::vector<std::vector<cv::Point3i>> dense_constraint;

    // initialize total model
    LoadTotalModelFromObj(g_total_model, std::string("model/mesh_nofeet.obj"));
    LoadModelColorFromObj(g_total_model, std::string("model/nofeetmesh_byTomas_bottom.obj"));  // contain the color information
    LoadTotalDataFromJson(g_total_model, std::string("model/adam_v1_plus2.json"), std::string("model/adam_blendshapes_348_delta_norm.json"), std::string("model/correspondences_nofeet.txt"));
    LoadCocoplusRegressor(g_total_model, std::string("model/regressor_0n1_root.json"));

    if (render_on) {
        render->CameraMode(0);
        render->options.K = calibK;
        glutDisplayFunc(emptyfunc);
        glutMainLoopEvent();
    }

    /*
    Stage 1: run single frame fitting & refitting
    */
    auto dense_constraint_entry = dense_constraint.begin();  // unused
    int image_index = FLAGS_start;
    std::vector<smpl::SMPLParams> params;
    // std::vector<CMeshModelInstance> meshes;

    if (FLAGS_stage == 1)
    {   
        boost::filesystem::create_directories(FLAGS_root_dirs + "/" + FLAGS_seqName + "/results/"+ FLAGS_tag + "predicted_body_3d_frontal/");
        
        ModelFitter model_fitter(g_total_model);
        model_fitter.setCalibK(calibK);
        smpl::SMPLParams refit_params;
        refit_params.m_adam_t.setZero();
        refit_params.m_adam_pose.setZero();
        refit_params.m_adam_coeffs.setZero();
        for (int i=FLAGS_start; i <= FLAGS_end; i++)
        {
            std::cout << "Reproducing image " << image_index << " ----------------" << std::endl;

            char basename[200];
            sprintf(basename, "%04d.txt", image_index+1);
            
            const std::string param_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/results/"+ FLAGS_tag + "predicted_body_3d_frontal/" + basename;
            // std::cout << FLAGS_root_dirs + "/" + FLAGS_seqName + "/results/"+ FLAGS_tag + "predicted_body_3d_frontal/" + basename << std::endl;
            readFrameParam(param_filename, refit_params);

            CMeshModelInstance mesh;
            GenerateMesh(mesh, gResultJoint, refit_params, g_total_model, 2, FLAGS_euler);

            std::string mesh_filename;
            if (save_mesh) {
                mesh_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/results/"+ FLAGS_tag + "predicted_body_3d_frontal/" + basename + ".obj";
                write_adam_obj(mesh, mesh_filename.c_str());
            }
            // comparing keypoints
            int num_joints = 21 * 3 + 2 * 21 * 3;
            std::string keypoint_filename;
            keypoint_filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/results/"+ FLAGS_tag + "predicted_body_3d_frontal/" + basename + "_keypoints.txt";
            
            std::ofstream myfile;
            myfile.open(keypoint_filename);
            for(int j=0; j < num_joints; j++){
                myfile << gResultJoint[j] << " ";
            }
            myfile << "\n";
            myfile.close();
            // 

            if (render_on) {
                VisualizedData vis_data;
                CopyMesh(mesh, vis_data);
                render->options.view_dist = gResultJoint[2 * 3 + 2];
                vis_data.vis_type = 1;

                if (image_index == FLAGS_start)
                {
                    render->CameraMode(0);
                    render->options.K = calibK;
                    render->RenderHand(vis_data);
                    vis_data.read_buffer = ret_bytes;
                    render->RenderAndRead();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                }
                render->CameraMode(0);
                render->options.K = calibK;

                vis_data.resultJoint = gResultJoint;
                render->RenderHand(vis_data);
                if (FLAGS_OpenGLactive) render->Display();

                vis_data.read_buffer = ret_bytes;
                render->RenderAndRead();

                // convert to opencv format
                cv::Mat frame(ROWS, COLS, CV_8UC4, ret_bytes);
                cv::flip(frame, frame, 0);
                // cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);  // convert to BGR

                sprintf(basename, "%s_%08d.png", FLAGS_seqName.c_str(), image_index+1);
                const std::string imgName = FLAGS_root_dirs + "/" + FLAGS_seqName + "/raw_image/" + basename;
                cv::Mat img(ROWS, COLS, CV_8UC3, cv::Scalar(0));
                
                ///// inserting image information
                // std::cout << imgName << std::endl;
                cv::Mat imgr = cv::imread(imgName);
                imgr.copyTo(img.rowRange(0, imgr.rows).colRange(0, imgr.cols));
                // /////
                
                cv::Mat aligned = alignMeshImageAlpha(frame, img);
                // cv::Mat aligned = alignMeshImage(frame, cv::imread(imgName));
                sprintf(basename, "%04d.png", image_index);
                const std::string filename = FLAGS_root_dirs + "/" + FLAGS_seqName + "/results/"+ FLAGS_tag + "predicted_body_3d_frontal/" + basename;
                // std::cout << "OPENING...." << filename << std::endl;
                assert(cv::imwrite(filename, aligned));
            }

            image_index++;
        }
    }
}
