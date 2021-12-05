///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////


// FeatureExtractionPython.cpp : Defines the entry point for the feature extraction python binding.

#include <filesystem>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Local includes
#include "LandmarkCoreIncludes.h"

#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>
#include <ImageManipulationHelpers.h>


namespace py = pybind11;


std::vector<std::string> split_string(std::string str, char delim)
{
	std::string tmp; 
	std::stringstream ss(str);
	std::vector<std::string> words;

	while(getline(ss, tmp, delim))
		words.push_back(tmp);
	return words;
}


template<typename T>
py::array_t<T> mat2vec(cv::Mat_<T> mat)
{
	if (!mat.isContinuous()) 
	{
		mat = mat.clone();
	}
	return py::array_t<T>(mat.total() * mat.channels(), (T*)mat.datastart);
}


class OpenFace
{
private:
	std::unique_ptr<LandmarkDetector::FaceModelParameters> det_parameters;
	std::unique_ptr<FaceAnalysis::FaceAnalyser> face_analyser;

public:
	py::array_t<float> gazeAngle;
	py::array_t<float> pose_estimate;
	py::array_t<int> visibilities;
	std::vector<std::pair<std::string, double>> au_reg;
	std::vector<std::pair<std::string, double>> au_class;
	std::unique_ptr<LandmarkDetector::CLNF> face_model;

	OpenFace(std::string args = "")
	{
		/*
		first argument must be path to directory containing "model" and "AU_predictors" subdirectories
		
		possible additional arguments for LandmarkDetector:
		-mloc, -fdloc, -sigma, -w_reg, -reg, -multi_view, -validate_detections, -n_iter, -wild
		possible additional arguments for FaceAnalyser:
		-au_static, -g, -nomask, -simscale, -simsize
		*/
		std::vector<std::string> arguments{split_string(args, ' ')};
		if (args=="")
		{
			arguments.insert(arguments.begin(), std::filesystem::current_path().string());
		}
		else
		{
			arguments[0] = arguments[0] + "/model";
		}

		// Load the modules that are being used for tracking and face analysis
		// Load face landmark detector
		det_parameters.reset(new LandmarkDetector::FaceModelParameters(arguments));

		det_parameters->refine_hierarchical = false;
		det_parameters->use_face_template = true;
		det_parameters->window_sizes_small[0] = 0;
		det_parameters->window_sizes_small[1] = 7;
		det_parameters->window_sizes_small[2] = 3;
		det_parameters->window_sizes_small[3] = 0;

		face_model.reset(new LandmarkDetector::CLNF(det_parameters->model_location));

		if (!face_model->loaded_successfully)
		{
			std::runtime_error("could not load the landmark detector");
		}

		if (!face_model->eye_model)
		{
			std::runtime_error("no eye model found");
		}

		// Load facial feature extractor and AU analyser
		FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
		face_analyser.reset(new FaceAnalysis::FaceAnalyser(face_analysis_params));

		if (face_analyser->GetAUClassNames().size() == 0 && face_analyser->GetAUClassNames().size() == 0)
		{
			std::runtime_error("no Action Unit models found");
		}
	}


	bool detect(py::array_t<uint8_t> img, std::array<double, 4> camera_intrinsics)
	{
		cv::Mat captured_image(img.shape(0), img.shape(1), CV_8UC3, const_cast<unsigned char*>(img.data()));	// uint8 3 channel!
		auto & [fx, fy, cx, cy] = camera_intrinsics;

		// Converting to grayscale
		cv::Mat_<uchar> grayscale_image;
		Utilities::ConvertToGrayscale_8bit(captured_image, grayscale_image);

		// The actual facial landmark detection / tracking
		bool detection_success = LandmarkDetector::DetectLandmarksInVideo(captured_image, *face_model, *det_parameters, grayscale_image);
		if (!detection_success)
		{
			return false;
		}

		// Gaze tracking, absolute gaze direction
		cv::Point3f gazeDirection0;
		cv::Point3f gazeDirection1;
		GazeAnalysis::EstimateGaze(*face_model, gazeDirection0, fx, fy, cx, cy, true);
		GazeAnalysis::EstimateGaze(*face_model, gazeDirection1, fx, fy, cx, cy, false);
		gazeAngle = mat2vec(cv::Mat_<float>(GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1), false));

		// Perform AU detection
		face_analyser->AddNextFrame(captured_image, face_model->detected_landmarks, face_model->detection_success, 0.0, true);	// we use webcam/online mode, so timestamp is not needed
		au_reg = face_analyser->GetCurrentAUsReg();
		au_class = face_analyser->GetCurrentAUsClass();

		// Work out the pose of the head from the tracked model
		pose_estimate = mat2vec(cv::Mat_<float>(LandmarkDetector::GetPose(*face_model, fx, fy, cx, cy), false));

		visibilities = mat2vec(face_model->GetVisibilities());
		
		return true;
	}
};


PYBIND11_MODULE(openface, m) {
	m.doc() = "basic Openface python binding";

	py::class_<OpenFace>(m, "OpenFace")
		.def(py::init<std::string>(), py::arg("arguments") = "", "First argument passed in must be path to directory containing 'model' and 'AU_predictors' subdirectories. Optionally some additionally arguments can get passed to LandmarkDetector and FaceAnalyser.")
		.def("detect", &OpenFace::detect, py::arg("image"), py::arg("camera_intrinsics"), "Image must have 3 channels and type uint8. camera_intrinsics=[fx, fy, cx, cy]")
		.def_readonly("pose", &OpenFace::pose_estimate)
		.def_readonly("gaze", &OpenFace::gazeAngle)
		.def_readonly("au", &OpenFace::au_reg)
		.def_readonly("au_binary", &OpenFace::au_class)
		.def_readonly("landmark_visibility", &OpenFace::visibilities)
		.def_property_readonly("landmark_data", [](const OpenFace & o){return mat2vec(o.face_model->detected_landmarks);})
		.def_property_readonly("confidence", [](const OpenFace & o){return o.face_model->detection_certainty;});
}

