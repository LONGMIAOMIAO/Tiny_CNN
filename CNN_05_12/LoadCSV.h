#pragma once

#include <iostream>  
#include <string>  
#include <vector>  
#include <fstream>  
#include <sstream>  
#include "Test.h"

namespace tinyDNN
{
	class LoadCSV
	{
	public:
		using MatrixD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

		LoadCSV();
		~LoadCSV();

		//**********************************************************

		//装载一维Mnist训练集
		static void loadCSVTrain();
		//装载一维Mnist测试集
		static void loadCSVTest();
		//将训练集图片转换为Vector图片类型
		static void loadCSV_Train_Vector();
		//将测试集图片转换为Vector图片类型
		static void loadCSV_Test_Vector();

		//**********************************************************

		static std::shared_ptr<Inter_LayerQL<double>> input_Layer;
		static std::shared_ptr<Inter_LayerQL<double>> output_Layer;

		static std::shared_ptr<Inter_LayerQL<double>> input_Layer_T;
		static std::shared_ptr<Inter_LayerQL<double>> output_Layer_T;

		//**********************************************************

		static std::vector<std::shared_ptr<MatrixQL<double>>> conv_Input_Vector;
		static std::vector<std::shared_ptr<MatrixQL<double>>> conv_Input_Vector_T;

	};

	class LoadCifar_10
	{
	public:
		static void loadCifar_10_Train();

		static std::vector< std::vector< std::shared_ptr<MatrixQL<double> > > > cifar_Input_Vector;
		static std::shared_ptr< MatrixQL<double> > cifar_Out_Lable;

		static std::vector< std::vector< std::shared_ptr<MatrixQL<double> > > > cifar_Input_Vector_T;
		static std::shared_ptr< MatrixQL<double> > cifar_Out_Lable_T;
	};

	class LoadTuLi
	{
	public:
		static void load_Tuli();
		static void load_Tuli_T();

		static std::vector<std::shared_ptr<MatrixQL<double>>> tuli_Train;
		static std::vector<std::shared_ptr<MatrixQL<double>>> tuli_Test;
	};
}
