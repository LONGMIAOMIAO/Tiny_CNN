#pragma once
#include "PooLayerQL.h"
#include "LoadCSV.h"
#include <algorithm> 
#include <iterator>
namespace tinyDNN
{
	//template <typename Dtype>
	class Pool_Test
	{
	public:
		Pool_Test()
		{
			this->pool_Average_Vector();
		}

		void pool_Average_Vector()
		{
			//测试加载卷积二维图像,训练集
			LoadCSV::loadCSVTrain();
			LoadCSV::loadCSV_Train_Vector();

			//制作输入层，1行784列
			std::shared_ptr<Inter_LayerQL<double>> inLayer_01 = std::make_shared<Inter_LayerQL<double>>(28,28);
			std::copy(LoadCSV::conv_Input_Vector.begin() + 999, LoadCSV::conv_Input_Vector.begin() + 999 + 3, std::back_inserter(inLayer_01->forward_Matrix_Vector));
			std::for_each(inLayer_01->forward_Matrix_Vector.begin(), inLayer_01->forward_Matrix_Vector.end(), [](std::shared_ptr<MatrixQL<double>> matrixTest) { std::cout << ( matrixTest->getMatrixQL()*9).cast<int>() << std::endl; });

			std::shared_ptr<LayerQL<double>> pool_Layer = std::make_shared<PooLayerQL<double>>(Pool_Layer,14,14);
			std::shared_ptr<Inter_LayerQL<double>> inLayer_02 = inLayer_01 + pool_Layer;

			pool_Layer->calForward();

			std::for_each(inLayer_02->forward_Matrix_Vector.begin(), inLayer_02->forward_Matrix_Vector.end(), [](std::shared_ptr<MatrixQL<double>> matrixTest) { std::cout << (matrixTest->getMatrixQL() *9).cast<int>() << std::endl; });

			inLayer_02->backward_Matrix_Vector = inLayer_02->forward_Matrix_Vector;
			
			pool_Layer->calBackward();

			std::for_each(inLayer_01->backward_Matrix_Vector.begin(), inLayer_01->backward_Matrix_Vector.end(), [](std::shared_ptr<MatrixQL<double>> matrixTest) { std::cout << (matrixTest->getMatrixQL() * 9).cast<int>() << std::endl; });

		}

		void pool_Test()
		{
				
		}

		~Pool_Test(){}
	};
}