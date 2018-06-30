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
			//this->pool_Average_Vector();
			this->pool_Average_Vector_Normal();
		}

		void pool_Average_Vector_Normal()
		{
			//制作输入层
			std::shared_ptr<Inter_LayerQL<double>> inLayer_01 = std::make_shared<Inter_LayerQL<double>>(6, 6);
			for (int i = 0; i < 3; i++)
			{
				std::shared_ptr<MatrixQL<double>> oneM = std::make_shared<MatrixQL<double>>(6, 6);
				double startNum = 0.1 * (i + 1);
				for (int p = 0; p < 6; p++)
				{
					for (int q = 0; q < 6; q++)
					{
						oneM->setMatrixQL()(p, q) = startNum;
						startNum = startNum + 0.1 * (i + 1);
					}
				}
				//this->conv_Kernel_Vector.push_back(oneSlice_Kernel);
				inLayer_01->forward_Matrix_Vector.push_back(oneM);
			}

			std::shared_ptr<LayerQL<double>> pool_Layer = std::make_shared<PooLayerQL<double>>(Pool_Layer, 3, 3);

			std::shared_ptr<Inter_LayerQL<double>> inLayer_02 = inLayer_01 + pool_Layer;

			int reTime = 3;
			while (reTime > 0)
			{
				std::for_each(inLayer_01->forward_Matrix_Vector.begin(), inLayer_01->forward_Matrix_Vector.end(), [](std::shared_ptr<MatrixQL<double>> matrixTest) { std::cout << matrixTest->getMatrixQL() << std::endl; std::cout << "池化前**********************" << std::endl; });
				pool_Layer->calForward();
				std::for_each(inLayer_02->forward_Matrix_Vector.begin(), inLayer_02->forward_Matrix_Vector.end(), [](std::shared_ptr<MatrixQL<double>> matrixTest) { std::cout << matrixTest->getMatrixQL() << std::endl;  std::cout << "池化后**********************" << std::endl; });
				reTime--;
			}


			inLayer_02->backward_Matrix_Vector = inLayer_02->forward_Matrix_Vector;

			int reTime_02 = 3;
			while (reTime_02 > 0)
			{
				pool_Layer->calBackward();
				std::for_each(inLayer_01->backward_Matrix_Vector.begin(), inLayer_01->backward_Matrix_Vector.end(), [](std::shared_ptr<MatrixQL<double>> matrixTest) { std::cout << (matrixTest->getMatrixQL()) << std::endl; });
				reTime_02--;
			}
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