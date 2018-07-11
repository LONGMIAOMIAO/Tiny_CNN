#pragma once
#include "Data_AugmentationQL.h"
#include "NetQL.h"

namespace tinyDNN
{
	class Data_Augmentation_Test
	{
	public:
		Data_Augmentation_Test()
		{
			this->data_Aug_Forward();
		}

		~Data_Augmentation_Test(){}

		void data_Aug_Forward()
		{
			std::shared_ptr<MatrixQL<double>> v_01 = std::make_shared<MatrixQL<double>>(5, 5);
			std::shared_ptr<MatrixQL<double>> v_02 = std::make_shared<MatrixQL<double>>(5, 5);
			std::shared_ptr<MatrixQL<double>> v_03 = std::make_shared<MatrixQL<double>>(5, 5);

			int coutNum = 0;
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					v_01->setMatrixQL()(i, j) = coutNum;
					v_02->setMatrixQL()(i, j) = coutNum;
					v_03->setMatrixQL()(i, j) = coutNum;

					coutNum++;
				}
			}

			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(5, 5);

			in_01->forward_Matrix_Vector.push_back(v_01);
			in_01->forward_Matrix_Vector.push_back(v_02);
			in_01->forward_Matrix_Vector.push_back(v_03);

			std::for_each(in_01->forward_Matrix_Vector.begin(), in_01->forward_Matrix_Vector.end(), [](std::shared_ptr<MatrixQL<double>> inMat) { std::cout << inMat->getMatrixQL() << std::endl;
			std::cout << "=======================" << std::endl; });

			std::shared_ptr<LayerQL<double>> dataTest = std::make_shared<Data_AugmentationQL<double>>(Data_Augmentation_Layer,0,0);

			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + dataTest;

			for (int i = 0; i < 10; i++)
			{
				dataTest->calForward();

				std::for_each(o_01->forward_Matrix_Vector.begin(), o_01->forward_Matrix_Vector.end(), [&](std::shared_ptr<MatrixQL<double>> inMat) { std::cout << inMat->getMatrixQL() << std::endl;
				std::cout << i << "=======================" << std::endl; });

			}
		}
	};
}