#pragma once
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include "MatrixQL.h"
#include "LoadCSV.h"
#include "LayerQL.h"
#include "Inter_LayerQL.h"
#include "Conv_LayerQL.h"
#include "Padding_LayerQL.h"

namespace tinyDNN
{
	class Conv_Test
	{
	public:
		Conv_Test()
		{
			//this->conv_One_Kernel_Test();
			this->conv_Kernel_Test();
		}
		~Conv_Test(){}

		void conv_One_Kernel_Test()
		{
			//	配置类型输入层， 10 * 10 ， 然后是3层 ， 将 3 层置入 Vector
			std::vector<std::shared_ptr<MatrixQL<double>>> inMatrixVector;
			std::shared_ptr<MatrixQL<double>> test_01 = std::make_shared<MatrixQL<double>>(10,10);
			test_01->setMatrixQL().setOnes();
			//test_01->setMatrixQL().setRandom();
			std::shared_ptr<MatrixQL<double>> test_02 = std::make_shared<MatrixQL<double>>(10,10);
			test_02->setMatrixQL().setOnes();
			//test_02->setMatrixQL().setRandom();
			std::shared_ptr<MatrixQL<double>> test_03 = std::make_shared<MatrixQL<double>>(10,10);
			test_03->setMatrixQL().setOnes();
			//test_03->setMatrixQL().setRandom();

			inMatrixVector.push_back(test_01);
			inMatrixVector.push_back(test_02);
			inMatrixVector.push_back(test_03);

			//	配置输出层，	6*6	， 一层
			std::shared_ptr<MatrixQL<double>> outMatrix = std::make_shared<MatrixQL<double>>(10,10);
			outMatrix->setMatrixQL().setZero();

			//	设置卷积核， 将卷积核
			Conv_Kernel<double> conv_Test(10,10,5,3,2);

			conv_Test.conv_CalForward( inMatrixVector, outMatrix );

			//***********************************************************************************//
			std::cout << outMatrix->getMatrixQL() << std::endl;
			
		}

		void conv_Kernel_Test()
		{
			std::shared_ptr<Inter_LayerQL<double>> inMatrixVector = std::make_shared<Inter_LayerQL<double>>( 3, 3 );

			std::shared_ptr<MatrixQL<double>> test_01 = std::make_shared<MatrixQL<double>>(3, 3);
			test_01->setMatrixQL().setConstant(0.1);
			//test_01->setMatrixQL().setRandom();

			std::shared_ptr<MatrixQL<double>> test_02 = std::make_shared<MatrixQL<double>>(3, 3);
			test_02->setMatrixQL().setConstant(0.2);
			//test_02->setMatrixQL().setRandom();

			//std::shared_ptr<MatrixQL<double>> test_03 = std::make_shared<MatrixQL<double>>(3, 3);
			//test_03->setMatrixQL().setConstant(0.3);
			////test_03->setMatrixQL().setRandom();

			inMatrixVector->forward_Matrix_Vector.push_back( test_01 );
			inMatrixVector->forward_Matrix_Vector.push_back( test_02 );
			//inMatrixVector->forward_Matrix_Vector.push_back( test_03 );


			//std::shared_ptr<LayerQL<double>> paddingMatrix = std::make_shared<Padding_LayerQL<double>>(Padding_Layer,10,10,2);

			//std::shared_ptr<Inter_LayerQL<double>> outMatrix_01 = inMatrixVector + paddingMatrix;

			std::shared_ptr<LayerQL<double>> conv_Kerel_Test = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 2, 3, 3, 3, 2, 1);

			std::shared_ptr<Inter_LayerQL<double>> outMatrix = inMatrixVector + conv_Kerel_Test;

			//paddingMatrix->calForward();

			conv_Kerel_Test->calForward();

			//********************************************************************************************************//
			//测试前向传播
			int stepNum = 3;

			while ( stepNum >0 )
			{
				std::cout << "前向传播**************************************************************" << std::endl;
				conv_Kerel_Test->calForward();
				for (auto i = outMatrix->forward_Matrix_Vector.begin(); i != outMatrix->forward_Matrix_Vector.end(); i++)
				{
					std::cout << (*i)->getMatrixQL() << std::endl;

				}
				stepNum--;
			}

			//*******************************************************************************************************
			//测试反向传播
			std::shared_ptr<MatrixQL<double>> test_03 = std::make_shared<MatrixQL<double>>(3, 3);
			test_03->setMatrixQL().setConstant(0.1);

			std::shared_ptr<MatrixQL<double>> test_04 = std::make_shared<MatrixQL<double>>(3, 3);
			test_04->setMatrixQL().setConstant(0.2);


			outMatrix->backward_Matrix_Vector.push_back(test_03);
			outMatrix->backward_Matrix_Vector.push_back(test_04);

			conv_Kerel_Test->calBackward();

			int stepNum_02 = 3;

			while (stepNum_02 > 0)
			{
				std::cout << "反向传播*****************************************************************************" << std::endl;
				for (auto i = inMatrixVector->backward_Matrix_Vector.begin(); i != inMatrixVector->backward_Matrix_Vector.end(); i++)
				{
					std::cout << (*i)->getMatrixQL() << std::endl;
				}
				stepNum_02--;
			}
			
			//*******************************************************************************************************
			//测试权重更新

			int stepNum_03 = 3;

			while (stepNum_03 > 0)
			{
				conv_Kerel_Test->upMatrix();
				stepNum_03--;
			}
		}
	};
}