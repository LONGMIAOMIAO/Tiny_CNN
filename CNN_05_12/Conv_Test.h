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
			this->conv_One_Kernel_Test();
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
			std::shared_ptr<Inter_LayerQL<double>> inMatrixVector = std::make_shared<Inter_LayerQL<double>>( 10, 10 );

			std::shared_ptr<MatrixQL<double>> test_01 = std::make_shared<MatrixQL<double>>(10, 10);
			test_01->setMatrixQL().setOnes();
			//test_01->setMatrixQL().setRandom();

			std::shared_ptr<MatrixQL<double>> test_02 = std::make_shared<MatrixQL<double>>(10, 10);
			test_02->setMatrixQL().setConstant(2);
			//test_02->setMatrixQL().setRandom();

			std::shared_ptr<MatrixQL<double>> test_03 = std::make_shared<MatrixQL<double>>(10, 10);
			test_03->setMatrixQL().setConstant(3);
			//test_03->setMatrixQL().setRandom();

			inMatrixVector->forward_Matrix_Vector.push_back( test_01 );
			inMatrixVector->forward_Matrix_Vector.push_back( test_02 );
			inMatrixVector->forward_Matrix_Vector.push_back( test_03 );


			//std::shared_ptr<LayerQL<double>> paddingMatrix = std::make_shared<Padding_LayerQL<double>>(Padding_Layer,10,10,2);

			//std::shared_ptr<Inter_LayerQL<double>> outMatrix_01 = inMatrixVector + paddingMatrix;

			std::shared_ptr<LayerQL<double>> conv_Kerel_Test = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 5, 10, 10, 5, 3, 2);

			std::shared_ptr<Inter_LayerQL<double>> outMatrix = inMatrixVector + conv_Kerel_Test;

			//paddingMatrix->calForward();

			conv_Kerel_Test->calForward();

			//********************************************************************************************************//

			for ( auto i = outMatrix->forward_Matrix_Vector.begin(); i != outMatrix->forward_Matrix_Vector.end(); i++ )
			{
				std::cout << (*i)->getMatrixQL() << std::endl;
				std::cout << "*****************************************************************************" << std::endl;
			}

			//*******************************************************************************************************
			//测试反向传播
			outMatrix->backward_Matrix_Vector = outMatrix->forward_Matrix_Vector;
			conv_Kerel_Test->calBackward();

			for ( auto i = inMatrixVector->backward_Matrix_Vector.begin(); i != inMatrixVector->backward_Matrix_Vector.end(); i++ )
			{
				std::cout << (*i)->getMatrixQL() << std::endl;
				std::cout << "*****************************************************************************" << std::endl;
			}

			conv_Kerel_Test->upMatrix();
		}
	};
}