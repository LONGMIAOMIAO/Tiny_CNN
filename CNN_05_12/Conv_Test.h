#pragma once
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include "MatrixQL.h"
#include "LoadCSV.h"
#include "LayerQL.h"
#include "Inter_LayerQL.h"
#include "Conv_LayerQL.h"

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
			//	������������㣬 10 * 10 �� Ȼ����3�� �� �� 3 ������ Vector
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

			//	��������㣬	6*6	�� һ��
			std::shared_ptr<MatrixQL<double>> outMatrix = std::make_shared<MatrixQL<double>>(6,6);
			outMatrix->setMatrixQL().setZero();

			//	���þ����ˣ� ��������
			Conv_Kernel<double> conv_Test(6,6,5,3);

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
			test_02->setMatrixQL().setOnes();
			//test_02->setMatrixQL().setRandom();

			std::shared_ptr<MatrixQL<double>> test_03 = std::make_shared<MatrixQL<double>>(10, 10);
			test_03->setMatrixQL().setOnes();
			//test_03->setMatrixQL().setRandom();

			inMatrixVector->forward_Matrix_Vector.push_back( test_01 );
			inMatrixVector->forward_Matrix_Vector.push_back( test_02 );
			inMatrixVector->forward_Matrix_Vector.push_back( test_03 );

			std::shared_ptr<LayerQL<double>> conv_Kerel_Test = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 5, 6, 6, 5, 3);

			std::shared_ptr<Inter_LayerQL<double>> outMatrix = inMatrixVector + conv_Kerel_Test;

			conv_Kerel_Test->calForward();

			//********************************************************************************************************//

			for ( auto i = outMatrix->forward_Matrix_Vector.begin(); i != outMatrix->forward_Matrix_Vector.end(); i++ )
			{
				std::cout << (*i)->getMatrixQL() << std::endl;
				std::cout << "*****************************************************************************" << std::endl;
			}
		}
	};
}