#pragma once
#include "Relu_LayerQL.h"
#include "NetQL.h"

namespace tinyDNN
{

	class Relu_LayerQL_Test
	{
	public:
		Relu_LayerQL_Test()
		{
			this->reluTest_Matrix();
			//this->reluTest_Vector();
		}
		~Relu_LayerQL_Test() {}

		void reluTest_Matrix()
		{
			std::shared_ptr<MatrixQL<double>> v_01 = std::make_shared<MatrixQL<double>>(3, 3);
			std::shared_ptr<MatrixQL<double>> v_02 = std::make_shared<MatrixQL<double>>(3, 3);

			int numStart = -4;
			for ( int i = 0; i < 3; i++ )
			{
				for ( int j = 0; j < 3; j++ )
				{
					v_01->setMatrixQL()(i, j) = numStart;
					v_02->setMatrixQL()(i, j) = numStart;

					numStart++;
				}
			}

			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(3, 3);

			in_01->forward_Matrix = v_01;

			in_01->forward_Matrix_Vector.push_back(v_01);
			in_01->forward_Matrix_Vector.push_back(v_02);

			std::shared_ptr<LayerQL<double>> relu_LayerQL_Test = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);
			relu_LayerQL_Test->pRelu_k = 0.1;
			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + relu_LayerQL_Test;

			//向前传播测试
			int s = 3;
			while (s > 0)
			{
				relu_LayerQL_Test->calForward();

				std::cout << o_01->forward_Matrix->getMatrixQL() << std::endl;
				s--;
			}

			o_01->backward_Matrix = std::make_shared<MatrixQL<double>>(3,3);
			o_01->backward_Matrix->setMatrixQL().setConstant(2.0);
			//反向传播测试
			int ss = 3;
			while (ss > 0)
			{
				relu_LayerQL_Test->calBackward();

				std::cout << in_01->backward_Matrix->getMatrixQL() << std::endl;
				ss--;
			}
		}

		void reluTest_Vector()
		{
			std::shared_ptr<MatrixQL<double>> v_01 = std::make_shared<MatrixQL<double>>(3, 3);
			std::shared_ptr<MatrixQL<double>> v_02 = std::make_shared<MatrixQL<double>>(3, 3);

			int numStart = -4;
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					v_01->setMatrixQL()(i, j) = numStart;
					v_02->setMatrixQL()(i, j) = numStart;

					numStart++;
				}
			}

			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(3, 3);

			in_01->forward_Matrix = v_01;

			in_01->forward_Matrix_Vector.push_back(v_01);
			in_01->forward_Matrix_Vector.push_back(v_02);

			std::shared_ptr<LayerQL<double>> relu_LayerQL_Test = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);
			relu_LayerQL_Test->pRelu_k = 0.1;
			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + relu_LayerQL_Test;

			//向前传播测试
			int s = 3;
			while (s > 0)
			{
				relu_LayerQL_Test->calForward();
	
				std::for_each(o_01->forward_Matrix_Vector.begin(), o_01->forward_Matrix_Vector.end(), []( std::shared_ptr<MatrixQL<double>> mat ) 
				{ std::cout << mat->getMatrixQL() << std::endl; });

				s--;
			}

			//o_01->backward_Matrix = std::make_shared<MatrixQL<double>>(3, 3);
			//o_01->backward_Matrix->setMatrixQL().setOnes();
			std::shared_ptr<MatrixQL<double>> backMatrix = std::make_shared<MatrixQL<double>>(3,3);
			std::shared_ptr<MatrixQL<double>> backMatrix_02 = std::make_shared<MatrixQL<double>>(3, 3);
			o_01->backward_Matrix_Vector.push_back(backMatrix);
			o_01->backward_Matrix_Vector.push_back(backMatrix_02);

			std::for_each(o_01->backward_Matrix_Vector.begin(), o_01->backward_Matrix_Vector.end(), [](std::shared_ptr<MatrixQL<double>> mat)
			{  mat->setMatrixQL().setConstant(2.0); });

			//反向传播测试
			int ss = 3;
			while (ss > 0)
			{
				relu_LayerQL_Test->calBackward();

				std::for_each(in_01->backward_Matrix_Vector.begin(), in_01->backward_Matrix_Vector.end(), [](std::shared_ptr<MatrixQL<double>> mat)
				{ std::cout << mat->getMatrixQL() << std::endl; });

				ss--;
			}
		}
	};
}