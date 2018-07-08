#pragma once
#include "Sigmoid_LayerQL.h"
#include "NetQL.h"
namespace tinyDNN
{
	class Sigmoid_Test
	{
	public:
		Sigmoid_Test()
		{
			this->Sigmoid_Vector_Forward_Backward();
		}
		~Sigmoid_Test(){}

		void Sigmoid_Vector_Forward_Backward()
		{
			std::shared_ptr<MatrixQL<double>> v_01 = std::make_shared<MatrixQL<double>>(5, 5);
			v_01->setMatrixQL().setZero();

			std::shared_ptr<MatrixQL<double>> v_02 = std::make_shared<MatrixQL<double>>(5, 5);
			v_02->setMatrixQL().setConstant(1.0);

			std::shared_ptr<MatrixQL<double>> v_03 = std::make_shared<MatrixQL<double>>(5, 5);
			v_03->setMatrixQL().setConstant(2.0);

			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(5, 5);

			in_01->forward_Matrix = v_01;

			in_01->forward_Matrix_Vector.push_back(v_01);
			in_01->forward_Matrix_Vector.push_back(v_02);
			in_01->forward_Matrix_Vector.push_back(v_03);

			std::shared_ptr<LayerQL<double>> sigmoid_Layer_Test = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);

			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + sigmoid_Layer_Test;

			//向前传播测试++++++++++++++++++++++++++++++++++++++
			int s = 3;
			while (s > 0)
			{
				sigmoid_Layer_Test->calForward();
				sigmoid_Layer_Test->calForward(1);

				std::cout << "---------------------------------------------------------" << std::endl;
				std::cout << o_01->forward_Matrix->getMatrixQL() << std::endl;
				std::cout << "+++++++++++++++" << std::endl;
				std::for_each(o_01->forward_Matrix_Vector.begin(), o_01->forward_Matrix_Vector.end(),
					[](std::shared_ptr<MatrixQL<double>> matrix_t)
				{
					std::cout << matrix_t->getMatrixQL() << std::endl;
					std::cout << "*************" << std::endl;
				}
				);
				s--;
			}

			//向后传播测试++++++++++++++++++++++++++++++++++++++
			o_01->backward_Matrix = in_01->forward_Matrix;
			o_01->backward_Matrix->setMatrixQL().setConstant(0.5);
			//****************
			o_01->backward_Matrix_Vector = in_01->forward_Matrix_Vector;
			o_01->backward_Matrix_Vector[0]->setMatrixQL().setConstant(0.5);
			o_01->backward_Matrix_Vector[1]->setMatrixQL().setConstant(0.5);
			o_01->backward_Matrix_Vector[2]->setMatrixQL().setConstant(0.5);

			int t = 3;
			while (t > 0)
			{
				sigmoid_Layer_Test->calBackward();
				sigmoid_Layer_Test->calBackward(1);
				std::cout << "---------------------------------------------------------" << std::endl;
				std::cout << in_01->backward_Matrix->getMatrixQL() << std::endl;
				std::cout << "+++++++++++++++" << std::endl;
				std::for_each(in_01->backward_Matrix_Vector.begin(), in_01->backward_Matrix_Vector.end(),
					[](std::shared_ptr<MatrixQL<double>> matrix_t)
				{
					std::cout << matrix_t->getMatrixQL() << std::endl;
					std::cout << "*************" << std::endl;
				}
				);
				t--;
			}
		}
	};
}