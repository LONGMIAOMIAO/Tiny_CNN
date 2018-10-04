#pragma once
#include <vector>
//#include <memory>
#include "Bias_Conv_Layer.h"
#include "NetQL.h"

namespace tinyDNN
{
	class Bias_Conv_Layer_Test
	{
	public:
		Bias_Conv_Layer_Test()
		{
			this->bias_Conv_Forward();
		}

		~Bias_Conv_Layer_Test() {}

		void bias_Conv_Forward()
		{
			std::vector<std::shared_ptr<MatrixQL<double>>> vec_Ptr{ std::make_shared<MatrixQL<double>>(5,5),std::make_shared<MatrixQL<double>>(5,5),std::make_shared<MatrixQL<double>>(5,5) };

			vec_Ptr[0]->setMatrixQL().setConstant(3.3);
			vec_Ptr[1]->setMatrixQL().setConstant(6.3);
			vec_Ptr[2]->setMatrixQL().setConstant(1.3);

			std::shared_ptr<Inter_LayerQL<double>> in_1 = std::make_shared<Inter_LayerQL<double>>(5, 5);
			in_1->forward_Matrix_Vector = vec_Ptr;

			std::shared_ptr<LayerQL<double>> bias_Conv = std::make_shared<Bias_Conv_Layer<double>>( Bias_Conv_Layer_L, 3, 5, 5);
			std::shared_ptr<Inter_LayerQL<double>> out_1 = in_1 + bias_Conv;

			//	前传
			bias_Conv->calForward();
			for (int i = 0; i < 3; i++)
			{
				std::cout << out_1->forward_Matrix_Vector[i]->getMatrixQL() << std::endl;
			}

			//	反传
			out_1->backward_Matrix_Vector.push_back( std::make_shared<MatrixQL<double>>(5,5) );
			out_1->backward_Matrix_Vector.push_back(std::make_shared<MatrixQL<double>>(5, 5));
			out_1->backward_Matrix_Vector.push_back(std::make_shared<MatrixQL<double>>(5, 5));

			out_1->backward_Matrix_Vector[0]->setMatrixQL().setConstant(0.1);
			out_1->backward_Matrix_Vector[1]->setMatrixQL().setConstant(0.1);
			out_1->backward_Matrix_Vector[2]->setMatrixQL().setConstant(0.1);

			bias_Conv->calBackward();
			for (int i = 0; i < 3; i++)
			{
				std::cout << out_1->backward_Matrix_Vector[i]->getMatrixQL() << std::endl;
			}

			//	更新权重
			bias_Conv->upMatrix();

			for (int i = 0; i < 3; i++)
			{
				auto s = std::dynamic_pointer_cast<Bias_Conv_Layer<double>>(bias_Conv);
				std::cout << s->b_MatrixQL[i]->getMatrixQL() << std::endl;
			}
		}
	};
}