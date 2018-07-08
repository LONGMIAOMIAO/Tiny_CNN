#pragma once
#include "SoftMax_LayerQL.h"
#include "NetQL.h"

namespace tinyDNN
{
	class SoftMax_Layer_Test
	{
	public:
		SoftMax_Layer_Test()
		{
			this->softForAndBakcWard();
		}
		~SoftMax_Layer_Test()
		{
		}

		void softForAndBakcWard()
		{
			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(1, 5);
			//std::shared_ptr<MatrixQL<double>> in_01 = std::make_shared<MatrixQL<double>>(1, 5);
			int s = -2;
			for ( int i = 0; i < 5; i++ )
			{
				in_01->forward_Matrix->setMatrixQL()(0, i) = s;
				s++;
			}
			std::shared_ptr<LayerQL<double>> soft_01 = std::make_shared<SoftMax_LayerQL<double>>(SoftMax_Layer);

			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + soft_01;

			soft_01->calForward();

			std::cout << o_01->forward_Matrix->getMatrixQL() << std::endl;

			o_01->backward_Matrix = std::make_shared<MatrixQL<double>>(1,5);
			o_01->backward_Matrix->setMatrixQL().setZero();
			o_01->backward_Matrix->setMatrixQL()(0, 0) = 1;

			soft_01->calBackward();

			std::cout << in_01->backward_Matrix->getMatrixQL() << std::endl;

			//			std::shared_ptr<LayerQL<double>> relu_LayerQL_Test = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);
		}
	};
}