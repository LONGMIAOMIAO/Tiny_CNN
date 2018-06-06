#pragma once

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include "LayerQL.h"
#include "Fullconnect_LayerQL.h"
#include "Bias_LayerQL.h"
#include "Inter_LayerQL.h"
#include "Sigmoid_LayerQL.h"
#include "MSE_Loss_LayerQL.h"
#include "LoadCSV.h"
#include <winsock.h>
#include "NetQL.h"

namespace tinyDNN
{
	class Test
	{
	public:
		Test();
		~Test();

		void Fullconnect_Layer_Forward_Test();
		
		void Fullconnect_Layer_Backward_Test();

		void Fullconnect_Layer_Update_Test();

		void Fullconnect_Layer_Update_Batch_Test();

		
		void Bias_Layer_Forward_Test();

		void Bias_Layer_Backward_Test();

		void Bias_Layer_Update_Test();

		void Bias_Layer_Update_Batch_Test();


		void Sigmoid_LayerQL_Forward_Test();

		void Sigmoid_LayerQL_Backward_Test();

		void MSE_Loss_LayerQL_Backward_Test();

		void Operator_Test();

		void Mnist_Test();

		void Mnist_Test_02();

	public:
		static std::shared_ptr<Inter_LayerQL<double>> input_Layer;
		static std::shared_ptr<Inter_LayerQL<double>> output_Layer;

		static std::shared_ptr<Inter_LayerQL<double>> input_Layer_T;
		static std::shared_ptr<Inter_LayerQL<double>> output_Layer_T;
	};
}
