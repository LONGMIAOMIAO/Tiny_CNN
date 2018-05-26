#pragma once
#include "LayerQL.h"
#include "Fullconnect_LayerQL.h"
#include "Bias_LayerQL.h"

namespace tinyDNN
{
	class Test
	{
	public:
		Test();
		~Test();

		void Fullconnect_Layer_Forward_Test();
		
		void Fullconnect_Layer_Backward_Test();
		
		void Bias_Layer_Forward_Test();

		void Bias_Layer_Backward_Test();
	};
}
