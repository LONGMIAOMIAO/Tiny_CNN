#pragma once
#include "PooLayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class Pool_Test
	{
	public:
		Pool_Test()
		{
			this->pool_Average_Vector();
		}

		void pool_Average_Vector()
		{
			//测试加载卷积二维图像,训练集
			LoadCSV::loadCSVTrain();
			LoadCSV::loadCSV_Train_Vector();




		}

		void pool_Test()
		{
				
		}

		~Pool_Test(){}
	};
}