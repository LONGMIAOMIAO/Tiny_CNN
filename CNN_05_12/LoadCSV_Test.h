#pragma once
#include "LoadCSV.h"
namespace tinyDNN
{
	class LoadCSV_Test
	{
	public:
		LoadCSV_Test()
		{
			//this->dim2_Matrix();
			this->cifar_10_train();
		}
		~LoadCSV_Test()
		{
		}

		void dim2_Matrix()
		{
			//²âÊÔ¼ÓÔØ¾í»ı¶şÎ¬Í¼Ïñ,ÑµÁ·¼¯
			LoadCSV::loadCSVTrain();
			LoadCSV::loadCSV_Train_Vector();

			std::cout << LoadCSV::conv_Input_Vector[2]->getMatrixQL() << std::endl;
			std::cout << ( LoadCSV::conv_Input_Vector[2]->getMatrixQL() * 9).cast<int>() << std::endl;


			LoadCSV::loadCSVTest();
			LoadCSV::loadCSV_Test_Vector();

			std::cout << LoadCSV::conv_Input_Vector_T[2]->getMatrixQL() << std::endl;
			std::cout << (LoadCSV::conv_Input_Vector_T[2]->getMatrixQL() * 9).cast<int>() << std::endl;
		}

		void cifar_10_train()
		{
			LoadCifar_10::loadCifar_10_Train();
		}
	};
}