#pragma once

#include <iostream>  
#include <string>  
#include <vector>  
#include <fstream>  
#include <sstream>  
#include "Test.h"

namespace tinyDNN
{
	class LoadCSV
	{
	public:
		LoadCSV();
		~LoadCSV();
		//装载训练集
		static void loadCSVTrain();
		//装载测试集
		static void loadCSVTest();
	};
}
