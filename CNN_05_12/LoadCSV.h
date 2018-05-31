#pragma once

#include <iostream>  
#include <string>  
#include <vector>  
#include <fstream>  
#include <sstream>  
//#include "MatrixWAndB.h"
#include "Test.h"

namespace tinyDNN
{
	class LoadCSV
	{
	public:
		LoadCSV();
		~LoadCSV();

		static void loadCSVTrain();
		static void loadCSVTest();

	};
}
