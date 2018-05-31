#include "LoadCSV.h"

namespace tinyDNN
{
	LoadCSV::LoadCSV()
	{
	}

	LoadCSV::~LoadCSV()
	{
	}

	void LoadCSV::loadCSVTrain()
	{
		//MatrixWAndB::maTrixTrainToal = Eigen::MatrixXd(55000, 784);
		//MatrixWAndB::maTrixTrainToalL = Eigen::MatrixXi(55000, 10);
		//MatrixWAndB::maTrixTrainToalL.setZero();

		//Test::input_Layer->forward_Matrix->setMatrixQL();
		//Test::output_Layer->backward_Matrix->setMatrixQL();

		// 读文件  
		std::ifstream inFile("H:/CNN_0510/DATA/MNISTDATA_CSV/train.csv", std::ios::in);
		std::string lineStr;
		int lineNum = 0;
		while (std::getline(inFile, lineStr))
		{
			//// 存成二维表结构  
			std::stringstream ss(lineStr);
			std::string str;
			//// 按照逗号分隔  
			int inNum = 0;
			while (std::getline(ss, str, ','))
			{
				//MatrixWAndB::maTrixTrainToal(lineNum, inNum) = atoi(str.c_str());
				Test::input_Layer->forward_Matrix->setMatrixQL()(lineNum, inNum) = atoi(str.c_str());
				inNum++;
			}

			lineNum++;
		}
		//getchar();

		// 读文件  
		std::ifstream inFile_L("H:/CNN_0510/DATA/MNISTDATA_CSV/trainL.csv", std::ios::in);
		std::string lineStr_L;
		int lineNum_L = 0;
		while (std::getline(inFile_L, lineStr_L))
		{
			//// 存成二维表结构  
			std::stringstream ss(lineStr_L);
			std::string str;
			//// 按照逗号分隔  
			int inNum = 0;
			while (std::getline(ss, str, ','))
			{
				//MatrixWAndB::maTrixTrainToalL(lineNum_L, inNum) = atoi(str.c_str());
				Test::output_Layer->backward_Matrix->setMatrixQL()(lineNum_L, inNum) = atoi(str.c_str());
				inNum++;
			}
			lineNum_L++;

		}

		Test::input_Layer->forward_Matrix->setMatrixQL() = Test::input_Layer->forward_Matrix->getMatrixQL() / 255;


		//std::cout << MatrixWAndB::maTrixTrainToal.row(0) << std::endl;
		//std::cout << MatrixWAndB::maTrixTrainToalL.row(0) << std::endl;
	}


	void LoadCSV::loadCSVTest()
	{
		//MatrixWAndB::maTrixTestToal = Eigen::MatrixXd(55000, 784);

		//MatrixWAndB::maTrixTestToalL = Eigen::MatrixXi(55000, 10);
		//MatrixWAndB::maTrixTestToalL.setZero();

		Test::input_Layer_T->forward_Matrix->setMatrixQL();
		Test::output_Layer_T->backward_Matrix->setMatrixQL();

		// 读文件  
		std::ifstream inFile("H:/CNN_0510/DATA/MNISTDATA_CSV/test.csv", std::ios::in);
		std::string lineStr;
		int lineNum = 0;
		while (std::getline(inFile, lineStr))
		{
			//// 存成二维表结构  
			std::stringstream ss(lineStr);
			std::string str;
			//// 按照逗号分隔  
			int inNum = 0;
			while (std::getline(ss, str, ','))
			{
				Test::input_Layer_T->forward_Matrix->setMatrixQL()(lineNum, inNum) = atoi(str.c_str());
				inNum++;
			}

			lineNum++;
		}
		//getchar();

		// 读文件  
		std::ifstream inFile_L("H:/CNN_0510/DATA/MNISTDATA_CSV/testL.csv", std::ios::in);
		std::string lineStr_L;
		int lineNum_L = 0;
		while (std::getline(inFile_L, lineStr_L))
		{
			//// 存成二维表结构  
			std::stringstream ss(lineStr_L);
			std::string str;
			//// 按照逗号分隔  
			int inNum = 0;
			while (std::getline(ss, str, ','))
			{
				Test::output_Layer_T->backward_Matrix->setMatrixQL()(lineNum_L, inNum) = atoi(str.c_str());
				inNum++;
			}
			lineNum_L++;

		}

		Test::input_Layer_T->forward_Matrix->setMatrixQL() = Test::input_Layer_T->forward_Matrix->getMatrixQL() / 255;


		//std::cout << MatrixWAndB::maTrixTrainToal.row(0) << std::endl;
		//std::cout << MatrixWAndB::maTrixTrainToalL.row(0) << std::endl;
	}
}