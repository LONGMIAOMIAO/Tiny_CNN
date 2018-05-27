#include "Test.h"

namespace tinyDNN
{
	Test::Test()
	{
		////Full_Layer ForwardTrans
		//this->Fullconnect_Layer_Forward_Test();
		////Full_Layer BackwardTrans
		//this->Fullconnect_Layer_Backward_Test();
		////Bias_Layer ForwardTrans
		//this->Bias_Layer_Forward_Test();
		////Bias_Layer BackwardTrans
		//this->Bias_Layer_Backward_Test();
		//Operator_Layer
		this->Operator_Test();
	}

	Test::~Test()
	{
	}
	
	void Test::Fullconnect_Layer_Forward_Test()
	{
		std::unique_ptr<MatrixQL<double>> feed_Left_01 = std::make_unique<MatrixQL<double>>(2, 3);
		feed_Left_01->setMatrixQL() << 1, 2, 3, 4, 5, 6;

		std::unique_ptr<LayerQL<double>> fullLayerTest = std::make_unique<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 3, 5);

		std::unique_ptr<MatrixQL<double>> feed_Right_01 = std::make_unique<MatrixQL<double>>(2, 5);

		fullLayerTest->calForward(feed_Left_01, feed_Right_01);

		std::cout << feed_Right_01->getMatrixQL() << std::endl;
	}

	void Test::Fullconnect_Layer_Backward_Test()
	{
		std::unique_ptr<MatrixQL<double>> loss_Right_01 = std::make_unique<MatrixQL<double>>(2,5);
		loss_Right_01->setMatrixQL() << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

		std::unique_ptr<LayerQL<double>> fullLayerTest = std::make_unique<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 3, 5);

		std::unique_ptr<MatrixQL<double>> loss_Left_01 = std::make_unique<MatrixQL<double>>(2,3);

		fullLayerTest->calBackward(loss_Right_01,loss_Left_01);

		std::cout << loss_Left_01->getMatrixQL() << std::endl;
	}

	void Test::Bias_Layer_Forward_Test()
	{
		std::unique_ptr<MatrixQL<double>> feed_Left_01 = std::make_unique<MatrixQL<double>>(2, 3);
		feed_Left_01->setMatrixQL() << 1, 2, 3, 4, 5, 6;

		std::unique_ptr<LayerQL<double>> bias_Layer = std::make_unique<Bias_LayerQL<double>>(Bias_Layer, 2, 3);

		std::unique_ptr<MatrixQL<double>> feed_Right_01 = std::make_unique<MatrixQL<double>>(2, 3);

		bias_Layer->calForward(feed_Left_01, feed_Right_01);

		std::cout << feed_Right_01->getMatrixQL() << std::endl;
	}

	void Test::Bias_Layer_Backward_Test()
	{
		std::unique_ptr<MatrixQL<double>> loss_Right_01 = std::make_unique<MatrixQL<double>>(2, 3);
		loss_Right_01->setMatrixQL() << 1, 2, 3, 4, 5, 6;

		std::unique_ptr<LayerQL<double>> bias_Layer = std::make_unique<Bias_LayerQL<double>>(Bias_Layer, 2, 3);

		std::unique_ptr<MatrixQL<double>> loss_Left_01 = std::make_unique<MatrixQL<double>>(2, 3);

		bias_Layer->calBackward(loss_Right_01, loss_Left_01);

		std::cout << loss_Left_01->getMatrixQL() << std::endl;
	}

	void Test::Operator_Test()
	{
		std::shared_ptr<LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(Inter_Layer);
		
		std::shared_ptr<LayerQL<double>> bias_Layer = std::make_shared<Bias_LayerQL<double>>(Bias_Layer,2,3);

		std::shared_ptr<LayerQL<double>> out_01 = left_Layer + bias_Layer;

		//std::cout << out_01->layerType << std::endl;

		std::cout << bias_Layer->left_Layer->layerType << std::endl;
		std::cout << bias_Layer->right_Layer->layerType << std::endl;

		std::shared_ptr<LayerQL<double>> bias_Layer_02 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 2, 3);

		std::shared_ptr<LayerQL<double>> out_02 = out_01 + bias_Layer_02;

		std::cout << bias_Layer_02->left_Layer->layerType << std::endl;
		std::cout << bias_Layer_02->right_Layer->layerType << std::endl;

		std::shared_ptr<LayerQL<double>> bias_Layer_03 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 2, 3);

		std::shared_ptr<LayerQL<double>> out_03 = out_02 + bias_Layer_03;

		std::cout << bias_Layer_03->left_Layer->layerType << std::endl;
		std::cout << bias_Layer_03->right_Layer->layerType << std::endl;


	}
}