#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class SoftMax_LayerQL : public LayerQL<Dtype>
	{
	public:
		explicit SoftMax_LayerQL(LayerType type) : LayerQL(type)
		{
			std::cout << "SoftMax_LayerQL Start!" << std::endl;
		}
		~SoftMax_LayerQL() 
		{
			std::cout << "SoftMax_LayerQL Over!" << std::endl;
		}

		void calForward(int type = 0) const override final
		{
			//this->outData = inData.array().exp().matrix() / (inData.array().exp()).sum();
			this->right_Layer->forward_Matrix->setMatrixQL() = this->left_Layer->forward_Matrix->getMatrixQL().array().exp().matrix() / (this->left_Layer->forward_Matrix->getMatrixQL().array().exp()).sum();
		}

		void calBackward(int type = 0) override final
		{
			this->left_Layer->backward_Matrix->setMatrixQL() = this->right_Layer->forward_Matrix->getMatrixQL() - this->right_Layer->backward_Matrix->getMatrixQL();
		}

		void upMatrix() override final {};
		void upMatrix_batch(Dtype upRate) override final {};
	};

}