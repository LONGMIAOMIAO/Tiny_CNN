#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class MSE_Loss_LayerQL : public LayerQL<Dtype>
	{
	public:
		MSE_Loss_LayerQL(LayerType type);
		~MSE_Loss_LayerQL() override final;

		void calForward(int type = 0) const override final;
		void calBackward(int type = 0) override final;

		void upMatrix() override final {};

		void upMatrix_batch(Dtype upRate) override final {};
	};

	template <typename Dtype>
	MSE_Loss_LayerQL<Dtype>::MSE_Loss_LayerQL(LayerType type) : LayerQL(type)
	{
		std::cout << "MSE_Loss_LayerQL Start!" << std::endl;
	}

	template <typename Dtype>
	MSE_Loss_LayerQL<Dtype>::~MSE_Loss_LayerQL()
	{
		std::cout << "MSE_Loss_LayerQL Over!" << std::endl;
	}

	template <typename Dtype>
	void MSE_Loss_LayerQL<Dtype>::calForward(int type = 0) const{}

	template <typename Dtype>
	void MSE_Loss_LayerQL<Dtype>::calBackward(int type = 0)
	{
		//std::cout << this->left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		//std::cout << this->right_Layer->backward_Matrix->getMatrixQL() << std::endl;

		//反向传播，用输入值-lable值
		this->left_Layer->backward_Matrix->setMatrixQL() = this->left_Layer->forward_Matrix->getMatrixQL() - this->right_Layer->backward_Matrix->getMatrixQL();
	}
}