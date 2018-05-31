#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class MSE_Loss_LayerQL : public LayerQL<Dtype>
	{
	public:
		explicit MSE_Loss_LayerQL(LayerType type);
		~MSE_Loss_LayerQL();

		void calForward() const override final;
		void calBackward() override final;
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
	void MSE_Loss_LayerQL<Dtype>::calForward() const
	{

	}

	template <typename Dtype>
	void MSE_Loss_LayerQL<Dtype>::calBackward()
	{
		this->left_Layer->backward_Matrix->setMatrixQL() = this->left_Layer->forward_Matrix->getMatrixQL() - this->right_Layer->backward_Matrix->getMatrixQL();

	}


}