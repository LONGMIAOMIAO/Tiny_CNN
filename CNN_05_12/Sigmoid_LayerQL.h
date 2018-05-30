#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class Sigmoid_LayerQL : public LayerQL<Dtype>
	{
	public:
		explicit Sigmoid_LayerQL(LayerType type);
		~Sigmoid_LayerQL();

		void calForward() const override final;
		void calBackward() override final;
	};

	template <typename Dtype>
	Sigmoid_LayerQL<Dtype>::Sigmoid_LayerQL(LayerType type) : LayerQL(type)
	{
		std::cout << "Sigmoid_LayerQL Start!" << std::endl;

	}

	template <typename Dtype>
	Sigmoid_LayerQL<Dtype>::~Sigmoid_LayerQL()
	{
		std::cout << "Sigmoid_LayerQL Over!" << std::endl;
	}

	template <typename Dtype>
	void Sigmoid_LayerQL<Dtype>::calForward() const
	{
		this->right_Layer->forward_Matrix->setMatrixQL() = 1.0 / ((-1.0 * (this->left_Layer->forward_Matrix->getMatrixQL()) ).array().exp() + 1.0);
	}
	template <typename Dtype>
	void Sigmoid_LayerQL<Dtype>::calBackward()
	{
		this->left_Layer->backward_Matrix->setMatrixQL() = this->right_Layer->backward_Matrix->getMatrixQL().array() *( (this->right_Layer->forward_Matrix->getMatrixQL()).array() * (1 - (this->right_Layer->forward_Matrix->getMatrixQL()).array()) );

	}

}