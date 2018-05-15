#pragma once
#include "LayerQL.h"
namespace tinyDNN
{
	template <typename Dtype> 
	class Bias_LayerQL : public LayerQL<Dtype>
	{
	public:
		explicit Bias_LayerQL( LayerType type );
		~Bias_LayerQL() final;

		void calForward(std::unique_ptr<MatrixQL<Dtype>>& matrixIn, std::unique_ptr<MatrixQL<Dtype>>& matrixOut) const override final;
		void calBackward(std::unique_ptr<MatrixQL<Dtype>>& matrixIn, std::unique_ptr<MatrixQL<Dtype>>& matrixOut) final;

	};
	//*******************************************************************************************************************************
	template <typename Dtype>
	Bias_LayerQL<Dtype>::Bias_LayerQL(LayerType type) : LayerQL(type)
	{
		std::cout << "Bias_LayerQL Start!" << std::endl;
	}

	template <typename Dtype>
	Bias_LayerQL<Dtype>::~Bias_LayerQL()
	{
		std::cout << "Bias_LayerQL Over!" << std::endl;
	}

	template <typename Dtype>
	void Bias_LayerQL<Dtype>::calForward(std::unique_ptr<MatrixQL<Dtype>>& matrixLeft, std::unique_ptr<MatrixQL<Dtype>>& matrixRight) const
	{

	}

	template <typename Dtype>
	void Bias_LayerQL<Dtype>::calBackward(std::unique_ptr<MatrixQL<Dtype>>& matrixRight, std::unique_ptr<MatrixQL<Dtype>>& matrixLeft)
	{

	}

}