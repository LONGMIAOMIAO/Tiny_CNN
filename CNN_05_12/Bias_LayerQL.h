#pragma once
#include "LayerQL.h"
namespace tinyDNN
{
	template <typename Dtype> 
	class Bias_LayerQL : public LayerQL<Dtype>
	{
	public:
		explicit Bias_LayerQL( LayerType type, int rowNum, int colNum);
		~Bias_LayerQL() override final;

		void calForward(std::unique_ptr<MatrixQL<Dtype>>& feed_Left, std::unique_ptr<MatrixQL<Dtype>>& feed_Right) const override final;
		void calBackward(std::unique_ptr<MatrixQL<Dtype>>& loss_Right, std::unique_ptr<MatrixQL<Dtype>>& loss_Left)   override final;

		//std::unique_ptr<LayerQL<Dtype>> operator+(const std::unique_ptr<LayerQL<Dtype>>& operRight) const override final;

	protected:
		std::unique_ptr<MatrixQL<Dtype>> b_MatrixQL;
	};
	//*******************************************************************************************************************************
	template <typename Dtype>
	Bias_LayerQL<Dtype>::Bias_LayerQL(LayerType type, int rowNum, int colNum) : LayerQL(type)
	{
		std::cout << "Bias_LayerQL Start!" << std::endl;
		this->b_MatrixQL = std::make_unique<MatrixQL<Dtype>>( rowNum, colNum);
		//这里后期需要改
		this->b_MatrixQL->setMatrixQL().setConstant(5.23);
	}

	template <typename Dtype>
	Bias_LayerQL<Dtype>::~Bias_LayerQL()
	{
		std::cout << "Bias_LayerQL Over!" << std::endl;
	}

	template <typename Dtype>
	void Bias_LayerQL<Dtype>::calForward(std::unique_ptr<MatrixQL<Dtype>>& feed_Left, std::unique_ptr<MatrixQL<Dtype>>& feed_Right) const
	{
		feed_Right->setMatrixQL() = feed_Left->getMatrixQL() + this->b_MatrixQL->getMatrixQL();
	}

	template <typename Dtype>
	void Bias_LayerQL<Dtype>::calBackward(std::unique_ptr<MatrixQL<Dtype>>& loss_Right, std::unique_ptr<MatrixQL<Dtype>>& loss_Left)
	{
		loss_Left->setMatrixQL() = loss_Right->getMatrixQL();
	}

	//template <typename Dtype>
	//std::unique_ptr<LayerQL<Dtype>> Bias_LayerQL<Dtype>::operator+(const std::unique_ptr<LayerQL<Dtype>>& operRight) const
	//{
	//	//std::unique_ptr<LayerQL<Dtype>> tt;
	//	//return tt;
	//	return NULL;
	//};

}