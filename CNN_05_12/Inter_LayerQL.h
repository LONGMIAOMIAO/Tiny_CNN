#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class Inter_LayerQL : public LayerQL<Dtype>
	{
	public:

		//template <typename Dtype> friend class Fullconnect_LayerQL;
		Inter_LayerQL(LayerType type/*, int rowNum, int rolNum*/);
		~Inter_LayerQL() override final;

		//std::unique_ptr<LayerQL<Dtype>> operator+(const std::unique_ptr<LayerQL<Dtype>>& operRight) const override final;
		void calForward(std::unique_ptr<MatrixQL<Dtype>>& feed_Left, std::unique_ptr<MatrixQL<Dtype>>& feed_Right) const override final {};
		void calBackward(std::unique_ptr<MatrixQL<Dtype>>& loss_Right, std::unique_ptr<MatrixQL<Dtype>>& loss_Left) override final {};



	protected:
		std::unique_ptr<MatrixQL<Dtype>> forward_Matrix;
		std::unique_ptr<MatrixQL<Dtype>> backward_Matrix;
	};


	template <typename Dtype>
	Inter_LayerQL<Dtype>::Inter_LayerQL(LayerType type/*, int rowNum, int rolNum*/) : LayerQL(type)
	{
		std::cout << "Inter_Layer Start!" << std::endl;

	}

	template <typename Dtype>
	Inter_LayerQL<Dtype>::~Inter_LayerQL()
	{
		std::cout << "Inter_Layer End!" << std::endl;
	}

	//template <typename Dtype>
	//std::unique_ptr<LayerQL<Dtype>> Inter_LayerQL<Dtype>::operator+(const std::unique_ptr<LayerQL<Dtype>>& operRight) const
	//{
	//	operRight->left_Layer = this;


	//	std::unique_ptr<LayerQL<Dtype>> tt = std::make_unique<LayerQL<Dtype>>(Inter_Layer);

	//	return tt;
	//};
}
