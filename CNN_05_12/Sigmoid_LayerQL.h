#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class Sigmoid_LayerQL : public LayerQL<Dtype>
	{
	public:
		Sigmoid_LayerQL(LayerType type);
		~Sigmoid_LayerQL() override final;

		void calForward(int type = 0) const override final;
		void calForward_Matrix() const;
		void calForward_Matrix_Vector() const;

		void calBackward(int type = 0) override final;
		void calBackward_Matrix();
		void calBackward_Matrix_Vector();

		void upMatrix() override final {};
		void upMatrix_batch(Dtype upRate) override final {};
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
	void Sigmoid_LayerQL<Dtype>::calForward(int type = 0) const
	{
		switch (this->layerType)
		{
		case Sigmoid_Layer:
			this->calForward_Matrix();
			break;
		case Sigmoid_Conv_Layer:
			this->calForward_Matrix_Vector();
			break;
		}
	}

	template <typename Dtype>
	void Sigmoid_LayerQL<Dtype>::calForward_Matrix() const
	{
		//前向传播
		this->right_Layer->forward_Matrix->setMatrixQL() = 1.0 / ((-1.0 * (this->left_Layer->forward_Matrix->getMatrixQL())).array().exp() + 1.0);
	}

	template <typename Dtype>
	void Sigmoid_LayerQL<Dtype>::calForward_Matrix_Vector() const
	{
		int vectorNum = this->left_Layer->forward_Matrix_Vector.size();

		this->right_Layer->forward_Matrix_Vector.clear();

		for ( int i = 0; i < vectorNum; i++ )
		{
			std::shared_ptr<MatrixQL<Dtype>> r_Matrix = std::make_shared<MatrixQL<Dtype>>(0,0);

			r_Matrix->setMatrixQL() = 1.0 / ((-1.0 * (this->left_Layer->forward_Matrix_Vector[i]->getMatrixQL())).array().exp() + 1.0);
			this->right_Layer->forward_Matrix_Vector.push_back(r_Matrix);
		}
	}

	template <typename Dtype>
	void Sigmoid_LayerQL<Dtype>::calBackward(int type = 0)
	{
		switch (this->layerType)
		{
		case Sigmoid_Layer:
			this->calBackward_Matrix();
			break;
		case Sigmoid_Conv_Layer:
			this->calBackward_Matrix_Vector();
			break;
		}
	}

	template <typename Dtype>
	void Sigmoid_LayerQL<Dtype>::calBackward_Matrix()
	{
		//反向传播，需要前向传播的参数
		this->left_Layer->backward_Matrix->setMatrixQL() = this->right_Layer->backward_Matrix->getMatrixQL().array() *((this->right_Layer->forward_Matrix->getMatrixQL()).array() * (1 - (this->right_Layer->forward_Matrix->getMatrixQL()).array()));
	}

	template <typename Dtype>
	void Sigmoid_LayerQL<Dtype>::calBackward_Matrix_Vector()
	{
		int vectorNum = this->right_Layer->backward_Matrix_Vector.size();

		this->left_Layer->backward_Matrix_Vector.clear();

		for ( int i = 0; i < vectorNum; i++ )
		{
			std::shared_ptr<MatrixQL<Dtype>> l_Matrix = std::make_shared<MatrixQL<Dtype>>(0,0);

			l_Matrix->setMatrixQL() = this->right_Layer->backward_Matrix_Vector[i]->getMatrixQL().array() *((this->right_Layer->forward_Matrix_Vector[i]->getMatrixQL()).array() * (1 - (this->right_Layer->forward_Matrix_Vector[i]->getMatrixQL()).array()));
			this->left_Layer->backward_Matrix_Vector.push_back(l_Matrix);
		}
	}
}