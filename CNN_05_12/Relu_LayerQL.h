#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class Relu_LayerQL : public LayerQL<Dtype>
	{
	public:
		Relu_LayerQL(LayerType type) : LayerQL(type)
		{
			std::cout << "Relu_LayerQL Start!" << std::endl;
		}
		~Relu_LayerQL()
		{
			std::cout << "Relu_LayerQL Start!" << std::endl;
		}
		//前传
		void calForward(int type = 0) const override final
		{
			switch (this->layerType)
			{
			case Relu_Layer:
				this->calForward_Matrix();
				break;
			case Relu_Conv_Layer:
				this->calForward_Matrix_Vector();
				break;
			}
		}
		void calForward_Matrix() const
		{
			this->right_Layer->forward_Matrix->setMatrixQL() = (this->left_Layer->forward_Matrix->getMatrixQL().array() < 0).select(this->pRelu_k*this->left_Layer->forward_Matrix->getMatrixQL(), this->left_Layer->forward_Matrix->getMatrixQL());
		}
		void calForward_Matrix_Vector() const
		{
			int vectorNum = this->left_Layer->forward_Matrix_Vector.size();

			this->right_Layer->forward_Matrix_Vector.clear();

			for (int i = 0; i < vectorNum; i++)
			{
				std::shared_ptr<MatrixQL<Dtype>> r_Matrix = std::make_shared<MatrixQL<Dtype>>(0, 0);

				r_Matrix->setMatrixQL() = (this->left_Layer->forward_Matrix_Vector[i]->getMatrixQL().array() < 0).select(this->pRelu_k*this->left_Layer->forward_Matrix_Vector[i]->getMatrixQL(), this->left_Layer->forward_Matrix_Vector[i]->getMatrixQL());

				this->right_Layer->forward_Matrix_Vector.push_back(r_Matrix);
			}
		}
		//反传
		void calBackward(int type = 0) override final
		{
			switch (this->layerType)
			{
			case Relu_Layer:
				this->calBackward_Matrix();
				break;
			case Relu_Conv_Layer:
				this->calBackward_Matrix_Vector();
				break;
			}
		}
		void calBackward_Matrix()
		{
			int rowNum = this->left_Layer->forward_Matrix->getMatrixQL().rows();
			int colNum = this->left_Layer->forward_Matrix->getMatrixQL().cols();

			std::shared_ptr<MatrixQL<Dtype>> oneK = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
			oneK->setMatrixQL().setOnes();
			//(inMatrix.array() < 0).select(pRelu_k, oneK);
			this->left_Layer->backward_Matrix->setMatrixQL() = this->right_Layer->backward_Matrix->getMatrixQL().array() * ((this->right_Layer->forward_Matrix->getMatrixQL().array() < 0).select(this->pRelu_k, oneK->getMatrixQL()).array());
		}
		void calBackward_Matrix_Vector()
		{
			int vectorNum = this->right_Layer->backward_Matrix_Vector.size();

			this->left_Layer->backward_Matrix_Vector.clear();

			for (int i = 0; i < vectorNum; i++)
			{
				std::shared_ptr<MatrixQL<Dtype>> l_Matrix = std::make_shared<MatrixQL<Dtype>>(0, 0);

				int rowNum = this->left_Layer->forward_Matrix_Vector[i]->getMatrixQL().rows();
				int colNum = this->left_Layer->forward_Matrix_Vector[i]->getMatrixQL().cols();

				std::shared_ptr<MatrixQL<Dtype>> oneK = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
				oneK->setMatrixQL().setOnes();

				l_Matrix->setMatrixQL() = this->right_Layer->backward_Matrix_Vector[i]->getMatrixQL().array() * ((this->right_Layer->forward_Matrix_Vector[i]->getMatrixQL().array() < 0).select(this->pRelu_k, oneK->getMatrixQL()).array());

				this->left_Layer->backward_Matrix_Vector.push_back(l_Matrix);
			}
		}

		void upMatrix() override final {};
		void upMatrix_batch(Dtype upRate) override final {};
	};
}