#include "Padding_LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	Padding_LayerQL<Dtype>::Padding_LayerQL(LayerType type, int rowNum, int colNum, int padSize) : LayerQL(type), rowNum(rowNum), colNum(colNum), padSize(padSize)
	{
		std::cout << "PooLayerQL Start!" << std::endl;

	}

	template <typename Dtype>
	Padding_LayerQL<Dtype>::~Padding_LayerQL()
	{
		std::cout << "PooLayerQL End!" << std::endl;
	}

	template <typename Dtype>
	void Padding_LayerQL<Dtype>::calForward(int type = 0) const
	{
		this->calForward_Vector();
	}

	template <typename Dtype>
	void Padding_LayerQL<Dtype>::calForward_Vector() const
	{
		this->right_Layer->forward_Matrix_Vector.clear();
		std::for_each(this->left_Layer->forward_Matrix_Vector.begin(), this->left_Layer->forward_Matrix_Vector.end(), [&]( std::shared_ptr<MatrixQL<Dtype>>& matrixPtr ) 
		{
			std::shared_ptr<MatrixQL<Dtype>> paddingMatrix = std::make_shared<MatrixQL<Dtype>>( rowNum + 2 * padSize, colNum + 2 * padSize );
			paddingMatrix->setMatrixQL().setZero();

			paddingMatrix->setMatrixQL().block(padSize, padSize, rowNum, colNum) = matrixPtr->getMatrixQL().block(0, 0, rowNum, colNum);

			this->right_Layer->forward_Matrix_Vector.push_back( paddingMatrix );
		});
	}


	template <typename Dtype>
	void Padding_LayerQL<Dtype>::calBackward(int type = 0)
	{
		this->calBackward_Vector();
	}


	template <typename Dtype>
	void Padding_LayerQL<Dtype>::calBackward_Vector()
	{
		this->left_Layer->backward_Matrix_Vector.clear();

		//std::for_each(this->right_Layer->backward_Matrix_Vector.begin(), this->right_Layer->backward_Matrix_Vector.end(), [&](std::shared_ptr<MatrixQL<Dtype>>& matrixPtr))
		//{

		//});

	}


	template class Padding_LayerQL<double>;
}