#include "PooLayerQL.h"
namespace tinyDNN
{
	template <typename Dtype>
	PooLayerQL<Dtype>::PooLayerQL(LayerType type, int rowNum, int colNum) : LayerQL(type), rowNum(rowNum), colNum(colNum)
	{
		std::cout << "PooLayerQL Start!" << std::endl;

		//this->rowNum = rowNum;
		//this->colNum = colNum;
	}

	template <typename Dtype>
	PooLayerQL<Dtype>::~PooLayerQL()
	{
		std::cout << "PooLayerQL Over!" << std::endl;
	}

	template <typename Dtype>
	void PooLayerQL<Dtype>::calForward() const
	{
		//this->calForward_MaxNum();
		this->calForward_Average();

		//Eigen::MatrixXd tt(4, 4);
		//tt.setZero();
		//tt.block(0, 0, 2, 2).setConstant(12.22);
		//std::cout << tt << std::endl;
	}




	template <typename Dtype>
	void PooLayerQL<Dtype>::calForward_Vector_Average() const
	{	
		for ( auto k = this->left_Layer->forward_Matrix_Vector.begin(); k != this->left_Layer->forward_Matrix_Vector.end(); k++ )
		{
			std::shared_ptr<MatrixQL<Dtype>>& inMatrix = *k;
			
			std::shared_ptr<MatrixQL<Dtype>> outMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum,colNum);
			
			for (int i = 0; i < rowNum; i++)
			{
				for (int j = 0; j < colNum; j++)
				{
					outMatrix->setMatrixQL()(i, j) = inMatrix->getMatrixQL().block(i * 2, j * 2, 2, 2).mean();
				}
			}
			this->right_Layer->forward_Matrix_Vector.push_back( outMatrix );
		}
	}




	template <typename Dtype>
	void PooLayerQL<Dtype>::calForward_MaxNum() const
	{
		for (int i = 0; i < rowNum; i++)
		{
			for (int j = 0; j < colNum; j++)
			{
				this->right_Layer->forward_Matrix->setMatrixQL()(i, j) = this->left_Layer->forward_Matrix->getMatrixQL().block(i * 2, j * 2, 2, 2).maxCoeff();
			}
		}
	}

	template <typename Dtype>
	void PooLayerQL<Dtype>::calForward_Average() const
	{
		for (int i = 0; i < rowNum; i++)
		{
			for (int j = 0; j < colNum; j++)
			{
				this->right_Layer->forward_Matrix->setMatrixQL()(i, j) = this->left_Layer->forward_Matrix->getMatrixQL().block(i * 2, j * 2, 2, 2).mean();
			}
		}
	}

	template <typename Dtype>
	void PooLayerQL<Dtype>::calBackward()
	{
		this->calBackward_Average();

	}

	template <typename Dtype>
	void PooLayerQL<Dtype>::calBackward_Average()
	{
		for (int i = 0; i < rowNum; i++)
		{
			for (int j = 0; j < colNum; j++)
			{
				//this->right_Layer->forward_Matrix->setMatrixQL()(i, j) = this->left_Layer->forward_Matrix->getMatrixQL().block(i * 2, j * 2, 2, 2).mean();
				this->left_Layer->backward_Matrix->setMatrixQL().block(i * 2, j * 2, 2, 2).setConstant(this->right_Layer->backward_Matrix->getMatrixQL()(i, j));
			}
		}
	}

	template class PooLayerQL<double>;
}