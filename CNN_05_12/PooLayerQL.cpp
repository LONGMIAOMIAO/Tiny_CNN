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
	//向前传播
	template <typename Dtype>
	void PooLayerQL<Dtype>::calForward(int type = 0) const
	{
		this->calForward_Vector_Average();
		//this->calForward_MaxNum();
		//this->calForward_Average();

		//Eigen::MatrixXd tt(4, 4);
		//tt.setZero();
		//tt.block(0, 0, 2, 2).setConstant(12.22);
		//std::cout << tt << std::endl;
	}

	//采用	Vector向前传播	计算方法
	template <typename Dtype>
	void PooLayerQL<Dtype>::calForward_Vector_Average() const
	{	
		this->right_Layer->forward_Matrix_Vector.clear();
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

	//采用 矩阵最大值	 池化方法
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

	//采用	矩阵平均值	池化方法
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
	//========================================================================================================================
	//反向传播
	template <typename Dtype>
	void PooLayerQL<Dtype>::calBackward(int type = 0)
	{
		//this->calBackward_Average();
		this->calBackward_Vector_Average();
	}
	//采用	矩阵平均值	反向传播
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

	//采用	Vector 平均值	反向传播
	template <typename Dtype>
	void PooLayerQL<Dtype>::calBackward_Vector_Average()
	{
		this->left_Layer->backward_Matrix_Vector.clear();
		for ( auto k = this->right_Layer->backward_Matrix_Vector.begin(); k != this->right_Layer->backward_Matrix_Vector.end(); k++ )
		{
			std::shared_ptr<MatrixQL<Dtype>>& inMatrix_Back = *k;
			std::shared_ptr<MatrixQL<Dtype>> outMatrix_Back = std::make_shared<MatrixQL<Dtype>>(rowNum*2, colNum*2);

			for (int i = 0; i < rowNum; i++)
			{
				for (int j = 0; j < colNum; j++)
				{
					outMatrix_Back->setMatrixQL().block(i * 2, j * 2, 2, 2).setConstant(inMatrix_Back->getMatrixQL()(i, j));
				}
			}
			this->left_Layer->backward_Matrix_Vector.push_back( outMatrix_Back );
		}
	}

	template class PooLayerQL<double>;
}