#pragma once
#include "LayerQL.h"
#include <iostream>

namespace tinyDNN
{
	template <typename Dtype>
	class Dim_ReduceQL : public LayerQL<Dtype>
	{
	public:
		using MatrixD = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

		Dim_ReduceQL(LayerType type, int kerNum, int rowNum, int colNum) : LayerQL(type), kerNum(kerNum), rowNum(rowNum), colNum(colNum)
		{
			std::cout << "Dim_ReduceQL Start!" << std::endl;
		}
		~Dim_ReduceQL() override final { std::cout << "Dim_ReduceQL Over!" << std::endl; }

		void calForward(int type = 0) const override final
		{
			std::shared_ptr<MatrixQL<Dtype>> rightMatrix = std::make_shared<MatrixQL<Dtype>>(1, kerNum*rowNum*colNum);
			
			int dimNum = 0;
			for ( int i = 0 ; i < kerNum; i++ )
			{
				for ( int j = 0; j < rowNum; j ++ )
				{
					for (int k = 0; k < colNum; k++)
					{
						rightMatrix->setMatrixQL()(0, dimNum) = this->left_Layer->forward_Matrix_Vector[i]->getMatrixQL()(j,k);
						dimNum++;
					}
				}
			}
			this->right_Layer->forward_Matrix = rightMatrix;
		}


		void calBackward(int type = 0) override final
		{
			this->left_Layer->backward_Matrix_Vector.clear();
			
			MatrixD trans_01 = static_cast<MatrixD>(this->right_Layer->backward_Matrix->getMatrixQL());
			
			Eigen::Map<MatrixD> mapMatrix(	trans_01.data(), rowNum* kerNum , colNum  );

			for ( int i = 0; i < kerNum; i++ )
			{
				std::shared_ptr<MatrixQL<Dtype>> leftMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum,colNum);
				leftMatrix->setMatrixQL().block(0, 0, rowNum, colNum) = mapMatrix.block(0 + i * rowNum, 0, rowNum, colNum);
				this->left_Layer->backward_Matrix_Vector.push_back(leftMatrix);
			}
		}

		void upMatrix() override final {};
		void upMatrix_batch(Dtype upRate) override final {};

	private:
		int kerNum;
		int rowNum;
		int colNum;
	};
}