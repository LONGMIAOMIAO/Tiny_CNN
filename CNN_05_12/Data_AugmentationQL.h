#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class Data_AugmentationQL : public LayerQL<Dtype>
	{
	public:
		Data_AugmentationQL(LayerType type, int rowNum, int colNum) : LayerQL(type), rowNum(rowNum), colNum(colNum)
		{
			std::cout << "Data_AugmentationQL Start!" << std::endl;
		}
		~Data_AugmentationQL() override final
		{
			std::cout << "Data_AugmentationQL Over!" << std::endl;
		}
		void calForward(int type = 0) const override final
		{
			this->right_Layer->forward_Matrix_Vector.clear();
			int randCase = rand() % 2;
			for ( auto i = this->left_Layer->forward_Matrix_Vector.begin(); i != this->left_Layer->forward_Matrix_Vector.end(); i++ )
			{
				std::shared_ptr<MatrixQL<Dtype>> augMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);

				switch (randCase)
				{
				case 0:
					augMatrix = *i;
					break;
				case 1:
					augMatrix->setMatrixQL() = (*i)->getMatrixQL().rowwise().reverse();
					break;
				}
				
				this->right_Layer->forward_Matrix_Vector.push_back(augMatrix);
			}

		}
		void calBackward(int type = 0) override final {};

		void upMatrix() override final {};
		void upMatrix_batch(Dtype upRate) override final {};
	
	private:
		int rowNum;
		int colNum;
	};
}