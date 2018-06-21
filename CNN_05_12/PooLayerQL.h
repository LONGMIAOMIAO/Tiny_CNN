#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype>
	class PooLayerQL:public LayerQL<Dtype>
	{
	public:
		//	¶¨Òåµ×²ãEigen¾ØÕóÄ£°å
		//	using MatrixData = Eigen::Matrix <Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

		explicit PooLayerQL(LayerType type, int rowNum, int colNum);
		~PooLayerQL() override final;

		void calForward() const override final;
		void calForward_MaxNum() const;
		void calForward_Average() const;
		void calForward_Vector_Average() const;


		void calBackward() override final;
		void calBackward_Average();
		void calBackward_Vector_Average();


		void upMatrix() override final {};
		void upMatrix_batch(Dtype upRate) override final {};
	private:
		int rowNum;
		int colNum;
	};
}