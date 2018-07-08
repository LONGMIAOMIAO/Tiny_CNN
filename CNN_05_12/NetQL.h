#pragma once
#include "LayerQL.h"
#include <vector>

namespace tinyDNN
{
	template <typename Dtype>
	class NetQL
	{
	public:
		NetQL();
		~NetQL();

	public:
		//用来装载做好的每层Layer
		static std::vector<std::shared_ptr<LayerQL<Dtype>>> layerQLVector;
	};

	template <typename Dtype>
	std::vector<std::shared_ptr<LayerQL<Dtype>>> NetQL<Dtype>::layerQLVector;

	template <typename Dtype>
	NetQL<Dtype>::NetQL()
	{
		std::cout << "Net started!" << std::endl;
	}

	template <typename Dtype>
	NetQL<Dtype>::~NetQL()
	{	
		std::cout << "Net end!" << std::endl;
	}
}