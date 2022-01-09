#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace Technical_Function
{
	static vector<string> SplitData(string _Data, char _delimiter)
	{
		vector<string> Result;
		stringstream ss(_Data);
		string tmp;

		while(getline(ss, tmp, _delimiter))
		{
			cout << tmp << endl;
			Result.push_back(tmp);
		}

		return Result;
	}

	static vector<double> VS2VD(vector<string> _VS)
	{
		vector<double> Result;

		for(vector<string>::iterator iter = _VS.begin(); iter != _VS.end(); ++iter)
		{
			cout << (*iter) << endl;
			cout << stod((*iter)) << endl;
			//Result.push_back(stod(*iter));
		}

		return Result;
	}
};
