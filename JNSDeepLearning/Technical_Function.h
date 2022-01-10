#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace Technical_Function
{
	static vector<string> SplitData(string s, string divid)
	{
		vector<string> v;
		int start = 0;
		int d = s.find(divid);
		while (d != -1) 
		{
			v.push_back(s.substr(start, d - start));
			start = d + 1;
			d = s.find(divid, start);
		}
		v.push_back(s.substr(start, d - start));

		return v;
	}

	static vector<double> VS2VD(vector<string> _VS)
	{
		vector<double> Result;

		for(vector<string>::iterator iter = _VS.begin()+1; iter != _VS.end(); ++iter)
		{
			//cout << stod((*iter)) << endl;
			//cout << stod("301.13000000") << endl;
			//Result.push_back(stod(*iter));
		}

		return Result;
	}
};
