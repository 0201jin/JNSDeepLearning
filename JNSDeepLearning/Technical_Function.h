#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace Technical_Function
{
	static void NumberString(string& _s)
	{
		for(int i = 0; i < _s.size(); ++i)
		{
			//숫자가 아니면 삭제 : _s.erase(i, 1);
			if(isdigit(_s[i]) == 0 && _s[i] != '.')
			{
				_s.erase(i, 1);
				--i;
			}
		}
	}

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

		for(vector<string>::iterator iter = _VS.begin(); iter != _VS.end(); ++iter)
		{
			cout << endl;
			NumberString((*iter));
			cout << (*iter) << endl;
			cout << stod((*iter)) << endl;
			//cout << stod((*iter)) << endl;
			//cout << stod((*iter)) << endl;
			//cout << stod("301.13000000") << endl;
			//Result.push_back(stod(*iter));
		}

		return Result;
	}
};
