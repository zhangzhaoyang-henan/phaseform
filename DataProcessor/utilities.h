#include <fstream>
namespace Utility
{
	bool is_file_exist(const char *fileName)
	{
    	std::ifstream infile(fileName);
    	return infile.good();
	}
};