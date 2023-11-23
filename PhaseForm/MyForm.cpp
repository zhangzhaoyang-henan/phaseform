#include "MyForm.h"

using namespace PhaseForm;
using namespace std;


[STAThread]






int main(array<System::String^>^ args)
{
	Application::EnableVisualStyles();
	Application::Run(gcnew MyForm());
	
	return 0;
}

