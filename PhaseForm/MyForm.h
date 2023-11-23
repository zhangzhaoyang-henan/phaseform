#pragma once


namespace PhaseForm {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::IO;
	using namespace System::Drawing::Imaging;
	using namespace System::Runtime::InteropServices;
	using namespace std;


	/// <summary>
	/// MyForm 摘要
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO:  在此处添加构造函数代码
			//
		}

	protected:
		/// <summary>
		/// 清理所有正在使用的资源。
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: DevExpress::XtraEditors::PictureEdit^  pictureEdit1;


	private: DevExpress::XtraEditors::LabelControl^  labelControl1;
	private: DevExpress::XtraEditors::LabelControl^  labelControl2;
	private: DevExpress::XtraEditors::LabelControl^  labelControl3;
	private: DevExpress::XtraEditors::LabelControl^  labelControl4;
	private: System::Windows::Forms::Button^  button1;
	private: DevExpress::XtraEditors::TextEdit^  textEdit1;
	private: DevExpress::XtraEditors::TextEdit^  textEdit2;
	private: DevExpress::XtraEditors::HScrollBar^  hScrollBar1;
	private: System::Windows::Forms::ComboBox^  comboBox1;
	private: System::Windows::Forms::ComboBox^  comboBox2;

			 array<Bitmap^>^ img_list;

	private: System::Windows::Forms::Button^  button2;

	protected:

	private:
		/// <summary>
		/// 必需的设计器变量。
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// 设计器支持所需的方法 - 不要
		/// 使用代码编辑器修改此方法的内容。
		/// </summary>
		void InitializeComponent(void)
		{
			this->pictureEdit1 = (gcnew DevExpress::XtraEditors::PictureEdit());
			this->labelControl1 = (gcnew DevExpress::XtraEditors::LabelControl());
			this->labelControl2 = (gcnew DevExpress::XtraEditors::LabelControl());
			this->labelControl3 = (gcnew DevExpress::XtraEditors::LabelControl());
			this->labelControl4 = (gcnew DevExpress::XtraEditors::LabelControl());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->textEdit1 = (gcnew DevExpress::XtraEditors::TextEdit());
			this->textEdit2 = (gcnew DevExpress::XtraEditors::TextEdit());
			this->hScrollBar1 = (gcnew DevExpress::XtraEditors::HScrollBar());
			this->comboBox1 = (gcnew System::Windows::Forms::ComboBox());
			this->comboBox2 = (gcnew System::Windows::Forms::ComboBox());
			this->button2 = (gcnew System::Windows::Forms::Button());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureEdit1->Properties))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->textEdit1->Properties))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->textEdit2->Properties))->BeginInit();
			this->SuspendLayout();
			// 
			// pictureEdit1
			// 
			this->pictureEdit1->Location = System::Drawing::Point(1, 3);
			this->pictureEdit1->Name = L"pictureEdit1";
			this->pictureEdit1->Properties->ShowCameraMenuItem = DevExpress::XtraEditors::Controls::CameraMenuItemVisibility::Auto;
			this->pictureEdit1->Properties->SizeMode = DevExpress::XtraEditors::Controls::PictureSizeMode::Stretch;
			this->pictureEdit1->Size = System::Drawing::Size(1080, 624);
			this->pictureEdit1->TabIndex = 0;
			// 
			// labelControl1
			// 
			this->labelControl1->Location = System::Drawing::Point(1104, 68);
			this->labelControl1->Name = L"labelControl1";
			this->labelControl1->Size = System::Drawing::Size(96, 14);
			this->labelControl1->TabIndex = 3;
			this->labelControl1->Text = L"选择RawData文件";
			// 
			// labelControl2
			// 
			this->labelControl2->Location = System::Drawing::Point(1104, 189);
			this->labelControl2->Name = L"labelControl2";
			this->labelControl2->Size = System::Drawing::Size(72, 14);
			this->labelControl2->TabIndex = 4;
			this->labelControl2->Text = L"选择校准文件";
			// 
			// labelControl3
			// 
			this->labelControl3->Location = System::Drawing::Point(1104, 431);
			this->labelControl3->Name = L"labelControl3";
			this->labelControl3->Size = System::Drawing::Size(27, 14);
			this->labelControl3->TabIndex = 5;
			this->labelControl3->Text = L"A2：";
			// 
			// labelControl4
			// 
			this->labelControl4->Location = System::Drawing::Point(1104, 469);
			this->labelControl4->Name = L"labelControl4";
			this->labelControl4->Size = System::Drawing::Size(27, 14);
			this->labelControl4->TabIndex = 6;
			this->labelControl4->Text = L"A3：";
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(1104, 366);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(96, 40);
			this->button1->TabIndex = 7;
			this->button1->Text = L"图像优化";
			this->button1->UseVisualStyleBackColor = true;
			// 
			// textEdit1
			// 
			this->textEdit1->Location = System::Drawing::Point(1138, 428);
			this->textEdit1->Name = L"textEdit1";
			this->textEdit1->Size = System::Drawing::Size(131, 20);
			this->textEdit1->TabIndex = 8;
			// 
			// textEdit2
			// 
			this->textEdit2->Location = System::Drawing::Point(1138, 466);
			this->textEdit2->Name = L"textEdit2";
			this->textEdit2->Size = System::Drawing::Size(131, 20);
			this->textEdit2->TabIndex = 9;
			// 
			// hScrollBar1
			// 
			this->hScrollBar1->Location = System::Drawing::Point(1, 629);
			this->hScrollBar1->Name = L"hScrollBar1";
			this->hScrollBar1->Size = System::Drawing::Size(1080, 15);
			this->hScrollBar1->TabIndex = 10;
			this->hScrollBar1->Scroll += gcnew System::Windows::Forms::ScrollEventHandler(this, &MyForm::hScrollBar1_Scroll);
			// 
			// comboBox1
			// 
			this->comboBox1->FormattingEnabled = true;
			this->comboBox1->Location = System::Drawing::Point(1104, 100);
			this->comboBox1->Name = L"comboBox1";
			this->comboBox1->Size = System::Drawing::Size(190, 20);
			this->comboBox1->TabIndex = 11;
			this->comboBox1->DropDown += gcnew System::EventHandler(this, &MyForm::comboBox1_DropDown);
			// 
			// comboBox2
			// 
			this->comboBox2->FormattingEnabled = true;
			this->comboBox2->Location = System::Drawing::Point(1104, 226);
			this->comboBox2->Name = L"comboBox2";
			this->comboBox2->Size = System::Drawing::Size(190, 20);
			this->comboBox2->TabIndex = 12;
			this->comboBox2->DropDown += gcnew System::EventHandler(this, &MyForm::comboBox2_DropDown);
			// 
			// button2
			// 
			this->button2->Location = System::Drawing::Point(1104, 139);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(75, 23);
			this->button2->TabIndex = 13;
			this->button2->Text = L"button2";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &MyForm::button2_Click);
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 12);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1319, 651);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->comboBox2);
			this->Controls->Add(this->comboBox1);
			this->Controls->Add(this->hScrollBar1);
			this->Controls->Add(this->textEdit2);
			this->Controls->Add(this->textEdit1);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->labelControl4);
			this->Controls->Add(this->labelControl3);
			this->Controls->Add(this->labelControl2);
			this->Controls->Add(this->labelControl1);
			this->Controls->Add(this->pictureEdit1);
			this->Name = L"MyForm";
			this->Text = L"MyForm";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureEdit1->Properties))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->textEdit1->Properties))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->textEdit2->Properties))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
private: System::Void comboBox1_DropDown(System::Object^  sender, System::EventArgs^  e) {
	try
	{
		comboBox1->Items->Clear();

		DirectoryInfo^ folder = gcnew DirectoryInfo("./bin");
		array<DirectoryInfo^>^ dirInfo=  folder->GetDirectories();
		array<FileInfo^>^ file = folder->GetFiles();

		for each (FileInfo^ fileitem in file)
		{
			comboBox1->Items->Add(fileitem->DirectoryName+"\\"+fileitem->Name);
		}
	}

	catch (Exception^ e)
	{
		throw e;
	}
}
private: System::Void comboBox2_DropDown(System::Object^  sender, System::EventArgs^  e) {
	try
	{
		comboBox2->Items->Clear();

		DirectoryInfo^ folder = gcnew DirectoryInfo("./calibration");
		array<DirectoryInfo^>^ dirInfo = folder->GetDirectories();
		array<FileInfo^>^ file = folder->GetFiles();

		for each (FileInfo^ fileitem in file)
		{
			comboBox2->Items->Add(fileitem->DirectoryName + "\\" + fileitem->Name);
		}
	}
	catch (Exception^ e)
	{
		throw e;
	}
}
private: System::Void hScrollBar1_Scroll(System::Object^  sender, System::Windows::Forms::ScrollEventArgs^  e) {
	if (e->NewValue >= 0)
	{

		this->pictureEdit1->Image = this->img_list[e->NewValue];//this->getSpecificPage(img_location, 0);
		this->pictureEdit1->Invalidate();

		this->hScrollBar1->Text = System::Convert::ToString(e->NewValue + 1) + " / " + (hScrollBar1->Maximum + 1).ToString();
	}
}

	//  使用tiff文件转为byte
	public: array<Byte>^  getBytesFromBitmap(Bitmap^ bmp) {
		try {
			Bitmap ^b = gcnew Bitmap(bmp);
			MemoryStream ^ms = gcnew MemoryStream();
			b->Save(ms, System::Drawing::Imaging::ImageFormat::Bmp);
			array<Byte> ^bytes = ms->GetBuffer();  //byte[]   bytes=   ms.ToArray(); 这两句都可以
			ms->Close();
			return bytes;
		}
		catch (Exception^ ex) {
			throw ex;
		}
	}

	public:array<Byte>^ getBytesFromBin(String^ path)
	{
		array<Byte>^ imagebyte;
		try
		{
			FileStream^ fs = gcnew FileStream(path, FileMode::Open);
			BinaryReader^ br = gcnew BinaryReader(fs);

			while (br->BaseStream->Position<br->BaseStream->Length)
			{
				imagebyte = br->ReadBytes(fs->Length);
			}
			return imagebyte;
			fs->Close();
		}
		catch (Exception^ e)
		{
			throw e;
		}
	}


	private: Bitmap^ bitmapFromByteArray(array<Byte>^ bmpRawData, unsigned int frame, unsigned int width, unsigned int height)
	{
		// 新建一个8位灰度位图，并锁定内存区域操作
		Bitmap^ bitmap = gcnew Bitmap(width, height, PixelFormat::Format8bppIndexed);
		BitmapData^ bmpData = bitmap->LockBits(Rectangle(0, 0, width, height),
			ImageLockMode::WriteOnly, PixelFormat::Format8bppIndexed);

		// 计算图像参数
		int offset = bmpData->Stride - bmpData->Width;        // 计算每行未用空间字节数
		IntPtr ptr = bmpData->Scan0;                         // 获取首地址
		int scanBytes = bmpData->Stride * bmpData->Height;    // 图像字节数 = 扫描字节数 * 高度
		array<Byte>^ grayValues = gcnew array<Byte>(scanBytes);            // 为图像数据分配内存

		// 为图像数据赋值
		int posSrc = 0, posScan = 0;                        // rawValues和grayValues的索引
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				grayValues[posScan++] = bmpRawData[posSrc++ + frame*width*height];
			}
			// 跳过图像数据每行未用空间的字节，length = stride - width * bytePerPixel
			posScan += offset;
		}

		// 内存解锁
		Marshal::Copy(grayValues, 0, ptr, scanBytes);
		bitmap->UnlockBits(bmpData);  // 解锁内存区域

		// 修改生成位图的索引表，从伪彩修改为灰度

		// 获取一个Format8bppIndexed格式图像的Palette对象
		Bitmap^ bmp = gcnew Bitmap(1, 1, PixelFormat::Format8bppIndexed);
		ColorPalette^ palette = bmp->Palette;
		for (int i = 0; i < 256; i++)
		{
			palette->Entries[i] = Color::FromArgb(i, i, i);
		}
		// 修改生成位图的索引表
		bitmap->Palette = palette;

		return bitmap;
	}
	private: Bitmap^ bitmapFromByteArrayByOffset(array<Byte>^ bmpRawData, unsigned int dataOffset, unsigned int width, unsigned int height)
	{
		// 新建一个8位灰度位图，并锁定内存区域操作
		Bitmap^ bitmap = gcnew Bitmap(width, height, PixelFormat::Format8bppIndexed);
		BitmapData^ bmpData = bitmap->LockBits(Rectangle(0, 0, width, height),
			ImageLockMode::WriteOnly, PixelFormat::Format8bppIndexed);

		// 计算图像参数
		int offset = bmpData->Stride - bmpData->Width;        // 计算每行未用空间字节数
		IntPtr ptr = bmpData->Scan0;                         // 获取首地址
		int scanBytes = bmpData->Stride * bmpData->Height;    // 图像字节数 = 扫描字节数 * 高度
		array<Byte>^ grayValues = gcnew array<Byte>(scanBytes);            // 为图像数据分配内存

		// 为图像数据赋值
		int posSrc = 0, posScan = 0;                        // rawValues和grayValues的索引
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				grayValues[posScan++] = bmpRawData[posSrc++ + dataOffset];
			}
			// 跳过图像数据每行未用空间的字节，length = stride - width * bytePerPixel
			posScan += offset;
		}

		// 内存解锁
		Marshal::Copy(grayValues, 0, ptr, scanBytes);
		bitmap->UnlockBits(bmpData);  // 解锁内存区域

		// 修改生成位图的索引表，从伪彩修改为灰度

		// 获取一个Format8bppIndexed格式图像的Palette对象
		Bitmap^ bmp = gcnew Bitmap(1, 1, PixelFormat::Format8bppIndexed);
		ColorPalette^ palette = bmp->Palette;
		for (int i = 0; i < 256; i++)
		{
			palette->Entries[i] = Color::FromArgb(i, i, i);
		}
		// 修改生成位图的索引表
		bitmap->Palette = palette;

		return bitmap;
	}
			 /// <summary>
			 /// 将数组转换成彩色图片
			 /// </summary>
			 /// <param name="rawValues">图像的byte数组</param>
			 /// <param name="width">图像的宽</param>
			 /// <param name="height">图像的高</param>
			 /// <returns>Bitmap对象</returns>
	private: Bitmap^ ArrayToColorBitmap(array<Byte>^ rawValues, int width, int height)
	{
		//// 申请目标位图的变量，并将其内存区域锁定
		try
		{
			Bitmap^ bmp = gcnew Bitmap(width, height, PixelFormat::Format24bppRgb);
			Rectangle rect = Rectangle(0, 0, width, height);
			BitmapData^ bitmapData = bmp->LockBits(rect, ImageLockMode::WriteOnly, PixelFormat::Format24bppRgb);
			IntPtr iptr = bitmapData->Scan0;  // 获取bmpData的内存起始位置  

			//// 用Marshal的Copy方法，将刚才得到的内存字节数组复制到BitmapData中  
			System::Runtime::InteropServices::Marshal::Copy(rawValues, 0, iptr, width * height * 3);

			bmp->UnlockBits(bitmapData);

			//// 算法到此结束，返回结果  

			return bmp;
		}
		catch (Exception^ ex)
		{
			return nullptr;
		}
	}


	private: System::Void button2_Click(System::Object^  sender, System::EventArgs^  e) {
		
		delete this->pictureEdit1->Image;
		String^ imagepath = comboBox1->SelectedItem->ToString();
		array<Byte>^imagebyte= getBytesFromBin(imagepath);
		this->pictureEdit1->Image = this->bitmapFromByteArray(imagebyte, 0, 1200, imagebyte->Length/(1*1200));

	}
};
}
