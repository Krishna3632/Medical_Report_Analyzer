import pdfplumber

def test_pdf_extraction(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"✅ PDF opened successfully!")
            print(f"📄 Number of pages: {len(pdf.pages)}")
            
            # Test first page
            if pdf.pages:
                first_page = pdf.pages[0]
                text = first_page.extract_text()
                print(f"📝 First page text length: {len(text) if text else 0}")
                
                tables = first_page.extract_tables()
                print(f"📊 Tables found: {len(tables)}")
                
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test with your PDF
test_pdf_extraction("C:/Users/ks520/Downloads/pd.pdf")