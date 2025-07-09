# NeuroBridge Project - Automated Question Generation from PDF Text

## Task

This project extracts text from a PDF chapter using OCR, cleans and chunks the text, and uses a language model to generate various types of questions (MCQ, very short, short, long) from the content. The generated questions are validated for accuracy and exported to a CSV file, with a report on their accuracy distribution.

---

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
   Required packages include:
   - `pytesseract`
   - `langchain_community`
   - `pandas`
   - `sentence-transformers`
   - `langchain_google_genai`
   - `matplotlib`
   - `python-dotenv`
   
   Installing of pytesseract and tesseract should be taken care of carefully as it requires
   1). conda install -c conda-forge tesseract if using a conda venv
   2). If not, then you have to add Path in syste environment variable in addition to downloading using pip.

3. **Install Tesseract OCR**  
   Download and install from: https://github.com/tesseract-ocr/tesseract  
   Update the path in `main.py` if needed:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

4. **Set up Google API Key**  
   Add your Google API key to a `.env` file:
   ```
   GOOGLE_API_KEY= "YOUR_API_KEY"
   ```

5. **Prepare input PDF**  
   Place your PDF file in `input/pdf/demo_chapter.pdf`.

---

## Usage
Upload the downloaded NCERT textbook chapter PDF to input/pdf.
Run the main script:
```sh
python main.py
```

This will:
- Extract and clean text from the PDF.
- Chunk the text by section.
- Generate questions of different types using a language model.
- Validate the generated questions for accuracy.
- Save the questions and their metadata to `output/qas.csv`.
- Display a histogram of the accuracy scores.

---

## Validation Report

After running, the script generates a validation report in the form of a histogram showing the distribution of accuracy scores for the generated questions.

- The output CSV (`output/qas.csv`) contains columns: `question`, `type`, `answer`, `reason`, `accuracy_score`.
- The histogram visualizes how many questions fall into each accuracy bracket (0-100, in steps of 10).

Example (from a previous run):

- Most questions scored 90-100, indicating high accuracy.
- Some questions may score lower if the model's answer is incomplete or less relevant.

---

## File Structure

- `main.py` - Main pipeline script.
- `input/pdf/` - Input PDF files.
- `input/text/` - Cleaned text and chunks.
- `output/qas.csv` - Generated questions and validation scores.

---

## Notes

- The script uses time delays (`time.sleep(60)`) between question batches to avoid API rate limits.
- For best results, use high-quality PDFs and ensure Tesseract OCR is correctly installed.