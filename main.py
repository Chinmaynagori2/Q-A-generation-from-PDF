import os
import pytesseract
from langchain_community.document_loaders import UnstructuredPDFLoader
import pandas as pd
import os, random, json, glob, re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
from langchain_core.output_parsers.json import JsonOutputParser
import re
import os
from pathlib import Path

# point pytesseract to the executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Tell Unstructured exactly which agent class to use
os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract"


loader = UnstructuredPDFLoader("input/pdf/demo_chapter.pdf",mode="elements",strategy="auto")
data = loader.load()
print("ocr text loaded from pdf:", len(data), "elements")

# Convert the list of Document objects in 'data' to a DataFrame
df = pd.DataFrame([{"page_content": doc.page_content, **doc.metadata} for doc in data])
# filer out rows with uncategorized text.
df = df[df["category"] != "UncategorizedText"]
# remove the "Title" where is does not contain an actual heading(in NCERT, it is like "1.1.2 Heading") or "Activity". Because it is most likely the text from image or Footer or Header.
df = df[(df["category"] != "Title") | (df["page_content"].str.strip().str.match(r"^(Activity|\d+(\.\d+)*)"))]
# filter out narrative text that is too short(possibly just the text in a figure or page number)
clean_df1 = df[~((df['category'] == "NarrativeText") & (df['page_content'].str.len() <= 2))]


# Remove leading spaces from NarrativeText that starts with "e", "o", or "." and replace with "- " for indicating a bullet point
mask = (clean_df1['category'] == "NarrativeText") & (clean_df1['page_content'].str.strip().str.match(r"^(e|o|\.)\s+"))
clean_df1.loc[mask, 'page_content'] = clean_df1.loc[mask, 'page_content'].str.replace(
    r"^(e|o|\.)\s+", "- ", regex=True
)

# save clean text to a file
Path("input/text").mkdir(parents=True, exist_ok=True)
cleaned_text = "\n\n".join(clean_df1["page_content"].tolist())
with open("input/text/cleaned_ocr_text.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)
print("cleaned text saved")


###################################################################################################
def chunk_by_section(text: str) -> list[str]:
    # Positive lookahead for lines that start with section numbering
    chunks = re.split(r"\n(?=\d+(?:\.\d+)+\s+[A-Z])", text)
    return [c.strip() for c in chunks if c.strip()]

cleaned_path = Path("input/text/cleaned_ocr_text.txt")
chunks_dir   = Path("input/text/chunks")
chunks_dir.mkdir(parents=True, exist_ok=True)

# Read the entire cleaned text
text = cleaned_path.read_text(encoding="utf-8")

# chunking by sections of the chapter like "1.1 Introduction", "1.2 Background", etc.
chunks = chunk_by_section(text)
print(f"Found {len(chunks)} section-based chunks.")

# Save each chunk in chunks folder
for idx, chunk in enumerate(chunks, start=1):
    fname = chunks_dir / f"chunk_{idx:02d}.txt"
    fname.write_text(chunk, encoding="utf-8")
    print(f"Saved {fname.name} ({len(chunk.split())} words)")

####################################################################################################

# initializing llm and embedder
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
# use gemini-2.5-flash for better performance if available(100 calls per day)


chunk_paths = sorted(glob.glob("input/text/chunks/chunk_*.txt"))
chunks = [open(path, "r", encoding="utf-8").read().strip() for path in chunk_paths]
# merging small chunks together
merged_chunks = []
i = 0
while i < len(chunks):
    if len(chunks[i].split()) < 15 and i + 1 < len(chunks):
        merged = chunks[i] + "\n\n" + chunks[i + 1]
        merged_chunks.append(merged)
        i += 2
    else:
        merged_chunks.append(chunks[i])
        i += 1

chunks = merged_chunks


# prompts for the different types of questions using json output parser
parser = JsonOutputParser()
PROMPTS = {
    "MCQ": PromptTemplate(template="""Context:
{context}

Generate ONE multiple-choice question with 4 options (A-D). 
Provide the correct option and a short explanation of why it's correct in the `reason` field.
Output EXACTLY JSON:
{{
  "question": "...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "B",
  "reason": "..."
}}
{format_instructions}
""",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
),
    "Very Short": PromptTemplate(template="""Context:
{context}

Generate ONE very-short-answer question (answer ≤ 5 words). 
Include a brief explanation of why the answer is correct in `reason`.
Output EXACTLY JSON:
{{
  "question": "...",
  "answer": "...",
  "reason": "..."
}}""",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
),
    "Short": PromptTemplate(template= """Context:
{context}

Generate ONE short-answer question (answer ≈ 2-3 sentences). 
Include a concise explanation of how you got the answer in the `reason` field.
Output EXACTLY JSON:
{{
  "question": "...",
  "answer": "...",
  "reason": "..."
}}""",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
),
    "Long": PromptTemplate(template= """Context:
{context}

Generate ONE long-answer question (answer ≈ 1 paragraph).It can go in detail about the 
topic - a discussion, explanation, or analysis.
Provide the answer and then a brief explanation of how you got the answer in the `reason` field.
Output EXACTLY JSON:
{{
  "question": "...",
  "answer": "...",
  "reason": "..."
}}""",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
}



seen_q_embeds = []
# function to check if the question is topically unique
def unique_question(question, threshold=0.8):
    emb = embedder.encode(question, convert_to_tensor=True)
    for old in seen_q_embeds:
        if util.cos_sim(emb, old).item() > threshold:
            return False
    seen_q_embeds.append(emb)
    return True

def pick_questions(qtype, chunks, needed):
    qas = []
    pool = chunks.copy()
    random.shuffle(pool)         # shuffle the chunks to get random questions
    for ch in pool:
        if len(qas) >= needed:
            break
        # For long answers, if chunk is too small, merge with the next chunk
        context = ch
        if qtype == "Long" and len(ch.split()) < 100:
            # find next chunk to merge
            idx = chunks.index(ch)
            if idx + 1 < len(chunks):
                context = ch + "\n\n" + chunks[idx+1]

        prompt = PROMPTS[qtype].format(context=context)
        try:
            resp = llm.invoke(prompt).content.strip()
            if resp.startswith("```"):
                resp = resp.strip("`").split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
            qa = json.loads(resp)
        except:
            print("trying again")        # there can be times when correct json is not returned and may give error
            continue
        # check if the question is unique
        if unique_question(qa["question"]) == False:
            continue
        qa["type"] = qtype

        
        val_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {qa['question']}\n"
            f"Answer: {qa['answer']}\n\n"
            f"I have given context and quesetion and answer pair. Now give score of it's accuracy between 0 and 100 (not just 0 and 100)"
            f"Give a score from 0 to 100, where:\n"
            f"100 = fully accurate and directly supported\n"
            f"75 = mostly accurate, minor detail missing or vague\n"
            f"50 = partially accurate or incomplete\n"
            f"25 = mostly inaccurate\n"
            f"0 = completely wrong or unrelated\n\n"
            f"Output only the integer score."
        )

        # val_prompt = (
        #     f"Context:\n{context}\n\n"
        #     f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
        #     "i have given context and quesetion and answer pair. Now give score of it's accuracy between 0 and 100 (not just 0 and 100). output only the integer."
        # )
        score = llm.invoke(val_prompt).content.strip()
        qa["accuracy_score"] = int(score)
        qas.append(qa)

    return qas


#generating 5 questions of each type with random 5 chunks
all_mcqs       = pick_questions("MCQ",        chunks, 5)
time.sleep(60)
all_very_short = pick_questions("Very Short", chunks, 5)
time.sleep(60)
all_short      = pick_questions("Short",     chunks, 5)
time.sleep(60)
all_long       = pick_questions("Long",      chunks, 5)


results = all_mcqs + all_very_short + all_short + all_long
df = pd.DataFrame(results, columns=["question","type","answer","reason","accuracy_score"])
os.makedirs("output", exist_ok=True)
df.to_csv("output/qas.csv", index=False)


# plot the distribution of accuracy_score
plt.figure(figsize=(8, 5))
df['accuracy_score'].hist(bins=range(0, 110, 10), edgecolor='black')
plt.title('Distribution of accuracy_score')
plt.xlabel('Accuracy Score')
plt.ylabel('Count')
plt.xticks(range(0, 110, 10))
plt.show()
