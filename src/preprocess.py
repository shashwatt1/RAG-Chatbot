import os
import re
import PyPDF2


def process_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Fix hyphenated words at the end of lines
                page_text = re.sub(r'(\w)-\n(\w)', r'\1\2', page_text)
                raw_text += page_text + "\n\n"

    # clean up text formatting while keeping lists intact
    text = re.sub(r'(?<!\n)\n(?!\n)(?![A-Z0-9•\-])', ' ', raw_text)
    blocks = [b.strip() for b in re.split(r'\n\n+', text) if b.strip()]

    # extract standalone sentences
    sentences = []
    for b in blocks:
        b = re.sub(r'\s+', ' ', b)
        # split by standard end punctuation
        sents = re.split(r'(?<=[.!?])\s+', b)
        for s in sents:
            if s.strip():
                sentences.append(s.strip())

    # chunk constraints
    target_words = 200
    overlap_words = 25
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in sentences:
        words = len(sent.split())
        current_chunk.append(sent)
        current_length += words
        
        # save chunk if we hit the target
        if current_length >= target_words:
            chunks.append(" ".join(current_chunk))
            
            # create text overlap to preserve context across chunks
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                s_words = len(s.split())
                if overlap_len + s_words <= overlap_words * 1.5:
                    overlap_chunk.insert(0, s)
                    overlap_len += s_words
                else:
                    if not overlap_chunk: # keep at least one sentence
                        overlap_chunk.insert(0, s)
                    break
                    
            current_chunk = overlap_chunk
            current_length = sum(len(s.split()) for s in current_chunk)

    # append any leftover text
    if current_chunk and (not chunks or current_length > overlap_words * 1.5):
        chunks.append(" ".join(current_chunk))

    # attach metadata to each chunk
    results = []
    for i, c in enumerate(chunks):
        results.append({
            "chunk_id": str(i),
            "text": c,
            "source": os.path.basename(file_path)
        })

    return results
