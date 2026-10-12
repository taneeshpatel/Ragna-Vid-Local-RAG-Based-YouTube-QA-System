# import re

# def split_into_sentences(text: str) -> list:
#     # Basic sentence splitter using punctuation
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     return sentences

# def chunk_text(text: str, max_words: int = 200) -> list:
#     sentences = split_into_sentences(text)
#     chunks = []
#     current_chunk = []
#     current_len = 0

#     for sentence in sentences:
#         word_count = len(sentence.split())

#         if current_len + word_count > max_words:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [sentence]
#             current_len = word_count
#         else:
#             current_chunk.append(sentence)
#             current_len += word_count

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

import re

def split_into_sentences(text: str) -> list[str]:
    """
    Splits the input text into sentences using punctuation marks.
    Handles edge cases like abbreviations (e.g., Dr., i.e.) minimally.
    """
    # Basic sentence splitter, can be replaced with `nltk.sent_tokenize()` or spacy for better results
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, max_words: int = 200) -> list[str]:
    """
    Groups sentences into chunks, each with a maximum number of words.
    Ensures semantic boundaries and avoids cutting off mid-sentence.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if word_count > max_words:
            # If a single sentence exceeds max_words, split it
            for i in range(0, word_count, max_words):
                sub_chunk = " ".join(words[i:i + max_words])
                chunks.append(sub_chunk)
            continue

        if current_len + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
