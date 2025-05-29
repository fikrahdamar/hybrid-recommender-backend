from rapidfuzz import fuzz
import re
import string

def preprocess_title(title):
    title = title.lower()
    title = re.sub(r'\(\d{4}\)', '', title)  # hapus tahun
    title = title.translate(str.maketrans('', '', string.punctuation))  # hapus tanda baca
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def get_best_matches(source_titles, target_titles, threshold=90):
    matches = {}
    target_processed = {title: preprocess_title(title) for title in target_titles}
    
    for source in source_titles:
        source_proc = preprocess_title(source)
        best_match = None
        best_score = 0
        for target, target_proc in target_processed.items():
            score = (
                fuzz.token_sort_ratio(source_proc, target_proc) * 0.6 +
                fuzz.partial_ratio(source_proc, target_proc) * 0.4
            )
            if score > best_score:
                best_score = score
                best_match = target
        if best_score >= threshold:
            matches[source] = (best_match, best_score)
    return matches
