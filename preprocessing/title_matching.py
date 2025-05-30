from rapidfuzz import fuzz
from tqdm import tqdm
from .text_cleaning import preprocess_title


def get_best_matches(source_titles, target_titles, threshold=90):
    matches = {}
    target_processed = {title: preprocess_title(title) for title in target_titles}
    
    for source in tqdm(source_titles, desc="Matching titles"):
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
