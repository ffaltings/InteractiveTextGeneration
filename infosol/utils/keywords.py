import re
import numpy as np

def order_keywords(target_text, keywords, highlighter):
    highlighted_string = highlighter.highlight(target_text, keywords)
    kw_re = re.compile('<kw>[^<]*</kw>')
    ordered_keywords = kw_re.findall(highlighted_string)
    ordered_keywords = [re.sub('<kw>', '', re.sub('</kw>', '', kw)) for kw in ordered_keywords]
    for i in range(len(ordered_keywords)):
        kw_matched = False
        for kw_pair in keywords:
            if kw_pair[0] == ordered_keywords[i]:
                ordered_keywords[i] = kw_pair
                kw_matched = True
                break
        if not kw_matched: ordered_keywords[i] = (ordered_keywords[i], 0)
    return ordered_keywords

def choose_top_kws(kws, n_kws):
    scores = [k[1] for k in kws]
    top_kws = [kws[i] for i in np.sort(np.argsort(scores)[-n_kws:])]
    return top_kws
