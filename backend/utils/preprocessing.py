import re


def clean_text(text: str) -> str:
    """
    Social-media aware text normalization.

    Inspired by preprocessing patterns used in the resources benchmark:
    remove URLs/HTML, strip mentions, normalize hashtags/retweets, and
    collapse whitespace while keeping core lexical signal.
    """
    text = str(text).strip()
    if not text:
        return ""

    # URLs and html tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    # Common social text artifacts
    text = re.sub(r"^RT[\s:]+", " ", text)  # retweet marker
    text = re.sub(r"@\w+", " ", text)  # mentions
    text = re.sub(r"#", " ", text)  # keep hashtag token but drop '#'

    # Remove escaped placeholders often seen in scraped reddit data
    text = re.sub(r"\[removed\]|\[deleted\]", " ", text, flags=re.IGNORECASE)

    # Keep punctuation-light text for transformer tokenization
    text = re.sub(r"[^\w\s'\-.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
