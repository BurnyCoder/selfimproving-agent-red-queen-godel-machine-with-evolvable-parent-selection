"""Outcome-blind removal of publication credits in the released Section-formatted text."""

import re
from statistics import median

# These are manuscript-level diagnostics. A funding phrase can also describe research.
# Keep the original exploratory heading definition fixed for the before/after comparison.
ACK = re.compile(r"(?:^|\n)(?:Section:\s*)?acknowledge?ments?\b", re.I)
FUNDING = re.compile(r"\b(?:funded by|supported (?:in part )?by)\b", re.I)
THANKS = re.compile(
    r"\b(?:thank(?:s|ful)?(?:\s+to)?|grateful to)\s+(?:the\s+)?(?:anonymous\s+)?reviewers?\b",
    re.I,
)
HEADING = re.compile(
    r"^(?:Section:[ \t]*)?(?:acknowledge?ments?\b(?: and disclosures? of funding)?"
    r"|(?:funding(?: (?:statement|information|disclosures?))?|disclosures? of funding)"
    r"(?=[ \t]*(?:$|[.:])))\s*[.:–—-]?\s*",
    re.I,
)
BOUNDARY = re.compile(
    r"^(?:Section:|Title:|Abstract:|References\s*[:.]?$|Bibliography\b|Appendix\b|Supplementary material\b)",
    re.I,
)
CREDIT = re.compile(
    r"\b(?:we|the authors)\s+(?:(?:would (?:also )?like to|wish to|also)\s+)?thank\b"
    r"|\b(?:we|the authors)\s+(?:are (?:grateful|thankful)|(?:would like to )?acknowledge)\b"
    r"|\b(?:acknowledges?|acknowledged)\s+(?:funding|(?:financial )?support)\b"
    r"|\b(?:funded|sponsored|financed|co-funded)\s+(?:in part\s+)?by\b"
    r"|\b(?:financial support|funding|grant(?:s)?)\s+(?:from|number|no\.?|agreement)\b"
    r"|\bsupported (?:in part )?by\b.{0,115}?\b(?:grants?|science fund|fellowships?|scholarship|research council|award|philanthropy|financial gifts?|ministry|DARPA|NSF|NIH|ERC|DFG|AFOSR|ONR|UKRI)\b",
    re.I,
)
# A named sponsor or a personal funding statement outside a labeled block.
SPONSOR = re.compile(
    r"\b(?:[Tt]his (?:work|research|project|study)|[A-Z]{2,6}(?: and [A-Z]{2,6})?) "
    r"(?:is |was |were |has been )?(?:(?:also|in part|partially|partly) )*supported "
    r"(?:in part )?by (?:the |The )?[A-Z](?![a-z]* (?:task|experiment|model)\b)"
)
SENTENCES = re.compile(r"(?<!\b[A-Z]\.)(?<!\bNo\.)(?<!\bDr\.)(?<=[.!?])(?=\s+[A-Z])")
# Some conversions place a scientific paragraph inside the credit section without a new header.
SCIENCE = re.compile(
    r"^(?:embeddings?\b|where\b|by assum|for (?:some|each|any)\b|proof\b|theorem\b|lemma\b|"
    r"(?:our|the) (?:first|second|experiments?|model|results?|method|algorithm)\b|we (?:show|derive|evaluate)\b)",
    re.I,
)
FIGURE = re.compile(r"^(?:figure|table)\s+\S+:$", re.I)
NAMED_THANKS = re.compile(r"^[A-Z][\w'-]+(?:[ ]+[A-Z][\w'-]+){1,3} (?:also )?thanks\b")


def remove_disclosures(text):
    """Drop the labeled credit and its prose, keeping adjacent sections and scientific lines."""
    kept = []
    in_credit_section = False
    in_references = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.casefold().rstrip(": .") in ("references", "bibliography"):
            in_references = True
        elif FIGURE.match(stripped) or stripped.casefold().startswith(
            ("section:", "appendix ", "caption:", "figures:", "tables:")
        ):
            in_references = False
        if in_references or stripped.startswith("Title:"):
            in_credit_section = False
            kept.append(line)
            continue
        if stripped.casefold().startswith("caption:"):
            caption = stripped.partition(":")[2].strip()
            in_credit_section = False
            if (
                NAMED_THANKS.match(caption)
                or HEADING.match(caption)
                or CREDIT.match(caption)
            ):
                # The extractor sometimes promotes an entire credit block to a figure.
                sentences = SENTENCES.split(caption)
                science = next(
                    (
                        i
                        for i, part in enumerate(sentences)
                        if SCIENCE.match(part.strip())
                    ),
                    len(sentences),
                )
                if science == len(sentences):
                    if (
                        len(kept) >= 2
                        and FIGURE.match(kept[-2].strip())
                        and kept[-1].strip().lower().startswith("type:")
                    ):
                        del kept[-2:]
                    continue
                line = "Caption: " + "".join(sentences[science:]).strip()
        header = HEADING.match(stripped)
        if header:
            tail = stripped[header.end() :]
            in_credit_section = True
            line = tail
        elif BOUNDARY.match(stripped):
            in_credit_section = False
        if not stripped:
            kept.append(line)
            continue
        parts = SENTENCES.split(line)
        keep = []
        for part in parts:
            if in_credit_section and SCIENCE.match(part.strip()):
                in_credit_section = False
            if not in_credit_section and not (
                CREDIT.search(part) or THANKS.search(part) or SPONSOR.search(part)
            ):
                keep.append(part)
        remaining = "".join(keep)
        if remaining.strip():
            kept.append(remaining if remaining == line else remaining.strip())
    return "\n".join(kept).strip()


def development_counts(papers):
    """Count papers carrying a cue and describe character length within each historical class."""
    result = {}
    for label in ("accept", "reject"):
        texts = [p["text"] for p in papers if p["label"] == label]
        lengths = sorted(map(len, texts))
        result[label] = dict(
            n=len(texts),
            acknowledgments=sum(bool(ACK.search(s)) for s in texts),
            funding_language=sum(bool(FUNDING.search(s)) for s in texts),
            reviewer_thanks=sum(bool(THANKS.search(s)) for s in texts),
            characters=dict(min=min(lengths), median=median(lengths), max=max(lengths)),
        )
    return result
