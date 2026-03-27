
import re
from typing import List

def extract_claims(text: str, llm_pipeline) -> List[str]:
    """
    Use LLM to extract atomic factual claims from a paragraph.
    """
    prompt = f"""Extract all factual claims from the following text. 
List each claim as a separate bullet point starting with '- '.

Text: {text}

Claims:"""
    
    response = llm_pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0
    )
    output = response[0]['generated_text'].split("Claims:")[-1].strip()
    
    # Parse bullet points
    claims = []
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('- '):
            claim = line[2:].strip()
            if claim:
                claims.append(claim)
    return claims if claims else [text]  # fallback


def clean_wiki_text(text: str) -> str:
    """Remove wiki markup, extra spaces, etc."""
    text = re.sub(r'\[\d+\]', '', text)  # remove citations [1]
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
