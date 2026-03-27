
from transformers import pipeline
from utils import extract_claims, clean_wiki_text
from retriever import WikiRetriever

class SelfCritiqueAgent:
    def __init__(self, model_name="google/gemma-2-2b-it"):
        print(f"Loading model: {model_name}")
        self.llm = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.retriever = WikiRetriever(top_k=2)

    def generate_draft(self, question: str) -> str:
        prompt = f"Answer the following question concisely and factually:\nQ: {question}\nA:"
        response = self.llm(
            prompt,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.llm.tokenizer.eos_token_id
        )
        draft = response[0]['generated_text'].split("A:")[-1].strip()
        return draft.split("\n")[0]  # take first sentence

    def verify_claim(self, claim: str, evidence_list: List[str]) -> str:
        evidence_text = " ".join(evidence_list[:2])
        prompt = f"""Is the following claim supported by the evidence?

Claim: {claim}
Evidence: {evidence_text}

Answer only one word: Supported, Refuted, or Unknown."""
        
        response = self.llm(
            prompt,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.llm.tokenizer.eos_token_id
        )
        verdict = response[0]['generated_text'].split("Answer only one word:")[-1].strip().split()[0]
        return verdict if verdict in ["Supported", "Refuted", "Unknown"] else "Unknown"

    def critique_and_revise(self, question: str, draft: str) -> tuple:
        # Step 1: Extract claims
        claims = extract_claims(draft, self.llm)
        
        critiques = []
        for claim in claims:
            evidence = self.retriever.retrieve(claim)
            clean_evidence = [clean_wiki_text(e) for e in evidence]
            verdict = self.verify_claim(claim, clean_evidence)
            if verdict == "Refuted":
                critiques.append(f'The claim "{claim}" appears to be inaccurate based on retrieved evidence.')
            elif verdict == "Unknown":
                critiques.append(f'The claim "{claim}" could not be verified.')

        if not critiques:
            return draft, "All claims verified.", draft

        critique_text = " ".join(critiques)
        # Step 2: Revise
        revise_prompt = f"""You previously answered: "{draft}"
However, a fact-check revealed: {critique_text}
Please provide a revised, accurate, and concise answer to the original question: {question}"""
        
        revised = self.llm(
            revise_prompt,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.llm.tokenizer.eos_token_id
        )[0]['generated_text'].split("answer to the original question:")[-1].strip()
        revised = revised.split("\n")[0]

        return draft, critique_text, revised
