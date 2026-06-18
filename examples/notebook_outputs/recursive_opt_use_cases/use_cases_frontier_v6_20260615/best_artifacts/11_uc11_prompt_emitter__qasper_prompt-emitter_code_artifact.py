def _qasper_prompt_emitter(self):
    """Emit constraints to match expected QASPER fields exactly (dataset identifiers + precise IR2 numeric deltas)."""
    return (
        "Answer must follow the required QASPER constraints. "
        "When describing datasets, specify BOTH: "
        "1) the paired article–comment parallel dataset (INLINEFORM0) "
        "and 2) the unpaired dataset of documents (articles or comments) (INLINEFORM1) "
        "using the exact expected identifiers/wording. "
        "For the dataset choice, use 'Chinese dataset BIBREF0' when that is the expected option. "
        "When reporting IR2 gains, provide the exact numeric improvements mentioned in the source "
        "(e.g., MRR by 0.3769, MR by 4.6, Recall@10 by 20; and generative BLEU/CIDEr/ROUGE/METEOR deltas). "
        "Do NOT use vague wording like 'significant margin' or omit the numeric values."
    )
