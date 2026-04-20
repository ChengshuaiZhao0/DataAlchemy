"""Special tokens used by the symbolic tokenizer.

Transformation tokens `[F1]`, `[F2]`, `[F3]` identify the three base
transformations ``f_rot``, ``f_pos``, ``f_reverse`` (see paper Appendix B).
"""

BOS_TOKEN: str = "<s>"
PAD_TOKEN: str = "<pad>"
EOS_TOKEN: str = "</s>"
UNK_TOKEN: str = "<unk>"
MASK_TOKEN: str = "<mask>"
SEP_TOKEN: str = "<sep>"
THINK_TOKEN: str = "<think>"
ANSWER_TOKEN: str = "<answer>"

TRANSFORMATION_TOKENS: list[str] = ["[F1]", "[F2]", "[F3]"]
# Backward-compat alias for code/configs that still import the old name.
RULE_TOKENS = TRANSFORMATION_TOKENS

F_ROT_TOKEN: str = TRANSFORMATION_TOKENS[0]
F_POS_TOKEN: str = TRANSFORMATION_TOKENS[1]
F_REVERSE_TOKEN: str = TRANSFORMATION_TOKENS[2]
