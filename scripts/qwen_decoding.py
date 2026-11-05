"""Fast JSON decoding for the pinned Qwen tokenizer and LMFE 0.11.3.

Use LMFE's JSON free-text cache instead of a broad regex that traverses the
248k-token trie at each step. Numeric tokens remain usable in enum evidence
IDs, while the free explanation cache permits Korean prose without digits.
See THIRD_PARTY_NOTICES.md for upstream API provenance.
"""

PROSE_ALPHABET = ("".join(chr(i) for i in range(32, 127))
                  + "".join(chr(i) for i in range(0xAC00, 0xD7A4))
                  + "".join(chr(i) for i in range(0x3131, 0x318F))
                  + "·‘’“”…—")


def build_tokenizer_data(tokenizer, prose_filter="korean", forbid_numeric_prose=True):
    import transformers
    import transformers.tokenization_utils as compatibility
    if not hasattr(compatibility, "PreTrainedTokenizerBase"):
        compatibility.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

    data = build_token_enforcer_tokenizer_data(tokenizer)
    data.prose_filter=prose_filter
    data.forbid_numeric_prose=forbid_numeric_prose
    data.closing_boundary_tokens=[]
    for token_id,text,_ in data.regular_tokens:
        before=text.split('"',1)[0]
        if '"' in text and before and len(before)<=4 and all(c in '.,!?;:…—·)]} \t' for c in before) and not text.endswith('"'):
            data.closing_boundary_tokens.append((token_id,text))
    if prose_filter=="none" and not forbid_numeric_prose:
        return data
    # This small adapter targets the cache API of the explicitly pinned LMFE
    # release. The enum tokenizer trie is untouched, so E001 etc. stay valid.
    original = data.tokenizer_tree.json_freetext_tokens
    filtered = type(original)(data.use_bitmask, data.vocab_size)
    # The shortcut cache must obey the parser's character alphabet too.
    # Otherwise e.g. a Roman numeral is allowed by the cache, then makes the
    # character parser stop inside an unterminated JSON string. Partial UTF-8
    # tokens are excluded too: combining them can bypass this character filter.
    allowed_characters = set(PROSE_ALPHABET) if prose_filter=="korean" else set("".join(s for _,s,_ in data.regular_tokens))-set("�")
    allowed_characters -= set("0123456789\\") if forbid_numeric_prose else set("\\")
    for token_id, text, _ in data.regular_tokens:
        if all(character in allowed_characters for character in text):
            filtered.add_token(text, token_id)
    filtered.freeze()
    data.tokenizer_tree.json_freetext_tokens = filtered
    return data


def build_schema_decoder(tokenizer_data,schema,prose_filter="korean"):
    from lmformatenforcer import JsonSchemaParser,CharacterLevelParserConfig
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    config=CharacterLevelParserConfig()
    # The unrestricted condition uses the complete Unicode scalar range so
    # partial UTF-8 tokens cannot enter a parser with a narrower alphabet.
    config.alphabet=PROSE_ALPHABET if prose_filter=="korean" else "".join(chr(i) for i in range(0x110000) if not 0xD800<=i<=0xDFFF)
    base=build_transformers_prefix_allowed_tokens_fn(tokenizer_data,JsonSchemaParser(schema,config=config))
    # TokenEnforcer replaces parser.config with its tokenizer root alphabet.
    # Restore our explicit alphabet: a UTF-8-composed emoji need not be a root
    # character of any individual vocabulary token, but is valid Unicode.
    base.token_enforcer.root_parser.config=config
    return ClosingBoundaryTokens(base,tokenizer_data)


class ClosingBoundaryTokens:
    """Accept Qwen tokens like `.\",` across a JSON string boundary.

    LMFE's optimized cache handles a trailing quote and its trie handles a
    leading quote. A token with punctuation THEN quote THEN comma is missed
    by that shortcut. Validate a small punctuation-token list with the actual
    parser, without scanning the full vocabulary at every generation step.
    """
    def __init__(self,base,data):
        self.base=base;self.token_enforcer=base.token_enforcer;self.data=data

    def __call__(self,batch_id,sent):
        allowed=self.base(batch_id,sent)
        state=self.token_enforcer.prefix_states.get(tuple(sent.tolist()))
        key=state.parser.shortcut_key() if state else None
        if not isinstance(key,tuple) or key[0]!="json_freetext":return allowed
        additional=[]
        for token,text in self.data.closing_boundary_tokens:
            parser=state.parser
            try:
                valid=True
                for char in text:
                    where=parser.shortcut_key()
                    if isinstance(where,tuple) and where[0]=="json_freetext" and char!='"':
                        if self.data.prose_filter=="korean" and char not in PROSE_ALPHABET:
                            valid=False;break
                        if self.data.forbid_numeric_prose and (char.isdecimal() or char=="\\"):
                            valid=False;break
                    if char not in parser.get_allowed_characters():valid=False;break
                    parser=parser.add_character(char)
                if valid:additional.append(token)
            except (ValueError,KeyError,IndexError):pass
        return allowed+additional if additional else allowed


def build_decoder(tokenizer_data, evidence_ids):
    from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    schema = dict(type="object", properties={
        "judgment": dict(type="string", enum=["high_similarity", "review", "low_similarity", "insufficient_evidence"]),
        "evidence_ids": dict(type="array", items=dict(type="string", enum=evidence_ids or ["unused"]),
                             maxItems=len(evidence_ids), uniqueItems=True),
        "explanation": dict(type="string", minLength=1, maxLength=400)},
        required=["judgment", "evidence_ids", "explanation"], additionalProperties=False)
    return build_schema_decoder(tokenizer_data,schema,"korean")


class ExplanationNoRepeat:
    """Block repeated token phrases only inside the explanation free string.

    The JSON header and repeated digits in E001/E002 remain unrestricted.
    The prefix callback runs first in Transformers' logits processor chain.
    """
    def __init__(self, prefix, prompt_length, ngram_size=4):
        from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor
        self.prefix = prefix
        self.prompt_length = prompt_length
        self.blocker = NoRepeatNGramLogitsProcessor(ngram_size)

    def __call__(self, input_ids, scores):
        state = self.prefix.token_enforcer.prefix_states.get(tuple(input_ids[0].tolist()))
        key = state.parser.shortcut_key() if state is not None else None
        if isinstance(key, tuple) and key[0] == "json_freetext":
            return self.blocker(input_ids[:, self.prompt_length:], scores)
        return scores
