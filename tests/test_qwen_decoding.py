"""Optional real-tokenizer regression tests; no model weights or GPU needed.

Run with .venv-qwen/bin/python -m unittest discover -s tests -p test_qwen_decoding.py
"""
import importlib.util
import json
from pathlib import Path
import time
import unittest

AVAILABLE = importlib.util.find_spec("lmformatenforcer") is not None and Path("models/qwen/tokenizer.json").exists()


@unittest.skipUnless(AVAILABLE, "Qwen tokenizer and isolated decoder dependencies not installed")
class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoTokenizer
        from scripts.qwen_decoding import build_tokenizer_data
        torch.set_num_threads(2)
        cls.torch = torch
        cls.tokenizer = AutoTokenizer.from_pretrained("models/qwen", local_files_only=True)
        cls.data = build_tokenizer_data(cls.tokenizer)

    def accepts(self, text, evidence_ids=None):
        from scripts.qwen_decoding import build_decoder, ExplanationNoRepeat
        prefix = build_decoder(self.data, evidence_ids or ["E001"])
        sequence = self.tokenizer.encode("test prompt", add_special_tokens=False)
        no_repeat = ExplanationNoRepeat(prefix, len(sequence))
        for token in self.tokenizer.encode(text, add_special_tokens=False):
            if token not in prefix(0, self.torch.tensor(sequence)):
                return False
            scores = self.torch.zeros((1, len(self.tokenizer)))
            if not self.torch.isfinite(no_repeat(self.torch.tensor([sequence]), scores)[0, token]):
                return False
            sequence.append(token)
        return self.tokenizer.eos_token_id in prefix(0, self.torch.tensor(sequence))

    def test_korean_and_numeric_evidence_id_remain_valid(self):
        text = json.dumps(dict(judgment="review", evidence_ids=["E001"], explanation="음악 요소가 부분적으로 유사합니다."), ensure_ascii=False)
        started = time.perf_counter()
        self.assertTrue(self.accepts(text))
        self.assertLess(time.perf_counter()-started, 5., "free text must use the token cache")
        text = json.dumps(dict(judgment="review", evidence_ids=["E001", "E002", "E003"], explanation="음정과 리듬이 부분적으로 유사합니다."), ensure_ascii=False)
        self.assertTrue(self.accepts(text, ["E001", "E002", "E003"]))

    def test_numbers_and_escaped_numbers_are_rejected(self):
        for explanation in ("12초가 유사합니다.", "유사도가 0.99입니다."):
            with self.subTest(explanation=explanation):
                text = json.dumps(dict(judgment="review", evidence_ids=["E001"], explanation=explanation), ensure_ascii=False)
                self.assertFalse(self.accepts(text))
        self.assertFalse(self.accepts('{"judgment":"review","evidence_ids":["E001"],"explanation":"\\u0031초입니다."}'))

    def test_unknown_id_is_rejected(self):
        self.assertFalse(self.accepts('{"judgment":"review","evidence_ids":["E999"],"explanation":"유사합니다."}'))

    def test_partial_utf8_cannot_bypass_prose_character_filter(self):
        from scripts.qwen_decoding import PROSE_ALPHABET
        partial = {token for token, text, _ in self.data.regular_tokens if "�" in text}
        self.assertTrue(partial)
        allowed = set(self.data.tokenizer_tree.json_freetext_tokens.lookup_allowed_tokens(0, 400).allowed_tokens)
        self.assertFalse(partial & allowed)
        permitted = set(PROSE_ALPHABET) - set("0123456789\\")
        for token, text, _ in self.data.regular_tokens:
            if token in allowed:
                self.assertTrue(set(text) <= permitted)

    def test_degenerate_repetition_and_cjk_are_blocked(self):
        for text in ("음악 요소가 유사합니다. "*8, "두 곡은 和 和 和 和 和 로 유사합니다.", "주어진 증거들 (E Ⅰ"):
            self.assertFalse(self.accepts(json.dumps(dict(judgment="review", evidence_ids=["E001"], explanation=text), ensure_ascii=False)))


@unittest.skipUnless(AVAILABLE,"Qwen tokenizer and isolated decoder dependencies not installed")
class ClaimDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoTokenizer
        from scripts.qwen_decoding import build_tokenizer_data
        torch.set_num_threads(2);cls.torch=torch
        cls.tokenizer=AutoTokenizer.from_pretrained("models/qwen",local_files_only=True)
        cls.korean=build_tokenizer_data(cls.tokenizer,"korean",False)
        cls.unrestricted=build_tokenizer_data(cls.tokenizer,"none",False)

    def accepts(self,value,mode="korean",payload=None):
        from src.claims import response_schema
        from scripts.qwen_decoding import build_schema_decoder
        payload=payload if payload is not None else dict(evidence=[dict(evidence_id="E001")])
        prefix=build_schema_decoder(self.korean if mode=="korean" else self.unrestricted,response_schema(payload),mode)
        sequence=self.tokenizer.encode("test prompt",add_special_tokens=False)
        for token in self.tokenizer.encode(json.dumps(value,ensure_ascii=False),add_special_tokens=False):
            if token not in prefix(0,self.torch.tensor(sequence)):
                self.last_failure=dict(token=token,decoded_token=self.tokenizer.decode([token]),prefix=self.tokenizer.decode(sequence)[-180:])
                return False
            sequence.append(token)
        return self.tokenizer.eos_token_id in prefix(0,self.torch.tensor(sequence))

    def example(self):
        return dict(claims=[dict(claim_id="C001",evidence_ids=["E001"],factor="rhythm",role="supports",predicate="observed_value",
                    statement="리듬의 원시 코사인은 0.95입니다.",assertion=dict(quantity="raw_cosine",value=.95,operator="eq")),
                    dict(claim_id="C002",evidence_ids=["E001"],factor="rhythm",role="limits",predicate="uncalibrated_score",
                         statement="리듬·화성의 ‘보정’과 “검증” 범위를 구분합니다.")],verdict="review",explanation="주장 C001과 C002를 함께 검토해야 합니다.")

    def test_korean_punctuation_numeric_assertion_and_repeated_ids(self):
        self.assertTrue(self.accepts(self.example()),getattr(self,"last_failure",None))

    def test_unknown_evidence_id_and_empty_evidence_are_enforced(self):
        value=self.example();value["claims"][0]["evidence_ids"]=["E999"]
        self.assertFalse(self.accepts(value))
        self.assertFalse(self.accepts(self.example(),payload=dict(evidence=[])))
        self.assertTrue(self.accepts(dict(claims=[],verdict="insufficient_evidence",explanation="근거가 부족합니다."),payload=dict(evidence=[])))

    def test_unrestricted_utf8_mode_accepts_partial_emoji_tokens(self):
        value=self.example();value["explanation"]="한국어 설명과 음악 🎵 기호입니다."
        self.assertTrue(self.accepts(value,"none"),getattr(self,"last_failure",None))
        self.assertFalse(self.accepts(value,"korean"))

    def test_generation_stop_uses_chat_turn_end(self):
        from types import SimpleNamespace
        from scripts.qwen_runtime import generation_stop_args
        stop=generation_stop_args(SimpleNamespace(tokenizer=self.tokenizer))
        self.assertEqual(stop["eos_token_id"],self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.assertNotEqual(stop["eos_token_id"],stop["pad_token_id"])
