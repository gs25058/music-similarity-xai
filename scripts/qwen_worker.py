"""Fixed Qwen3.5-0.8B, isolated environment; JSON stdin/stdout.

Official HF AutoProcessor/AutoModel loading + chat-template non-thinking API.
Attribution and fixed revision: THIRD_PARTY_NOTICES.md, configs/model_versions.json.
"""
import contextlib
import json
import os
from pathlib import Path
import sys

PROMPT = """두 곡의 음악적 유사성을 추가 검토하는 음악 분석가입니다.
입력 evidence의 음정 진행, 리듬, 화성, 반주 근거를 종합하세요. 곡의 출처나 표절을 단정하지 마세요.
cosine은 유사도 점수이며 확률이 아닙니다. unavailable은 해당 정보가 없다는 뜻입니다.
structure_context는 구간 맥락일 뿐입니다. 구조 정보가 없어도 다른 음악 요소로 판단할 수 있습니다.
같은 후렴이라는 이유로 점수를 더하지 마세요. timbre는 반주가 아닙니다. predictions는 검증된 정답이 아닙니다.

반드시 JSON 하나만 출력하세요. judgment, evidence_ids, explanation 세 필드 모두 필수입니다.
judgment는 high_similarity, review, low_similarity, insufficient_evidence 중 정확히 하나만 선택하세요.
여러 요소가 높으면 high_similarity, 부분적이면 review, 뚜렷한 대응이 없으면 low_similarity,
판단할 음악 근거 자체가 부족하면 insufficient_evidence를 선택할 수 있습니다.
근거 ID는 evidence_ids 필드에만 쓰고, 입력에 있는 ID만 사용하세요.
explanation은 입력의 음악적 근거를 한국어 2~3문장, 400자 이내로 요약하세요.
explanation에는 숫자를 쓰지 마세요. 점수와 시간은 프로그램이 원본 근거에서 표시합니다.
높음, 부분적 일치 등 한국어로 설명하고 지시문을 반복하지 마세요.

올바른 출력 형식의 예시 (판단은 실제 근거에 따라 선택):
{"judgment":"review","evidence_ids":["E001"],"explanation":"대응 구간의 일부 음악 요소에서 유사성이 관찰됩니다. 누락된 근거와 요소별 차이를 추가 검토해야 합니다."}
"""


def main():
    request = json.load(sys.stdin)
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface_qwen").resolve()))
    spec = json.loads(Path("configs/model_versions.json").read_text())["qwen"]
    path = Path(spec["local_dir"]).resolve()
    if not (path/"config.json").exists():
        raise FileNotFoundError("Run scripts/prepare_models.py qwen first")
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # lm-format-enforcer 0.11.3 imports a tokenizer class from its v4 path.
        # v5 exports the same public class from tokenization_utils_base.
        import transformers.tokenization_utils as tokenizer_compat
        if not hasattr(tokenizer_compat, "PreTrainedTokenizerBase"):
            tokenizer_compat.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
        from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        torch.set_num_threads(2)
        processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        device = request.get("device", "cpu")
        model = AutoModelForImageTextToText.from_pretrained(str(path), local_files_only=True,
                    dtype=torch.float32 if device == "cpu" else torch.bfloat16).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        messages = [{"role": "system", "content": [{"type": "text", "text": PROMPT}]},
                    {"role": "user", "content": [{"type": "text", "text": json.dumps(request, ensure_ascii=False)}]}]
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.judge import validate_judgment
        ids = [e["evidence_id"] for e in request["evidence"]]
        schema = dict(type="object", properties={
            "judgment": dict(type="string", enum=["high_similarity", "review", "low_similarity", "insufficient_evidence"]),
            "evidence_ids": dict(type="array", items=dict(type="string", enum=ids or ["unused"]), maxItems=len(ids), uniqueItems=True),
            # A looping regex shares parser/token caches across string lengths.
            # Expanding a {1,400} bound creates hundreds of expensive states.
            "explanation": dict(type="string", pattern="[^0-9]*")},
            required=["judgment", "evidence_ids", "explanation"], additionalProperties=False)
        parser_config = CharacterLevelParserConfig()
        parser_config.alphabet += "".join(chr(i) for i in range(0xAC00, 0xD7A4))
        attempts = []
        for attempt in range(2):
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt", enable_thinking=False).to(device)
            prefix = build_transformers_prefix_allowed_tokens_fn(processor.tokenizer,
                        JsonSchemaParser(schema, config=parser_config))
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                        prefix_allowed_tokens_fn=prefix)
            text = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            # Strip an enclosing code fence only; never repair IDs or invent output.
            if text.startswith("```json") and text.endswith("```"):
                text = text[7:-3].strip()
            try:
                result = validate_judgment(json.loads(text), request["evidence"])
                attempts.append(dict(valid=True, raw_output=text))
                break
            except ValueError as error:
                attempts.append(dict(valid=False, error=str(error), raw_output=text))
                print("INVALID_MODEL_OUTPUT: " + text, file=sys.stderr)
                if attempt == 1:
                    raise
                messages.extend([{"role": "assistant", "content": [{"type": "text", "text": text}]},
                                 {"role": "user", "content": [{"type": "text", "text":
                                  "스키마 검증 오류: "+str(error)+". 원본 근거를 그대로 사용하고 judgment, evidence_ids, 한국어 explanation 세 필드를 모두 포함한 JSON을 다시 출력하세요."}]}])
    print(json.dumps(dict(result=result, metadata={**spec, "thinking": False, "weights_frozen": True,
                                                  "transformers": transformers.__version__, "decode": "greedy",
                                                  "max_new_tokens": 1024,
                                                  "format_enforcer": "lm-format-enforcer==0.11.3",
                                                  "first_pass_valid": attempts[0]["valid"], "attempts": attempts}), ensure_ascii=False))


if __name__ == "__main__":
    main()
