"""Apply chatterbox-pdzd fixes to the vendored chatterbox 0.1.7 sources. Idempotent."""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1]) / "src" / "chatterbox"


def _rstrip_lines(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n"))


def patch(path: pathlib.Path, old: str, new: str, marker: str):
    # trailing whitespace in the upstream sources is inconsistent; compare without it
    s = _rstrip_lines(path.read_text(encoding="utf-8"))
    old, new = _rstrip_lines(old), _rstrip_lines(new)
    if marker in s:
        print(f"skip (already patched): {path.name} [{marker}]")
        return
    if old not in s:
        sys.exit(f"pattern not found in {path}:\n{old[:200]}")
    path.write_text(s.replace(old, new, 1), encoding="utf-8")
    print(f"patched: {path.name} [{marker}]")


# 1) Repetition detector: the original comment says "3x same token in a row", but the code
#    checked only the last 2 tokens and the `self.complete` guard was commented out, so any
#    two identical speech tokens mid-sentence forced EOS and cut the phrase short.
import re as _re
_an = ROOT / "models/t3/inference/alignment_stream_analyzer.py"
_s = _an.read_text(encoding="utf-8")
if "pdzd-fix: original checked only the last 2 tokens" in _s:
    print("skip (already patched): alignment_stream_analyzer.py [repetition]")
else:
    _pat = _re.compile(
        r"        token_repetition = \(\n\s*# self\.complete and\s*\n\s*len\(self\.generated_tokens\) >= 3 and\s*\n"
        r"\s*len\(set\(self\.generated_tokens\[-2:\]\)\) == 1\s*\n\s*\)")
    _new = (
        "        # pdzd-fix: original checked only the last 2 tokens with the completion guard\n"
        "        # commented out, cutting phrases mid-word. Now: after the text is complete, 3 identical\n"
        "        # tokens in a row end generation; before that, only a real stuck loop (8 identical) does.\n"
        "        token_repetition = (\n"
        "            (self.complete and len(self.generated_tokens) >= 3\n"
        "             and len(set(self.generated_tokens[-3:])) == 1)\n"
        "            or (len(self.generated_tokens) >= 8 and len(set(self.generated_tokens[-8:])) == 1)\n"
        "        )")
    _s2, _n = _pat.subn(_new, _s, count=1)
    if _n != 1:
        sys.exit("repetition pattern not found")
    _an.write_text(_s2, encoding="utf-8")
    print("patched: alignment_stream_analyzer.py [repetition]")
patch(
    ROOT / "models/t3/inference/alignment_stream_analyzer.py",
    """            logger.warning(f"🚨 Detected 2x repetition of token {repeated_token}")""",
    """            logger.warning(f"🚨 Detected repetition of token {repeated_token}")""",
    "Detected repetition of token",
)

# 2) Russian stress: the original optional dependency (russian_text_stresser, GPL-3.0) is not on
#    PyPI and conflicts with gradio's deps, so stress was silently skipped. Use RUAccent
#    (Apache-2.0, homograph-aware). Manual "+" marks before a vowel are respected.
NEW_STRESS_FUNC = '''_RU_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def _plus_to_acute(text: str) -> str:
    """'м+ука' -> 'му' + U+0301 + 'ка' (the model vocab uses the combining acute accent)."""
    import re
    return re.sub(r"\\+([" + _RU_VOWELS + r"])", lambda m: m.group(1) + "\\u0301", text)


def add_russian_stress(text: str) -> str:
    """pdzd-fix: Russian stress via RUAccent (Apache-2.0). Manual '+' marks win."""
    global _russian_stresser
    if "+" in text:
        return _plus_to_acute(text)
    try:
        if _russian_stresser is None:
            from ruaccent import RUAccent
            _russian_stresser = RUAccent()
            _russian_stresser.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
        return _plus_to_acute(_russian_stresser.process_all(text))
    except ImportError:
        logger.warning("ruaccent not installed - Russian stress labeling skipped (pip install ruaccent)")
        return text
    except Exception as e:
        logger.warning(f"Russian stress labeling failed: {e}")
        return text'''
tok = ROOT / "models/tokenizers/tokenizer.py"
_t = tok.read_text(encoding="utf-8")
if "pdzd-fix: Russian stress via RUAccent" in _t:
    print("skip (already patched): tokenizer.py [stress]")
else:
    _a = _t.index("def add_russian_stress(")
    _b = _t.index("\nclass MTLTokenizer", _a)
    tok.write_text(_t[:_a] + NEW_STRESS_FUNC + "\n\n" + _t[_b:], encoding="utf-8")
    print("patched: tokenizer.py [stress]")
# stress must be added on the readable text, before lowercase + NFKD decomposition (й/ё split)
patch(
    tok,
    """        txt = self.preprocess_text(txt, language_id=language_id, lowercase=lowercase, nfkd_normalize=nfkd_normalize)

        # Language-specific text processing""",
    """        if language_id == 'ru':  # pdzd-fix: stress before NFKD so the stresser sees normal words
            txt = add_russian_stress(txt)
        txt = self.preprocess_text(txt, language_id=language_id, lowercase=lowercase, nfkd_normalize=nfkd_normalize)

        # Language-specific text processing""",
    "pdzd-fix: stress before NFKD",
)
# drop the original post-NFKD Russian branch so stress is not applied twice
_s = tok.read_text(encoding="utf-8")
_old = "        elif language_id == 'ru':\n            txt = add_russian_stress(txt)\n"
if _old in _s:
    tok.write_text(_s.replace(_old, "", 1), encoding="utf-8")
    print("patched: tokenizer.py [removed post-NFKD ru branch]")
else:
    print("skip: old ru branch already removed")

# 3) mtl_tts.generate: cache the voice profile, Russian text normalizer, configurable speed knobs
mtl = ROOT / "mtl_tts.py"
patch(
    mtl,
    """        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):""",
    """        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
        max_new_tokens=1000,
        n_cfm_timesteps=None,
    ):
        # pdzd-fix: max_new_tokens / n_cfm_timesteps exposed (were hardcoded; fewer CFM steps = faster on CPU)""",
    "pdzd-fix: max_new_tokens / n_cfm_timesteps exposed",
)
patch(
    mtl,
    """        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:""",
    """        if audio_prompt_path:
            # pdzd-fix: the original rebuilt the voice profile (3 encoders) on every call;
            # only rebuild when the reference file actually changes.
            import os
            src = (os.path.abspath(audio_prompt_path), os.path.getmtime(audio_prompt_path))
            if getattr(self, "_pdzd_cond_src", None) != src or self.conds is None:
                self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
                self._pdzd_cond_src = src
        else:""",
    "pdzd-fix: the original rebuilt the voice profile",
)
patch(
    mtl,
    """        # Norm and tokenize text
        text = punc_norm(text)""",
    """        # Norm and tokenize text
        if language_id and language_id.lower() == "ru":  # pdzd-fix: numbers, abbreviations, lyric markup
            from .text_ru import normalize_ru
            text = normalize_ru(text)
        text = punc_norm(text)""",
    "pdzd-fix: numbers, abbreviations, lyric markup",
)
patch(
    mtl,
    """                max_new_tokens=1000,  # TODO: use the value in config""",
    """                max_new_tokens=max_new_tokens,""",
    "max_new_tokens=max_new_tokens,",
)
patch(
    mtl,
    """            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )""",
    """            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
                n_cfm_timesteps=n_cfm_timesteps,
            )""",
    "n_cfm_timesteps=n_cfm_timesteps,\n            )",
)
print("all patches applied")
