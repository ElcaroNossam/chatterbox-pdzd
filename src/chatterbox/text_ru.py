"""Russian text normalization for TTS (chatterbox-pdzd).

Turns raw text into something the model can pronounce:
- strips our lyric-sheet markup: [Section] headers, (stage directions), | breath marks, leading line numbers
- expands abbreviations (т.е., т.к., и т.д., км, руб. ...)
- spells out numbers: plain cardinals, ordinals like 5-й / 5-я / 5-е / 5-го, percents, years
- keeps manual stress marks written as "+" before a vowel (м+ука), they are handled later by the stresser
"""
import re

try:
    from num2words import num2words
except ImportError:  # pragma: no cover
    num2words = None

ABBREVIATIONS = [
    (r"\bт\.\s?е\.", "то есть"),
    (r"\bт\.\s?к\.", "так как"),
    (r"\bи\s?т\.\s?д\.", "и так далее"),
    (r"\bи\s?т\.\s?п\.", "и тому подобное"),
    (r"\bт\.\s?п\.", "тому подобное"),
    (r"\bв\s?т\.\s?ч\.", "в том числе"),
    (r"\bдр\.", "другие"),
    (r"\bгг\.", "годы"),
    (r"\bул\.", "улица"),
    (r"\bд\.", "дом"),
    (r"\bкв\.", "квартира"),
]

# ordinal suffix -> (num2words gender/case hint, ending to keep)
ORDINAL_SUFFIXES = {
    "й": "masculine",
    "ый": "masculine",
    "ой": "masculine",
    "ий": "masculine",
    "я": "feminine",
    "ая": "feminine",
    "е": "neuter",
    "ое": "neuter",
}

# number + unit: agree the noun with the number (1 километр, 3 километра, 5 километров)
UNITS = {
    r"км": ("километр", "километра", "километров"),
    r"кг": ("килограмм", "килограмма", "килограммов"),
    r"мин\.?": ("минута", "минуты", "минут"),
    r"сек\.?": ("секунда", "секунды", "секунд"),
    r"%": ("процент", "процента", "процентов"),
    r"₽": ("рубль", "рубля", "рублей"),
    r"руб\.?": ("рубль", "рубля", "рублей"),
    r"коп\.?": ("копейка", "копейки", "копеек"),
    r"\$": ("доллар", "доллара", "долларов"),
}
FEMININE_UNITS = {"минута", "секунда", "копейка"}


def ru_plural(n: int, forms) -> str:
    n = abs(n) % 100
    if 11 <= n <= 19:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _cardinal_gender(n: int, feminine: bool) -> str:
    words = _cardinal(n)
    if feminine:  # одна минута, две секунды
        words = __import__("re").sub(r"\bодин$", "одна", words)
        words = __import__("re").sub(r"\bдва$", "две", words)
    return words


def _cardinal(n: int) -> str:
    if num2words is None:
        return str(n)
    return num2words(n, lang="ru")


def _ordinal(n: int, suffix: str) -> str:
    if num2words is None:
        return f"{n}-{suffix}"
    gender = ORDINAL_SUFFIXES.get(suffix)
    if gender:
        try:
            return num2words(n, lang="ru", to="ordinal", gender=gender)
        except TypeError:
            return num2words(n, lang="ru", to="ordinal")
    # oblique cases (5-го, 5-му, 5-м, 5-х): build from the masculine form
    base = num2words(n, lang="ru", to="ordinal")  # e.g. "пятый"
    stem = re.sub(r"(ый|ий|ой)$", "", base)
    soft = base.endswith("ий") and not base.endswith("кий")
    endings = {
        "го": "его" if soft else "ого",
        "му": "ему" if soft else "ому",
        "м": "им" if soft else "ым",
        "х": "их" if soft else "ых",
    }
    return stem + endings.get(suffix, base[len(stem):])


def strip_markup(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\[[^\]]*\].*$", "", line)          # [Куплет 1] 16 тактов ...
        line = re.sub(r"\([^)]*\)", "", line)                    # (ровно), (тише)
        line = re.sub(r"^\s*\d{1,3}\s+(?=\D)", "", line)        # leading line numbers "01 "
        line = line.replace("|", " ")
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def normalize_ru(text: str) -> str:
    text = strip_markup(text)
    for pattern, repl in ABBREVIATIONS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    # years: "2026 г." / "в 2026 году" -> ordinal + год
    text = re.sub(r"(\d{3,4})\s?г\.", lambda m: _ordinal(int(m.group(1)), "й") + " год", text)
    text = re.sub(r"\bг\.", "год", text)
    # number + unit with agreement
    for unit_re, forms in UNITS.items():
        text = re.sub(
            r"(\d+)\s?" + unit_re + r"(?![а-яА-Я])",
            lambda m, f=forms: f"{_cardinal_gender(int(m.group(1)), f[0] in FEMININE_UNITS)} {ru_plural(int(m.group(1)), f)}",
            text,
        )
    # ordinals: 5-й, 5-я, 5-е, 5-го, 5-му, 5-м, 5-х (also with long dash/no dash variants)
    text = re.sub(
        r"(\d+)\s?[-–]\s?(ый|ий|ой|ая|ое|й|я|е|го|му|м|х)\b",
        lambda m: _ordinal(int(m.group(1)), m.group(2)),
        text,
    )
    # decimals: 2,5 -> "две целых пять" is rarely wanted in lyrics/voiceover; read as "два и пять"
    text = re.sub(r"(\d+)[.,](\d+)", lambda m: f"{_cardinal(int(m.group(1)))} и {_cardinal(int(m.group(2)))}", text)
    # plain numbers (incl. years)
    text = re.sub(r"\d+", lambda m: _cardinal(int(m.group(0))), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
