"""Disease content (why it happens, what to do), loaded from app/content/diseases.json.

Every text in the file is {"en": ..., "bn": ...}. `localize` picks one language, so the API
never returns a half-translated block.
"""

import json
from pathlib import Path

LANGS = ("en", "bn")


class ContentError(ValueError):
    """The content file is missing something the model or the pages rely on."""


def localize(value, lang: str):
    """Replace every {"en","bn"} pair inside `value` by the text in `lang`."""
    if isinstance(value, dict):
        if set(value) == set(LANGS):
            return value[lang]
        return {key: localize(item, lang) for key, item in value.items()}
    if isinstance(value, list):
        return [localize(item, lang) for item in value]
    return value


def _texts(value, path=""):
    """Yield (path, pair) for every language pair found in `value`."""
    if isinstance(value, dict):
        if set(value) == set(LANGS):
            yield path, value
            return
        for key, item in value.items():
            yield from _texts(item, f"{path}/{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _texts(item, f"{path}[{i}]")


class Content:
    def __init__(self, doc: dict):
        self.doc = doc

    @classmethod
    def load(cls, path: Path) -> "Content":
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ContentError(f"content file not found: {path}") from None
        except json.JSONDecodeError as error:
            raise ContentError(f"content file is not valid JSON: {error}") from None
        return cls(doc)

    def validate(self, class_ids) -> None:
        """Raise ContentError unless every model class has complete, sourced text."""
        problems = []
        diseases = self.doc.get("diseases", {})
        for class_id in class_ids:
            if class_id not in diseases:
                problems.append(f"no entry for class {class_id}")
        for class_id, entry in diseases.items():
            if entry.get("kind") not in ("disease", "healthy"):
                problems.append(f"{class_id}: kind must be 'disease' or 'healthy'")
            for path, pair in _texts(entry, class_id):
                if not all(isinstance(pair[lang], str) and pair[lang].strip() for lang in LANGS):
                    problems.append(f"{path}: empty text")
            if entry.get("kind") == "disease":
                if not entry.get("sources"):
                    problems.append(f"{class_id}: no sources")
                for source in entry.get("sources", []):
                    if not str(source.get("url", "")).startswith("https://"):
                        problems.append(f"{class_id}: source url is not https")
        for key in ("disclaimer", "chemical_note", "unsure", "helpline", "crops"):
            if key not in self.doc:
                problems.append(f"missing top-level '{key}'")
        if problems:
            raise ContentError("; ".join(problems[:10]))

    def name(self, class_id: str, lang: str) -> str | None:
        entry = self.doc.get("diseases", {}).get(class_id)
        return entry["name"][lang] if entry else None

    def disease(self, class_id: str, lang: str) -> dict | None:
        """The full page for one class, in one language, with the shared notes attached."""
        entry = self.doc.get("diseases", {}).get(class_id)
        if entry is None:
            return None
        page = localize(entry, lang)
        page["class_id"] = class_id
        page["disclaimer"] = self.doc["disclaimer"][lang]
        if entry["kind"] == "disease":
            page["chemical_note"] = self.doc["chemical_note"][lang]
            page["helpline"] = self._helpline(lang)
        return page

    def classes(self, class_ids, lang: str) -> list[dict]:
        """[{class_id, crop, name}] for the model's classes, e.g. for a "what was it?" menu."""
        return [
            {"class_id": cid, "crop": cid.split("_")[0], "name": self.name(cid, lang)}
            for cid in class_ids
        ]

    def unsure(self, lang: str) -> dict:
        block = localize(self.doc["unsure"], lang)
        block["kind"] = "unsure"
        block["disclaimer"] = self.doc["disclaimer"][lang]
        return block

    def _helpline(self, lang: str) -> dict:
        helpline = self.doc["helpline"]
        return {"text": helpline["text"][lang], "verified": bool(helpline["verified"])}
