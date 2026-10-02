"""English instruction -> plan: the stage C front end behind `r1_mission_cli.py ask`.

Stdlib only and Python 3.8 clean, like the rest of the package: the robot runs foxy
on Ubuntu 20.04 and its apt is broken.

The split this module enforces: the model TRANSCRIBES, the robot JUDGES.

  1. normalize()   deterministic. Number words and units become digits and meters,
                   seconds, degrees, m/s, rad/s before the model sees anything:
                   small models are bad at arithmetic, and "10 feet" should be 3.05 m
                   because a regex said so, not because a 1.5B model guessed.
  2. the model     writes down what was asked, step by step, in a small JSON
                   vocabulary that can also express what the robot CANNOT do (walk
                   backward, strafe, "climb the stairs"). It is not told the robot's
                   limits, so it has no reason to bend a request until it fits them.
                   That is the failure that matters: "walk backward 2 m" quietly
                   becoming "walk 2 m" passes every envelope check.
  3. translate()   deterministic. Decides what is refused and why, from the same
                   deployed envelope and limits every other path uses, and hands the
                   rest to plan_from_json -> compile_plan unchanged. All or nothing:
                   if any step is refused, nothing runs.
  4. render_*()    deterministic English. The reply is never the model's prose. A
                   report mixes a measured angle (IMU) with an estimated distance (time
                   x commanded speed), and a paraphrase turns "about 2 m" into "2 m".
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import time
import urllib.error
import urllib.request

from . import plan as P

# ----------------------------------------------------------------- normalize
_SMALL = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_JOIN = re.compile(r"[ \t-]+")

NUM = r"(\d+(?:\.\d+)?|\.\d+)"
# Unit words an article or a fraction can stand in front of ("a meter", "half a
# minute", "2 feet and a half").
_UNIT_WORD = (r"(met(?:er|re)s?|m|f(?:ee|oo)t|ft|yards?|yds?|inch(?:es)?|"
              r"centimet(?:er|re)s?|cm|kilomet(?:er|re)s?|km|miles?|seconds?|secs?|s|"
              r"minutes?|mins?|hours?|hrs?|degrees?|degs?)")
_ARTICLE_UNIT = (r"(met(?:er|re)|foot|yard|inch|centimet(?:er|re)|kilomet(?:er|re)|mile|"
                 r"second|minute|hour|degree)")

# (unit pattern, factor, canonical unit). Speeds first: they contain length and time
# words. Each pattern is followed by (?![a-z/]) so "m" never eats "min" or "m/s".
_CONV = [
    (r"(?:m/s|mps|met(?:er|re)s?\s+(?:per|a)\s+second)", 1.0, "m/s"),
    (r"(?:km/h|kmh|kph|kilomet(?:er|re)s?\s+(?:per|an)\s+hour)", 1.0 / 3.6, "m/s"),
    (r"(?:mph|miles?\s+(?:per|an)\s+hour)", 0.44704, "m/s"),
    (r"(?:ft/s|f(?:ee|oo)t\s+(?:per|a)\s+second)", 0.3048, "m/s"),
    (r"(?:rad/s|radians?\s+(?:per|a)\s+second)", 1.0, "rad/s"),
    (r"(?:deg/s|°/s|degrees?\s+(?:per|a)\s+second)", math.pi / 180.0, "rad/s"),
    (r"(?:°|degs?|degrees?)", 1.0, "degrees"),
    (r"(?:rads?|radians?)", 180.0 / math.pi, "degrees"),
    (r"(?:km|kilomet(?:er|re)s?)", 1000.0, "m"),
    (r"(?:cm|centimet(?:er|re)s?)", 0.01, "m"),
    (r"(?:mm|millimet(?:er|re)s?)", 0.001, "m"),
    (r"(?:m|met(?:er|re)s?)", 1.0, "m"),
    (r"(?:ft|feet|foot)", 0.3048, "m"),
    (r"(?:yds?|yards?)", 0.9144, "m"),
    (r"(?:inch(?:es)?)", 0.0254, "m"),
    (r"(?:mi|miles?)", 1609.344, "m"),
    (r"(?:h|hrs?|hours?)", 3600.0, "seconds"),
    (r"(?:mins?|minutes?)", 60.0, "seconds"),
    (r"(?:ms|millisec(?:ond)?s?)", 0.001, "seconds"),
    (r"(?:s|secs?|seconds?)", 1.0, "seconds"),
]
_CONV = [(re.compile(r"(?<![\w.])" + NUM + r"\s*" + pat + r"(?![a-z/])"), f, unit)
         for pat, f, unit in _CONV]

# After number words have become digits, so "three quarters" is "3 quarters" here.
_FRACTIONS = (("a quarter", 0.25), ("1 quarter", 0.25), ("3 quarters", 0.75),
              ("a third", 1.0 / 3.0), ("1 third", 1.0 / 3.0), ("2 thirds", 2.0 / 3.0),
              ("half", 0.5), ("1 half", 0.5))

# "N times: <steps>" is expanded here, because every small model tested wrote the
# steps out once ("Twice: walk 2 m and turn right 180" -> one walk, one turn; eval
# 2026-09-30). Only this form, where the colon makes the scope explicit. "turn left
# twice" or "..., and do that 4 times" are left to the model, with a note.
_REPEAT = re.compile(r"^(?:please )?(?:(?:do|repeat) (?:this |that |the following )?|repeat )?"
                     r"(\d+) times\s*:\s*(.+?)\s*[.!]?$")
MAX_REPEAT = 10

# Words that make an amount vague. Their presence anywhere in a sentence switches
# off the default angle below, so "turn right, but only a little" is not 90 degrees.
_VAGUE = re.compile(r"\b(?:a bit|a little|a little bit|slightly|some more|a bit more|"
                    r"a little more|a few|further|farther|a tad|a touch|somewhat|"
                    r"a smidge|a couple of steps)\b")

# A bare stop never needs a model: it is the one instruction where latency matters.
FAST_STOP = re.compile(r"^\s*(?:please\s+)?(?:stop|halt|freeze)"
                       r"(?:\s+(?:now|moving|right\s+now|immediately|please))?[\s.!]*$")


def _fmt(v):
    """3.048 -> '3.05', 30.0 -> '30', 0.5 -> '0.5'."""
    s = "{:.2f}".format(float(v)).rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _read_number(parts, i):
    """Number words starting at parts[i] -> (value, index of the last word used),
    or (None, i). `parts` alternates separator, word, separator, ... (re.split with
    a capture group), so words sit at odd indices.

    Deliberately strict: "five two" is 5 then 2, not 7, and "and" only joins after
    hundred/thousand ("one hundred and eighty"), never "two and turn".
    """
    total, cur, prev, last, k = 0, 0, None, i, i
    while k < len(parts):
        w = parts[k]
        if k > i and not _JOIN.fullmatch(parts[k - 1]):
            break
        if k == i and w in ("a", "an"):
            cur, prev = 1, "a"
        elif w in _SMALL and prev in (None, "tens", "hundred", "thousand", "and") \
                and not (prev == "tens" and (_SMALL[w] >= 10 or _SMALL[w] == 0)):
            cur += _SMALL[w]
            prev = "small"
        elif w in _TENS and prev in (None, "hundred", "thousand", "and"):
            cur += _TENS[w]
            prev = "tens"
        elif w == "hundred" and prev in (None, "a", "small", "tens"):
            cur = (cur or 1) * 100
            prev = "hundred"
        elif w == "thousand" and prev in (None, "a", "small", "tens", "hundred"):
            total += (cur or 1) * 1000
            cur, prev = 0, "thousand"
        elif (w == "and" and prev in ("hundred", "thousand") and k + 2 < len(parts)
              and _JOIN.fullmatch(parts[k + 1])
              and (parts[k + 2] in _SMALL or parts[k + 2] in _TENS)):
            prev = "and"
        else:
            break
        if prev != "and":
            last = k
        k += 2
    if prev in (None, "a") or last < i:
        return None, i
    return total + cur, last


def _words_to_digits(text):
    parts = re.split(r"([a-z]+)", text)
    out, i = [parts[0]], 1
    while i < len(parts):
        w = parts[i]
        starts = (w in _SMALL or w in _TENS or w in ("hundred", "thousand") or
                  (w in ("a", "an") and i + 2 < len(parts)
                   and parts[i + 2] in ("hundred", "thousand")
                   and _JOIN.fullmatch(parts[i + 1])))
        val, j = _read_number(parts, i) if starts else (None, i)
        if val is None:
            out.append(w)
            out.append(parts[i + 1] if i + 1 < len(parts) else "")
            i += 2
            continue
        out.append(_fmt(val))
        out.append(parts[j + 1] if j + 1 < len(parts) else "")
        i = j + 2
    return "".join(out)


def normalize(text):
    """Instruction -> (normalized text, notes). Lower case, digits, and only m,
    seconds, degrees, m/s and rad/s. `notes` lists every unit conversion, e.g.
    "10 feet -> 3.05 m", so the operator can see what the number became."""
    t = re.sub(r"\s+", " ", str(text).strip().lower())
    t = t.replace("’", "'")
    t = _words_to_digits(t)
    t = re.sub(r"\btwice\b", "2 times", t)
    t = re.sub(r"\bthrice\b", "3 times", t)
    # Turns named as fractions of a circle, before the generic fractions below can
    # read "half a turn" as 0.5 of something. "a quarter turn" -> "a 90 degree turn".
    t = re.sub(r"\bquarter[ -]turn\b", "90 degree turn", t)
    t = re.sub(r"\bhalf a turn\b", "a 180 degree turn", t)
    t = re.sub(r"\bhalf[ -]turn\b", "180 degree turn", t)
    t = re.sub(r"\b(?:1|a) (?:full|complete|whole) (?:turn|circle|rotation|revolution)\b",
               "a 360 degree turn", t)
    t = re.sub(r"\b(?:full|complete|whole) (?:turn|circle|rotation|revolution)\b",
               "360 degree turn", t)
    t = re.sub(NUM + r" (?:full )?(?:turns|rotations|revolutions|circles)\b",
               lambda m: "{} degrees".format(_fmt(float(m.group(1)) * 360.0)), t)
    t = re.sub(r"\bu[ -]?turn\b", "180 degree turn", t)
    t = re.sub(r"\bturn around\b", "turn 180 degrees", t)
    t = re.sub(r"\babout[ -]face\b", "turn 180 degrees", t)
    # Fractions and articles in front of a unit. The order matters: "1 and a half
    # meters" must be read before "a half meters" can match inside it, and "half a
    # minute" before "a minute" can become "1 minute".
    for frac, val in _FRACTIONS:
        t = re.sub(r"\b" + frac + r" of (?:a|an|1) " + _UNIT_WORD + r"(?![a-z])",
                   lambda m, val=val: "{:.6g} {}".format(val, m.group(1)), t)
    t = re.sub(NUM + r" and a half " + _UNIT_WORD + r"(?![a-z])",
               lambda m: "{} {}".format(_fmt(float(m.group(1)) + 0.5), m.group(2)), t)
    t = re.sub(r"\bhalf (?:a|an) " + _UNIT_WORD + r"(?![a-z])", r"0.5 \1", t)
    t = re.sub(r"\ba half[ -]" + _UNIT_WORD + r"(?![a-z])", r"0.5 \1", t)
    t = re.sub(r"\ba couple of " + _UNIT_WORD + r"(?![a-z])", r"2 \1", t)
    t = re.sub(r"\b(?:a|an) " + _ARTICLE_UNIT + r"(?![a-z])", r"1 \1", t)
    t = re.sub(NUM + r" " + _UNIT_WORD + r" and a half\b",
               lambda m: "{} {}".format(_fmt(float(m.group(1)) + 0.5), m.group(2)), t)

    # "turn left" with no amount is a 90 degree turn -- decided here, visibly, and
    # only when nothing in the sentence is vague. Left to the model, the same rule
    # turned "turn right a little" into a 90 degree turn (dev set, 2026-09-30).
    if not _VAGUE.search(t):
        t = re.sub(r"(?<!degree )(?<!degrees )\b(turn|rotate|spin|pivot)(?: to the)? "
                   r"(left|right)\b(?!\s*(?:a\s+)?(?:\d|by\b|at\b|for\b|to\b))",
                   r"\1 \2 90 degrees", t)
        t = re.sub(r"\b(?:make|take|do) a (left|right)(?: turn)?\b(?!\s*\d)",
                   r"turn \1 90 degrees", t)

    notes = []
    for rx, factor, unit in _CONV:
        def conv(m, factor=factor, unit=unit):
            v = float(m.group(1)) * factor
            new = "{} {}".format(_fmt(v), unit)
            if abs(factor - 1.0) > 1e-12:
                notes.append("{} -> {}".format(m.group(0), new))
            return new
        t = rx.sub(conv, t)
    t = re.sub(r"\s+", " ", t).strip()
    m = _REPEAT.match(t)
    if m and 1 <= int(m.group(1)) <= MAX_REPEAT:
        n, body = int(m.group(1)), m.group(2).rstrip(" .!,")
        t = ", then ".join([body] * n) + "."
        notes.append("repeated {} times: {}".format(n, body))
    elif re.search(r"\b\d+ times\b|\brepeat", t):
        notes.append("asks for a repetition I did not expand: check that the plan repeats "
                     "what you want")
    return t, notes


# ------------------------------------------------------------ what the model sees
# The model is not told the robot's limits (see the module docstring). What it is
# told is how to write down a request -- including requests the robot will refuse.
SYSTEM = """You turn one instruction for a two-legged robot into JSON steps.

Write down what the instruction asks for, step by step, in order. Do not decide whether the robot can do it, and never change what was asked: the robot checks every step itself and refuses what it cannot do.

Step types:
- walk: direction is forward, backward, left or right. Add distance_m if a distance is given, or duration_s if a time is given, otherwise neither. Add speed_mps only if a speed in m/s is given. Add within_s only for a distance with a time limit, like "2 m in 5 seconds".
- turn: direction is left or right. Add angle_deg if an angle is given, or duration_s if a time is given, otherwise neither. Add rate_rad_s only if a turn rate in rad/s is given. Add within_s only for an angle with a time limit, like "90 degrees in 3 seconds".
- stand: stand still or wait. Add duration_s if a time is given.
- stop: stop moving.
- unsupported: anything that is not walking, turning, standing or stopping, such as jumping, climbing, sitting, picking something up, going to a place or an object, following someone, dancing or talking. Put the request in "what".

Rules:
- One step per action, in the order given. Write repeated actions out in full ("twice", "three times").
- Copy numbers exactly as written. Units are already m, seconds, degrees, m/s and rad/s. Never make up a number.
- A time belongs only to the action it is written with. "wait 2 seconds, then turn" gives the wait a duration, not the turn a time limit.
- A walk with no direction is forward. "back up", "reverse" and "walk backwards" are walk backward. "strafe", "sidestep", "slide" and "shuffle" to a side are walk left or right. "walk back" or "come back" after turning around means walking forward to return.
- A turn with no amount has no angle_deg. A turn with no direction is left.
- Vague amounts such as "a bit", "a little", "slightly", "some more", "further" or "a few steps" are not numbers: leave the amount out.
- Words such as "slowly" or "quickly" do not add a speed."""

# Worked examples, in normalized form, except the last: "twice:" is expanded by
# normalize() before any operator text reaches the model, but the example still
# teaches writing a repetition out, which the model must do for "... 2 times".
# None of them is in eval/*.jsonl: the test sets are held out from prompt design.
EXAMPLES = [
    ("walk forward 3 m, then turn right 90 degrees",
     {"steps": [{"op": "walk", "direction": "forward", "distance_m": 3},
                {"op": "turn", "direction": "right", "angle_deg": 90}]}),
    ("turn left 90 degrees, wait 4 seconds and walk for 6 seconds",
     {"steps": [{"op": "turn", "direction": "left", "angle_deg": 90},
                {"op": "stand", "duration_s": 4},
                {"op": "walk", "direction": "forward", "duration_s": 6}]}),
    ("back up 2 m and then slide to the left a bit",
     {"steps": [{"op": "walk", "direction": "backward", "distance_m": 2},
                {"op": "walk", "direction": "left"}]}),
    ("go ahead some more and turn right slightly",
     {"steps": [{"op": "walk", "direction": "forward"},
                {"op": "turn", "direction": "right"}]}),
    ("stand for 3 seconds and then turn right 180 degrees",
     {"steps": [{"op": "stand", "duration_s": 3},
                {"op": "turn", "direction": "right", "angle_deg": 180}]}),
    ("walk 1.5 m in 3 seconds, then open the door",
     {"steps": [{"op": "walk", "direction": "forward", "distance_m": 1.5, "within_s": 3},
                {"op": "unsupported", "what": "open the door"}]}),
    ("twice: turn 45 degrees at 0.2 rad/s and walk 1 m at 0.3 m/s",
     {"steps": [{"op": "turn", "direction": "left", "angle_deg": 45, "rate_rad_s": 0.2},
                {"op": "walk", "direction": "forward", "distance_m": 1, "speed_mps": 0.3},
                {"op": "turn", "direction": "left", "angle_deg": 45, "rate_rad_s": 0.2},
                {"op": "walk", "direction": "forward", "distance_m": 1, "speed_mps": 0.3}]}),
]

WALK_DIRS = ("forward", "backward", "left", "right")
TURN_DIRS = ("left", "right")


def transcription_schema(lim):
    """The structured-output target. One variant per step shape, so the grammar
    itself rules out "a walk with both a distance and a duration"."""
    num = {"type": "number"}

    def obj(props, required):
        return {"type": "object", "properties": props, "required": required,
                "additionalProperties": False}

    walk = {"const": "walk"}
    turn = {"const": "turn"}
    wdir = {"enum": list(WALK_DIRS)}
    tdir = {"enum": list(TURN_DIRS)}
    variants = [
        obj({"op": walk, "direction": wdir, "distance_m": num, "speed_mps": num,
             "within_s": num}, ["op", "direction", "distance_m"]),
        obj({"op": walk, "direction": wdir, "duration_s": num, "speed_mps": num},
            ["op", "direction", "duration_s"]),
        obj({"op": walk, "direction": wdir}, ["op", "direction"]),
        obj({"op": turn, "direction": tdir, "angle_deg": num, "rate_rad_s": num,
             "within_s": num}, ["op", "direction", "angle_deg"]),
        obj({"op": turn, "direction": tdir, "duration_s": num, "rate_rad_s": num},
            ["op", "direction", "duration_s"]),
        obj({"op": turn, "direction": tdir}, ["op", "direction"]),
        obj({"op": {"const": "stand"}, "duration_s": num}, ["op", "duration_s"]),
        obj({"op": {"const": "stand"}}, ["op"]),
        obj({"op": {"const": "stop"}}, ["op"]),
        obj({"op": {"const": "unsupported"}, "what": {"type": "string", "maxLength": 80}},
            ["op", "what"]),
    ]
    return {"type": "object",
            "properties": {"steps": {"type": "array", "items": {"anyOf": variants},
                                     "minItems": 1, "maxItems": int(lim.max_prims)}},
            "required": ["steps"], "additionalProperties": False}


def messages(text):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in EXAMPLES:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": json.dumps(a, separators=(",", ":"))})
    msgs.append({"role": "user", "content": text})
    return msgs


# The prompt the offline eval was run with (docs/stageC_nl_eval.md). tests/test_nl.py
# fails if the shipped prompt differs, so a prompt edit cannot ship without a re-run.
EVALUATED_PROMPT_ID = "c0c08d6816ca"


def prompt_id(lim):
    """Short hash of everything the model is given except the instruction, so a
    logged answer or an eval result can be tied to the exact prompt."""
    blob = json.dumps({"system": SYSTEM, "examples": EXAMPLES,
                       "schema": transcription_schema(lim)}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------- backends
class LLMError(Exception):
    """The model could not be asked, or its answer could not be used."""


def _post_json(url, body, timeout_s):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise LLMError("{} returned HTTP {}: {}".format(url, exc.code, detail))
    except socket.timeout:
        raise LLMError("no answer from {} within {:.0f} s".format(url, timeout_s))
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, socket.timeout):
            raise LLMError("no answer from {} within {:.0f} s".format(url, timeout_s))
        raise LLMError("could not reach {}: {}".format(url, reason))
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise LLMError("{} did not return JSON: {}".format(url, exc))


def _get_json(url, timeout_s=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- a probe: any failure means "not there"
        raise LLMError("could not reach {}: {}".format(url, exc))


def _secs(ns):
    return None if ns is None else round(float(ns) / 1e9, 4)


class OllamaBackend(object):
    """Ollama's /api/chat with `format` = the JSON schema (constrained decoding)."""

    name = "ollama"

    def __init__(self, url, model, timeout_s=120.0, options=None, keep_alive="30m",
                 schema=True, think=False):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.keep_alive = keep_alive
        self.schema = schema            # False: plain JSON mode, for the ablation only
        self.think = think              # thinking models (qwen3) answer faster without it
        self.options = {"temperature": 0.0, "seed": 0, "num_predict": 640, "num_ctx": 2048}
        self.options.update(options or {})

    def version(self):
        return _get_json(self.url + "/api/version").get("version")

    def transcribe(self, text, schema):
        body = {"model": self.model, "messages": messages(text), "stream": False,
                "format": schema if self.schema else "json",
                "options": self.options, "keep_alive": self.keep_alive}
        if self.think is not None:
            body["think"] = bool(self.think)
        t0 = time.monotonic()
        try:
            data = _post_json(self.url + "/api/chat", body, self.timeout_s)
        except LLMError as exc:
            if "think" in body and "not support thinking" in str(exc):
                del body["think"]           # a model without a thinking switch
                data = _post_json(self.url + "/api/chat", body, self.timeout_s)
            else:
                raise
        wall = time.monotonic() - t0
        msg = data.get("message") or {}
        meta = {"backend": self.name, "url": self.url, "model": self.model,
                "wall_s": round(wall, 4), "load_s": _secs(data.get("load_duration")),
                "prompt_tokens": data.get("prompt_eval_count"),
                "prompt_s": _secs(data.get("prompt_eval_duration")),
                "gen_tokens": data.get("eval_count"),
                "gen_s": _secs(data.get("eval_duration")),
                "done_reason": data.get("done_reason")}
        return msg.get("content") or "", meta


class OpenAIBackend(object):
    """/v1/chat/completions with a json_schema response_format: a self-built
    llama-server (the fallback runtime), or any OpenAI-compatible server."""

    name = "openai"

    def __init__(self, url, model, timeout_s=120.0, options=None, schema=True):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.schema = schema
        self.options = {"temperature": 0.0, "seed": 0, "max_tokens": 640}
        self.options.update(options or {})

    def version(self):
        return None

    def transcribe(self, text, schema):
        body = {"model": self.model, "messages": messages(text), "stream": False}
        body.update(self.options)
        body["response_format"] = (
            {"type": "json_schema",
             "json_schema": {"name": "steps", "strict": True, "schema": schema}}
            if self.schema else {"type": "json_object"})
        # Thinking off, said both ways: Ollama's OpenAI endpoint honours
        # reasoning_effort, llama-server honours chat_template_kwargs. With neither,
        # qwen3 spent all 640 tokens thinking and never wrote the JSON (dev box,
        # 2026-09-30).
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}
        t0 = time.monotonic()
        try:
            data = _post_json(self.url + "/v1/chat/completions", body, self.timeout_s)
        except LLMError as exc:
            if "HTTP 400" not in str(exc):
                raise
            body.pop("reasoning_effort", None)      # a server that rejects the field
            body.pop("chat_template_kwargs", None)
            data = _post_json(self.url + "/v1/chat/completions", body, self.timeout_s)
        wall = time.monotonic() - t0
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError("unexpected answer shape from {}".format(self.url))
        usage = data.get("usage") or {}
        tm = data.get("timings") or {}          # llama-server adds these
        meta = {"backend": self.name, "url": self.url, "model": self.model,
                "wall_s": round(wall, 4), "load_s": None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "prompt_s": (tm["prompt_ms"] / 1000.0) if "prompt_ms" in tm else None,
                "gen_tokens": usage.get("completion_tokens"),
                "gen_s": (tm["predicted_ms"] / 1000.0) if "predicted_ms" in tm else None,
                "done_reason": (data["choices"][0].get("finish_reason")
                                if data.get("choices") else None)}
        return content, meta


def make_backend(kind, url, model, timeout_s=120.0, options=None, schema=True):
    if kind == "ollama":
        return OllamaBackend(url, model, timeout_s, options, schema=schema)
    if kind == "openai":
        return OpenAIBackend(url, model, timeout_s, options, schema=schema)
    raise ValueError("unknown backend {!r} (ollama | openai)".format(kind))


# ---------------------------------------------------------------- translate
REASONS = ("backward", "sideways", "unsupported", "no_amount", "too_fast",
           "too_small", "limits", "invalid", "empty")

# Slowest walk the NL front end will slow a short walk down to. 0.1 m/s is the
# slowest command the real robot has walked at (PG-2, 2026-09-28, untethered).
WALK_MIN_VX = 0.1


class _Bad(Exception):
    pass


def _num(step, key):
    if key not in step or step[key] is None:
        return None
    v = step[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise _Bad("{} is not a number ({!r})".format(key, v))
    if v < 0:
        raise _Bad("{} is negative ({})".format(key, _fmt(v)))
    return float(v)


def step_text(s):
    """A transcribed step in plain words, for the operator and the refusal text."""
    if not isinstance(s, dict):
        return repr(s)

    def f(key):
        v = s.get(key)
        return _fmt(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    op = s.get("op")
    if op in ("walk", "turn"):
        t = "{} {}".format(op, s.get("direction", "forward" if op == "walk" else "left"))
        if s.get("distance_m") is not None:
            t += " {} m".format(f("distance_m"))
        elif s.get("angle_deg") is not None:
            t += " {} deg".format(f("angle_deg"))
        elif s.get("duration_s") is not None:
            t += " for {} s".format(f("duration_s"))
        if s.get("within_s") is not None:
            t += " in {} s".format(f("within_s"))
        if s.get("speed_mps") is not None:
            t += " at {} m/s".format(f("speed_mps"))
        if s.get("rate_rad_s") is not None:
            t += " at {} rad/s".format(f("rate_rad_s"))
        return t
    if op == "stand":
        return "stand" + (" for {} s".format(f("duration_s"))
                          if s.get("duration_s") is not None else "")
    if op == "stop":
        return "stop"
    if op == "unsupported":
        return str(s.get("what") or "something I cannot do")
    return json.dumps(s, ensure_ascii=False)


def _floor4(x):
    """Round a derived rate DOWN, so the step it produces lasts at least as long as
    intended: round() could turn a 2.000 s turn into 1.9999 s, under the floor."""
    return math.floor(x * 1e4) / 1e4


def min_turn_deg(lim):
    """Smallest closed-loop turn: min_primitive_s at the slowest measured rate."""
    return math.degrees(lim.turn_min_wz * lim.min_primitive_s)


def refusal_message(reason, env, lim, text, detail=None):
    """One deterministic English sentence per reason, from the deployed numbers."""
    vmax = env.vx[1]
    wmax = max(abs(env.wz[0]), abs(env.wz[1]))
    if reason == "backward":
        why = ("my forward speed was never negative in training" if env.vx[0] >= 0.0
               else "my mission layer has no backward step")
        return ("I cannot walk backward ({}): {}. I can turn around (180 degrees) and "
                "walk forward instead.".format(text, why))
    if reason == "sideways":
        why = ("that axis was fixed at zero for all of my training" if env.pinned("vy")
               else "my mission layer has no sideways step")
        return ("I cannot move sideways ({}): {}. I can turn 90 degrees and walk "
                "forward instead.".format(text, why))
    if reason == "unsupported":
        return ("I cannot {}: I can only walk forward, turn on the spot and stand "
                "still, one step after another.".format(text))
    if reason == "no_amount":
        return ("'{}' has no amount. Tell me how far (meters), how long (seconds) or "
                "how many degrees.".format(text))
    if reason == "too_fast":
        return ("'{}' needs {}; I walk at up to {:.2f} m/s and turn at up to {:.2f} "
                "rad/s.".format(text, detail or "more than I can do", vmax, wmax))
    if reason == "too_small":
        return ("'{}' is too small: {}.".format(text, detail or "below what I can do"))
    if reason == "limits":
        return "I will not run this plan: {}".format(detail or text)
    if reason == "empty":
        return "I did not find anything to do in that."
    return "I could not make sense of '{}'{}.".format(text, ": " + detail if detail else "")


def translate(transcript, env, lim):
    """The model's transcription -> (steps for plan_from_json, refusals).

    Every judgement is made here from the deployed envelope and limits, never by
    the model. Defaults the operator did not state are chosen here too, and they
    show up in the plan the operator confirms:
      * a closed-loop turn with no rate runs at cruise_wz, slowed only as far as
        needed to last min_primitive_s, never below turn_min_wz;
      * a distance walk with no speed runs at cruise_vx, slowed the same way,
        never below WALK_MIN_VX.
    """
    refusals, steps = [], []

    def refuse(i, s, reason, detail=None):
        text = step_text(s)
        refusals.append({"step": i + 1, "text": text, "reason": reason,
                         "message": refusal_message(reason, env, lim, text, detail)})

    if not isinstance(transcript, dict) or not isinstance(transcript.get("steps"), list):
        refusals.append({"step": None, "text": "", "reason": "invalid",
                         "message": "The model's answer had no list of steps."})
        return steps, refusals
    if not transcript["steps"]:
        refusals.append({"step": None, "text": "", "reason": "empty",
                         "message": refusal_message("empty", env, lim, "")})
        return steps, refusals

    vmax = env.vx[1]
    wmax = max(abs(env.wz[0]), abs(env.wz[1]))
    for i, s in enumerate(transcript["steps"]):
        if not isinstance(s, dict):
            refuse(i, s, "invalid", "not an object")
            continue
        op = s.get("op")
        text = step_text(s)
        try:
            if op == "unsupported":
                refuse(i, s, "unsupported")
            elif op == "stop":
                steps.append({"op": "stop", "text": text})
            elif op == "stand":
                dur = _num(s, "duration_s")
                if dur is None:
                    refuse(i, s, "no_amount")
                else:
                    steps.append({"op": "stand", "duration_s": dur, "text": text})
            elif op == "walk":
                direction = s.get("direction", "forward")
                if direction == "backward":
                    refuse(i, s, "backward")
                    continue
                if direction in ("left", "right"):
                    refuse(i, s, "sideways")
                    continue
                if direction != "forward":
                    raise _Bad("unknown walk direction {!r}".format(direction))
                dist, dur = _num(s, "distance_m"), _num(s, "duration_s")
                spd, within = _num(s, "speed_mps"), _num(s, "within_s")
                if dist is not None and dur is not None:
                    raise _Bad("both a distance and a duration")
                if dist is None and dur is None:
                    refuse(i, s, "no_amount")
                    continue
                vx = None
                if within is not None:
                    if dist is None or within <= 0.0:
                        raise _Bad("a time limit needs a distance")
                    vx = _floor4(dist / within)
                if spd is not None:
                    if vx is not None and abs(spd - vx) > 0.02 + 0.02 * vx:
                        raise _Bad("speed {} m/s and '{} m in {} s' disagree".format(
                            _fmt(spd), _fmt(dist), _fmt(within)))
                    vx = spd
                if vx is not None and vx > vmax + 1e-9:
                    refuse(i, s, "too_fast", "{:.2f} m/s".format(vx))
                    continue
                if vx is None and dist is not None and lim.cruise_vx > 0.0 \
                        and dist / lim.cruise_vx < lim.min_primitive_s:
                    vx = _floor4(dist / lim.min_primitive_s)
                    if vx < WALK_MIN_VX - 1e-9:
                        refuse(i, s, "too_small",
                               "the shortest walk is about {:.1f} m ({:.0f} s at {:.2f} "
                               "m/s, my slowest tested speed)".format(
                                   WALK_MIN_VX * lim.min_primitive_s, lim.min_primitive_s,
                                   WALK_MIN_VX))
                        continue
                st = {"op": "walk", "text": text}
                if dist is not None:
                    st["distance_m"] = dist
                else:
                    st["duration_s"] = dur
                if vx is not None:
                    st["vx"] = vx
                steps.append(st)
            elif op == "turn":
                direction = s.get("direction", "left")
                if direction not in TURN_DIRS:
                    raise _Bad("unknown turn direction {!r}".format(direction))
                ang, dur = _num(s, "angle_deg"), _num(s, "duration_s")
                rate, within = _num(s, "rate_rad_s"), _num(s, "within_s")
                if ang is not None and dur is not None:
                    raise _Bad("both an angle and a duration")
                if ang is None and dur is None:
                    refuse(i, s, "no_amount")
                    continue
                wz = None
                if within is not None:
                    if ang is None or within <= 0.0:
                        raise _Bad("a time limit needs an angle")
                    wz = _floor4(math.radians(ang) / within)
                if rate is not None:
                    if wz is not None and abs(rate - wz) > 0.02 + 0.02 * wz:
                        raise _Bad("rate {} rad/s and '{} deg in {} s' disagree".format(
                            _fmt(rate), _fmt(ang), _fmt(within)))
                    wz = rate
                if wz is not None and wz > wmax + 1e-9:
                    refuse(i, s, "too_fast", "{:.2f} rad/s".format(wz))
                    continue
                if ang is not None and wz is None and lim.cruise_wz > 0.0 and \
                        math.radians(ang) / lim.cruise_wz < lim.min_primitive_s:
                    wz = _floor4(math.radians(ang) / lim.min_primitive_s)
                    if wz < lim.turn_min_wz - 1e-12:
                        refuse(i, s, "too_small",
                               "the smallest turn is about {:.0f} degrees ({:.0f} s at "
                               "{:.2f} rad/s, my slowest measured turn)".format(
                                   min_turn_deg(lim), lim.min_primitive_s, lim.turn_min_wz))
                        continue
                st = {"op": "turn", "direction": direction, "text": text}
                if ang is not None:
                    st["yaw_deg"] = ang
                else:
                    st["duration_s"] = dur
                if wz is not None:
                    st["wz"] = wz
                steps.append(st)
            else:
                raise _Bad("unknown op {!r}".format(op))
        except _Bad as exc:
            refuse(i, s, "invalid", str(exc))
    return steps, refusals


class Understood(object):
    """Everything about one instruction: what was typed, what the model wrote,
    what the robot decided. `outcome` is "execute" only if every step passed."""

    def __init__(self, text, normalize_input=True):
        self.text = text
        # What the operator said, read deterministically. Amounts are checked
        # against this even when the model is shown something else (the ablation).
        self.said, self.notes = normalize(text)
        if normalize_input:
            self.normalized = self.said
        else:   # the eval's ablation: the model gets the words as typed
            self.normalized = re.sub(r"\s+", " ", str(text).strip().lower())
        self.fast_path = False
        self.raw = None
        self.meta = {}
        self.transcript = None
        self.invented = []       # amounts the model wrote that were never said
        self.steps = []
        self.refusals = []
        self.prims, self.total_s, self.total_m = [], 0.0, 0.0

    @property
    def outcome(self):
        return "execute" if self.prims and not self.refusals else "refuse"

    @property
    def reasons(self):
        return sorted(set(r["reason"] for r in self.refusals))

    def to_dict(self):
        return {"text": self.text, "normalized": self.normalized, "notes": self.notes,
                "fast_path": self.fast_path, "raw": self.raw, "meta": self.meta,
                "transcript": self.transcript, "invented": self.invented,
                "steps": self.steps,
                "refusals": self.refusals, "outcome": self.outcome,
                "plan": [p.summary() for p in self.prims],
                "total_s": round(self.total_s, 2), "total_m": round(self.total_m, 2)}


def parse_transcript(raw):
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    return obj


# Each amount field and the unit it must carry in the normalized instruction.
AMOUNT_UNITS = (("distance_m", r" m(?![/\w])"), ("duration_s", r" seconds\b"),
                ("within_s", r" seconds\b"), ("angle_deg", r" degrees\b"))


def drop_invented(transcript, said):
    """Remove every amount the operator did not say, WITH ITS UNIT. `said` is the
    normalized instruction, so a distance must appear there as "<x> m", a time as
    "<x> seconds", an angle as "<x> degrees".

    "Never make up a number" is in the prompt, but a prompt is a request; this is
    the guarantee. The unit matters: in "walk at 0.56 m/s for 5 seconds" a model
    wrote distance_m 0.56 -- a number that WAS said, as a speed (test set,
    2026-09-30). A step that loses its amount is refused for having none, and the
    operator is asked for one.

    Speeds and turn rates are not checked: "2 m in 4 s" may legitimately come back
    as 0.5 m/s, both are bounded by the envelope, and the plan the operator
    confirms prints them.
    """
    heard = {k: [float(x) for x in re.findall(r"(?<![\w.])" + NUM + unit, said)]
             for k, unit in AMOUNT_UNITS}
    dropped = []
    if not isinstance(transcript, dict) or not isinstance(transcript.get("steps"), list):
        return transcript, dropped
    steps = []
    for i, s in enumerate(transcript["steps"]):
        if isinstance(s, dict):
            s = dict(s)
            for k, _ in AMOUNT_UNITS:
                v = s.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and \
                        not any(abs(float(v) - x) <= 0.005 + 1e-9 for x in heard[k]):
                    dropped.append({"step": i + 1, "field": k, "value": v})
                    del s[k]
        steps.append(s)
    out = dict(transcript)
    out["steps"] = steps
    return out, dropped


def judge(u, env, lim):
    """Fill in the robot's decision on an Understood whose transcript is set."""
    if u.transcript is None:
        u.refusals = [{"step": None, "text": "", "reason": "invalid",
                       "message": "The model's answer was not valid JSON."}]
        return u
    checked, u.invented = drop_invented(u.transcript, u.said)
    u.steps, u.refusals = translate(checked, env, lim)
    if u.refusals:
        return u
    try:
        u.prims, u.total_s, u.total_m = P.plan_from_json({"plan": u.steps}, env, lim)
    except P.PlanError as exc:
        first = str(exc).split("\n")[0]
        u.refusals = [{"step": None, "text": "", "reason": "limits", "message":
                       refusal_message("limits", env, lim, "", first), "detail": str(exc)}]
        u.prims = []
    return u


def understand(text, backend, env, lim, normalize_input=True):
    """Instruction -> Understood. Raises LLMError only if the model could not be
    asked at all; anything wrong with its answer is a refusal, not an exception."""
    u = Understood(text, normalize_input)
    if FAST_STOP.match(u.normalized):
        u.fast_path = True
        u.transcript = {"steps": [{"op": "stop"}]}
        u.meta = {"backend": "fast-path", "wall_s": 0.0}
        return judge(u, env, lim)
    u.raw, u.meta = backend.transcribe(u.normalized, transcription_schema(lim))
    u.transcript = parse_transcript(u.raw)
    return judge(u, env, lim)


# ------------------------------------------------------------------ replies
def render_understood(u):
    lines = []
    if u.normalized != re.sub(r"\s+", " ", u.text.strip().lower()):
        lines.append("read as:    " + u.normalized)
    for n in u.notes:
        lines.append("            ({})".format(n))
    m = u.meta or {}
    if u.fast_path:
        lines.append("model:      not asked (a bare stop needs no model)")
    elif m:
        tok = m.get("gen_tokens")
        rate = ("{:.1f} tok/s".format(tok / m["gen_s"]) if tok and m.get("gen_s") else "")
        lines.append("model:      {} via {}, {:.1f} s{}".format(
            m.get("model"), m.get("backend"), m.get("wall_s") or 0.0,
            " ({} tokens{})".format(tok, ", " + rate if rate else "") if tok else ""))
    if isinstance(u.transcript, dict) and isinstance(u.transcript.get("steps"), list):
        lines.append("understood:")
        for i, s in enumerate(u.transcript["steps"]):
            lines.append("  {}. {}".format(i + 1, step_text(s)))
    return "\n".join(lines)


def render_refusals(u):
    lines = ["I will not do this, and nothing was executed:"]
    for r in u.refusals:
        lines.append("  - " + r["message"])
    return "\n".join(lines)


def _side(deg):
    return "left" if deg >= 0 else "right"


def render_report(rep, prims):
    """The executor's report as deterministic English. Measured and estimated
    quantities keep their labels: the angle is the IMU's, the distance is time x
    commanded speed."""
    lines = []
    recs = rep.get("executed") or []
    for i, r in enumerate(recs):
        p = prims[i] if i < len(prims) else None
        t = r.get("actual_s")
        op = r.get("op")
        head = "{}.".format(i + 1)
        if op == "walk" and p is not None:
            d = r.get("est_distance_m")
            cmd = r.get("commanded_s")
            if r.get("aborted") and d is not None and cmd and t is not None:
                d = d * min(1.0, t / cmd)
            s = "{} Walked forward for {:.1f} s at {:.2f} m/s".format(head, t or 0.0, p.vx)
            if d is not None:
                if r.get("est_err_m") is not None:
                    s += ": about {:.1f} +- {:.1f} m (time x calibrated speed; no odometry)".format(
                        d, r["est_err_m"])
                else:
                    s += (": about {:.1f} m (time x commanded speed; the speed was not "
                          "measured and there is no odometry)".format(d))
            lines.append(s + ".")
        elif op == "turn" and p is not None and p.closed_loop:
            a = r.get("achieved_deg")
            if a is None:
                lines.append("{} Did not start turning (no IMU heading).".format(head))
            else:
                lines.append("{} Turned {} {:.1f}° by my IMU (target {:.0f}°; not "
                             "checked against the floor).".format(
                                 head, _side(a), abs(a), abs(r.get("target_deg") or 0.0)))
        elif op == "turn" and p is not None:
            lines.append("{} Turned {} for {:.1f} s at {:.2f} rad/s (open loop; the angle "
                         "was not measured).".format(head, _side(p.wz), t or 0.0, abs(p.wz)))
        elif op == "stand":
            lines.append("{} Stood still for {:.1f} s.".format(head, t or 0.0))
        elif op == "stop":
            lines.append("{} Stopped.".format(head))
        if r.get("aborted"):
            lines[-1] += " Interrupted: {}.".format(r["aborted"])
    for j in range(len(recs), len(prims)):
        lines.append("{}. Not started: {}.".format(j + 1, prims[j].text or prims[j].summary()))
    if rep.get("final_state") == "DONE":
        lines.append("Done.")
    else:
        why = rep.get("abort_reason") or "unknown"
        lines.append("Stopped early: {}.".format(why) if recs else
                     "Nothing was executed: {}.".format(why))
    return "\n".join(lines)
