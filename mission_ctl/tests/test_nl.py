#!/usr/bin/env python3
"""Tests for the English front end (r1_mission/nl.py) and the `ask` command.
No ROS, no model, no pytest -- `python3 tests/test_nl.py`.

The model is replaced by a stand-in HTTP server on 127.0.0.1 that speaks both
Ollama's /api/chat and the OpenAI /v1/chat/completions shape, so the real HTTP
client, its timeouts and its error paths are exercised on the robot too.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from r1_mission import executor as ex   # noqa: E402
from r1_mission import nl               # noqa: E402
from r1_mission import node as N        # noqa: E402
from r1_mission import plan as P        # noqa: E402

TRAINED = P.Envelope([0.0, 1.0], [0.0, 0.0], [-0.5, 0.5], source="as trained")
LIM = N.limits_from_cfg(N.read_flat_yaml(os.path.join(os.path.dirname(HERE), "config",
                                                      "mission.yaml")))
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  {}  {}{}".format("ok  " if cond else "FAIL", name,
                              "" if cond else "  <- " + str(detail)))


def norm(t):
    return nl.normalize(t)[0]


print("\n== normalize: number words ==")
check("two meters", norm("walk two meters") == "walk 2 m", norm("walk two meters"))
check("one hundred and eighty", norm("turn right one hundred and eighty degrees")
      == "turn right 180 degrees", norm("turn right one hundred and eighty degrees"))
check("forty-five", norm("turn left forty-five degrees") == "turn left 45 degrees")
check("'and' does not glue two numbers", norm("walk two and turn") == "walk 2 and turn",
      norm("walk two and turn"))
check("five two stays two numbers", norm("five two") == "5 2", norm("five two"))
check("a hundred", norm("a hundred meters") == "100 m", norm("a hundred meters"))

print("\n== normalize: units ==")
cases = [
    ("Walk forward 10 feet.", "walk forward 3.05 m."),
    ("Walk 3 yards.", "walk 2.74 m."),
    ("Walk forward 50 centimeters.", "walk forward 0.5 m."),
    ("Walk forward for half a minute.", "walk forward for 30 seconds."),
    ("walk for a quarter of a minute", "walk for 15 seconds"),
    ("Walk forward for three quarters of a minute.", "walk forward for 45 seconds."),
    ("one and a half meters", "1.5 m"),
    ("a meter and a half", "1.5 m"),
    ("walk 2 feet and a half", "walk 0.76 m"),
    ("walk at 2 kilometers per hour", "walk at 0.56 m/s"),
    ("walk at 0.3 meters per second", "walk at 0.3 m/s"),
    ("turn at 30 deg/s", "turn at 0.52 rad/s"),
    ("turn 1.57 rad", "turn 89.95 degrees"),
    ("walk 10m then turn 90°", "walk 10 m then turn 90 degrees"),
    ("walk 5 min", "walk 300 seconds"),
    ("walk 2 m/s", "walk 2 m/s"),
    ("walk 5 m in 10 seconds", "walk 5 m in 10 seconds"),
]
for src, want in cases:
    check("'{}'".format(src), norm(src) == want, "got '{}'".format(norm(src)))
n, notes = nl.normalize("Walk forward 10 feet.")
check("a conversion is reported to the operator", notes == ["10 feet -> 3.05 m"], notes)
check("a same-unit spelling is not reported", nl.normalize("walk 2 meters")[1] == [])

print("\n== normalize: turns ==")
check("turn around is 180", norm("Turn around.") == "turn 180 degrees.")
check("a quarter turn is 90", norm("Make a quarter turn to the right.")
      == "make a 90 degrees turn to the right.", norm("Make a quarter turn to the right."))
check("u-turn", norm("Make a U-turn.") == "make a 180 degrees turn.")
check("plain turn gets the default 90", norm("then turn left.") == "then turn left 90 degrees.")
check("turn to the right gets it too", norm("turn to the right") == "turn right 90 degrees")
check("take a left", norm("take a left") == "turn left 90 degrees")
check("no default when an angle follows", norm("turn left 45 degrees") == "turn left 45 degrees")
check("no default for a timed turn", norm("turn left for 3 seconds") == "turn left for 3 seconds")
check("no default when the sentence is vague", norm("Turn right a little.") == "turn right a little.")
check("...even when the vague word is far away",
      norm("turn right, but only a little") == "turn right, but only a little")
check("no second angle inside 'a 90 degree turn to the right'",
      norm("walk 1 m and then turn right a quarter turn").count("90") == 1,
      norm("walk 1 m and then turn right a quarter turn"))

print("\n== normalize: repetition ==")
r = norm("Twice: walk forward 2 meters and turn right 180 degrees.")
check("'twice:' is expanded", r.count("walk forward 2 m") == 2 and r.count("180") == 2, r)
r = norm("Do this three times: walk forward 1 meter, then turn left 90 degrees.")
check("'do this 3 times:' is expanded", r.count("walk forward 1 m") == 3, r)
r, notes = nl.normalize("Turn left 90 degrees twice.")
check("a repetition without a colon is not guessed at", r == "turn left 90 degrees 2 times.", r)
check("...but the operator is told", any("repetition" in x for x in notes), notes)
check("a huge count is not expanded", "50 times" in norm("50 times: walk 1 m"))

print("\n== fast stop ==")
for t in ("Stop.", "stop", "Please stop now.", "HALT!", "freeze"):
    check("'{}' needs no model".format(t), nl.FAST_STOP.match(norm(t)) is not None)
check("'walk 2 m then stop' is not a bare stop",
      nl.FAST_STOP.match(norm("walk 2 m then stop")) is None)

print("\n== schema ==")
sch = nl.transcription_schema(LIM)
check("steps capped at max_prims", sch["properties"]["steps"]["maxItems"] == LIM.max_prims)
variants = sch["properties"]["steps"]["items"]["anyOf"]
check("ten step shapes", len(variants) == 10, len(variants))
check("every shape is closed", all(v["additionalProperties"] is False for v in variants))
check("walk can say backward and sideways (so the robot can refuse them)",
      set(variants[0]["properties"]["direction"]["enum"]) ==
      {"forward", "backward", "left", "right"})
check("the model is not told the envelope",
      "1.00" not in nl.SYSTEM and "0.50" not in nl.SYSTEM and "rad/s" in nl.SYSTEM)
for q, a in nl.EXAMPLES[:-1]:
    check("example '{}' is already normalized".format(q[:30]), norm(q) == q, norm(q))
check("the last example is the repetition one (kept as evaluated)",
      nl.EXAMPLES[-1][0].startswith("twice:"))
check("the shipped prompt is the one the eval was run with",
      nl.prompt_id(LIM) == nl.EVALUATED_PROMPT_ID,
      "prompt {} != evaluated {}: re-run eval/run_nl_eval.py and update "
      "EVALUATED_PROMPT_ID".format(nl.prompt_id(LIM), nl.EVALUATED_PROMPT_ID))
msgs = nl.messages("walk 2 m")
check("chat = system + examples + instruction",
      len(msgs) == 2 + 2 * len(nl.EXAMPLES) and msgs[-1]["content"] == "walk 2 m")
check("prompt id is stable", nl.prompt_id(LIM) == nl.prompt_id(LIM)
      and len(nl.prompt_id(LIM)) == 12)


def judged(text, steps):
    u = nl.Understood(text)
    u.transcript = {"steps": steps}
    return nl.judge(u, TRAINED, LIM)


print("\n== translate: the robot decides ==")
u = judged("walk forward 2 m, then turn left 90 degrees",
           [{"op": "walk", "direction": "forward", "distance_m": 2},
            {"op": "turn", "direction": "left", "angle_deg": 90}])
check("a plain plan executes", u.outcome == "execute" and len(u.prims) == 2, u.refusals)
check("default speeds", abs(u.prims[0].vx - LIM.cruise_vx) < 1e-9
      and abs(u.prims[1].wz - LIM.cruise_wz) < 1e-9)
u = judged("walk backward 1 m", [{"op": "walk", "direction": "backward", "distance_m": 1}])
check("backward is refused, by name", u.reasons == ["backward"], u.reasons)
check("...with the way to do it instead", "turn around" in u.refusals[0]["message"])
u = judged("strafe left 2 m", [{"op": "walk", "direction": "left", "distance_m": 2}])
check("sideways is refused, from the pinned axis", u.reasons == ["sideways"]
      and "fixed at zero" in u.refusals[0]["message"], u.refusals)
wide = P.Envelope([0.0, 1.0], [-0.3, 0.3], [-0.5, 0.5])
u = nl.Understood("strafe left 2 m")
u.transcript = {"steps": [{"op": "walk", "direction": "left", "distance_m": 2}]}
nl.judge(u, wide, LIM)
check("with vy trained, the reason changes (no sideways step in mission_ctl)",
      u.reasons == ["sideways"] and "no sideways step" in u.refusals[0]["message"])
u = judged("walk 2 m, then jump", [{"op": "walk", "direction": "forward", "distance_m": 2},
                                   {"op": "unsupported", "what": "jump"}])
check("all or nothing: one refused step refuses the plan",
      u.outcome == "refuse" and u.prims == [] and u.reasons == ["unsupported"])
u = judged("go a little further", [{"op": "walk", "direction": "forward"}])
check("no amount is refused", u.reasons == ["no_amount"])
u = judged("turn 90 degrees in 0.5 seconds",
           [{"op": "turn", "direction": "left", "angle_deg": 90, "within_s": 0.5}])
check("a time limit becomes a rate, and too fast is refused", u.reasons == ["too_fast"]
      and "3.14 rad/s" in u.refusals[0]["message"], u.refusals)
u = judged("walk 5 m in 2 seconds",
           [{"op": "walk", "direction": "forward", "distance_m": 5, "within_s": 2}])
check("2.5 m/s is refused as too fast", u.reasons == ["too_fast"])
u = judged("walk 2 m in 4 seconds",
           [{"op": "walk", "direction": "forward", "distance_m": 2, "within_s": 4}])
check("2 m in 4 s runs at 0.5 m/s", u.outcome == "execute"
      and abs(u.prims[0].vx - 0.5) < 1e-9 and abs(u.prims[0].duration_s - 4.0) < 1e-6)
u = judged("walk forward 40 m", [{"op": "walk", "direction": "forward", "distance_m": 40}])
check("the distance budget still refuses", u.reasons == ["limits"]
      and "budget" in u.refusals[0]["message"], u.refusals)
u = judged("turn right 45 degrees", [{"op": "turn", "direction": "right", "angle_deg": 45}])
check("a short turn is slowed to last min_primitive_s", u.outcome == "execute"
      and u.prims[0].duration_s >= LIM.min_primitive_s
      and abs(u.prims[0].wz) >= LIM.turn_min_wz, [p.summary() for p in u.prims])
u = judged("turn right 18 degrees", [{"op": "turn", "direction": "right", "angle_deg": 18}])
check("18 deg is just above the smallest turn", u.outcome == "execute", u.refusals)
u = judged("turn right 10 degrees", [{"op": "turn", "direction": "right", "angle_deg": 10}])
check("10 deg is below the smallest turn", u.reasons == ["too_small"]
      and "17 degrees" in u.refusals[0]["message"], u.refusals)
u = judged("walk forward 0.5 m", [{"op": "walk", "direction": "forward", "distance_m": 0.5}])
check("a short walk is slowed to last min_primitive_s", u.outcome == "execute"
      and u.prims[0].duration_s >= LIM.min_primitive_s and u.prims[0].vx >= nl.WALK_MIN_VX)
u = judged("walk forward 0.1 m", [{"op": "walk", "direction": "forward", "distance_m": 0.1}])
check("0.1 m is below the shortest walk", u.reasons == ["too_small"])
steps, refs = nl.translate({"steps": [{"op": "walk", "direction": "forward",
                                        "distance_m": 2, "duration_s": 5}]}, TRAINED, LIM)
check("distance and duration together are invalid", [r["reason"] for r in refs] == ["invalid"])
steps, refs = nl.translate({"steps": [{"op": "walk", "direction": "forward",
                                        "distance_m": -2}]}, TRAINED, LIM)
check("a negative amount is invalid", [r["reason"] for r in refs] == ["invalid"])
u = judged("walk 2 m", [{"op": "walk", "direction": "forward", "distance_m": "2"}])
check("a string amount is invalid", u.reasons == ["invalid"])
u = judged("hello", [])
check("an empty transcription is refused", u.reasons == ["empty"])
u = nl.Understood("walk 2 m")
u.transcript = None
nl.judge(u, TRAINED, LIM)
check("unparseable model output is refused, not raised", u.reasons == ["invalid"])
u = judged("stop", [{"op": "stop"}])
check("stop executes", u.outcome == "execute" and u.prims[0].op == "stop")

print("\n== provenance: amounts must have been said ==")
u = judged("walk forward slightly", [{"op": "walk", "direction": "forward",
                                      "distance_m": 0.1}])
check("an invented distance is dropped -> asked for", u.reasons == ["no_amount"]
      and u.invented == [{"step": 1, "field": "distance_m", "value": 0.1}], u.invented)
u = judged("turn right a little", [{"op": "turn", "direction": "right", "angle_deg": 90}])
check("an invented 90 deg is dropped", u.reasons == ["no_amount"])
u = judged("walk forward at 0.56 m/s for 5 seconds",
           [{"op": "walk", "direction": "forward", "distance_m": 0.56, "within_s": 5}])
check("a said number in the wrong unit is dropped (0.56 m/s is not 0.56 m)",
      u.reasons == ["no_amount"], u.refusals)
u = judged("walk forward for 3 seconds",
           [{"op": "walk", "direction": "forward", "distance_m": 3}])
check("3 seconds is not 3 m", u.reasons == ["no_amount"])
u = judged("walk forward 3.05 m", [{"op": "walk", "direction": "forward", "distance_m": 3.048}])
check("rounding within 5 mm is the same number", u.outcome == "execute")
u = judged("walk 2 m in 4 seconds",
           [{"op": "walk", "direction": "forward", "distance_m": 2, "speed_mps": 0.5}])
check("a derived speed is allowed (bounded by the envelope, printed in the plan)",
      u.outcome == "execute" and abs(u.prims[0].vx - 0.5) < 1e-9)

print("\n== replies ==")
lim = P.Limits(cruise_vx=0.4, cruise_wz=0.4, v_cal_assumed=True, turn_min_wz=0.15)
prims, _, _ = P.compile_plan(P.parse_script("walk 2m@0.4; turn left 90; stand 2s; stop"),
                             TRAINED, lim)
e = ex.Executor(prims, lim)
t, yaw = 0.0, 0.0
while t < 60:
    o = e.step(t, yaw=yaw, bridge_state="RUNNING")
    if o.state in (ex.DONE, ex.ABORTED):
        break
    yaw += o.wz * 0.02
    t += 0.02
txt = nl.render_report(e.report(), prims)
check("walk reply says the distance is an estimate",
      "about 2.0 m (time x commanded speed; the speed was not measured" in txt, txt)
check("turn reply says it is the IMU's angle", "by my IMU" in txt and "Turned left" in txt, txt)
check("stand and stop are reported", "Stood still" in txt and "Stopped." in txt, txt)
check("done", txt.endswith("Done."), txt)
e = ex.Executor(prims, lim)
t = 0.0
while t < 60:
    o = e.step(t, yaw=0.0, bridge_state="RUNNING" if t < 3.0 else "DEGRADED")
    if o.state in (ex.DONE, ex.ABORTED):
        break
    t += 0.02
txt = nl.render_report(e.report(), prims)
check("an interrupted walk reports the part walked, not the plan",
      "about 1.2 m" in txt or "about 1.1 m" in txt, txt)
check("steps not started are listed", txt.count("Not started") == 3, txt)
check("the abort reason is given", "Stopped early: bridge left RUNNING" in txt, txt)
e = ex.Executor(prims, lim)
t = 0.0
while t < 2.0:
    e.step(t, yaw=0.0, bridge_state="RUNNING")
    t += 0.02
e.interrupt("operator interrupt (Ctrl-C)")
txt = nl.render_report(e.report(), prims)
check("a Ctrl-C'd walk reports the part walked (stage C: it said 0.0 s, about 0.0 m)",
      "Walked forward for 2.0 s at 0.40 m/s: about 0.8 m" in txt
      and "Stopped early: operator interrupt (Ctrl-C)." in txt, txt)
rep = {"executed": [], "final_state": "ABORTED", "abort_reason": "never saw the bridge"}
check("nothing executed", nl.render_report(rep, prims).endswith(
    "Nothing was executed: never saw the bridge."), nl.render_report(rep, prims))


# ------------------------------------------------------------- stand-in server
ANSWERS = {
    "walk forward 2 m, then turn left 90 degrees.":
        {"steps": [{"op": "walk", "direction": "forward", "distance_m": 2},
                   {"op": "turn", "direction": "left", "angle_deg": 90}]},
    "walk forward 3.05 m.": {"steps": [{"op": "walk", "direction": "forward",
                                        "distance_m": 3.05}]},
    "strafe left 2 m.": {"steps": [{"op": "walk", "direction": "left", "distance_m": 2}]},
}
SEEN = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = (json.dumps(obj) if not isinstance(obj, bytes) else obj)
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send(200, {"version": "fake-0"})
        self._send(404, {"error": "no"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(req)
        model = req.get("model")
        if model == "slow":
            time.sleep(1.5)
        if model == "broken":
            return self._send(500, {"error": "llama runner process has terminated"})
        if model == "garbage":
            return self._send(200, b"not json at all")
        if model == "nothink" and "think" in req:
            return self._send(400, {"error": "\"nothink\" does not support thinking"})
        text = req["messages"][-1]["content"]
        content = json.dumps(ANSWERS.get(text, {"steps": [{"op": "unsupported",
                                                            "what": text}]}))
        if self.path == "/api/chat":
            return self._send(200, {"message": {"role": "assistant", "content": content},
                                    "done_reason": "stop", "load_duration": 5e8,
                                    "prompt_eval_count": 800, "prompt_eval_duration": 1e8,
                                    "eval_count": 20, "eval_duration": 1e9})
        if self.path == "/v1/chat/completions":
            return self._send(200, {"choices": [{"message": {"content": content},
                                                 "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 800, "completion_tokens": 20},
                                    "timings": {"prompt_ms": 100.0, "predicted_ms": 1000.0}})
        self._send(404, {"error": "no route"})


srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = "http://127.0.0.1:{}".format(srv.server_address[1])

print("\n== backends against a stand-in server ==")
be = nl.make_backend("ollama", URL, "m", timeout_s=5)
check("version", be.version() == "fake-0")
u = nl.understand("Walk forward 2 meters, then turn left.", be, TRAINED, LIM)
check("ollama round trip", u.outcome == "execute" and len(u.prims) == 2, u.to_dict())
req = SEEN[-1]
check("the request carries the schema as format", req["format"] == nl.transcription_schema(LIM))
check("greedy and seeded", req["options"]["temperature"] == 0.0 and req["options"]["seed"] == 0)
check("thinking switched off", req.get("think") is False)
check("the model got the normalized text", req["messages"][-1]["content"]
      == "walk forward 2 m, then turn left 90 degrees.")
check("timings are read", u.meta["gen_tokens"] == 20 and abs(u.meta["gen_s"] - 1.0) < 1e-9
      and abs(u.meta["load_s"] - 0.5) < 1e-9, u.meta)
u = nl.understand("walk forward 10 feet.", nl.make_backend("openai", URL, "m", 5),
                  TRAINED, LIM)
check("openai-style round trip", u.outcome == "execute"
      and abs(u.prims[0].distance_m - 3.05) < 1e-9, u.to_dict())
check("...with a json_schema response_format",
      SEEN[-1]["response_format"]["json_schema"]["schema"] == nl.transcription_schema(LIM))
check("...and thinking off, both ways (Ollama and llama-server spell it differently)",
      SEEN[-1].get("reasoning_effort") == "none"
      and SEEN[-1].get("chat_template_kwargs") == {"enable_thinking": False})
u = nl.understand("walk 2 m", nl.make_backend("ollama", URL, "nothink", 5), TRAINED, LIM)
check("a model without a thinking switch is retried without it",
      "think" not in SEEN[-1] and u.raw is not None)
n0 = len(SEEN)
u = nl.understand("Stop!", be, TRAINED, LIM)
check("a bare stop never calls the model", len(SEEN) == n0 and u.fast_path
      and u.outcome == "execute")


def raises(name, fn, want):
    try:
        fn()
    except nl.LLMError as exc:
        check(name, want in str(exc), str(exc))
        return
    check(name, False, "no LLMError")


raises("a slow model times out", lambda: nl.understand(
    "walk 2 m", nl.make_backend("ollama", URL, "slow", 0.5), TRAINED, LIM), "within")
raises("HTTP 500 is reported with the server's words", lambda: nl.understand(
    "walk 2 m", nl.make_backend("ollama", URL, "broken", 5), TRAINED, LIM), "HTTP 500")
raises("a non-JSON reply is reported", lambda: nl.understand(
    "walk 2 m", nl.make_backend("ollama", URL, "garbage", 5), TRAINED, LIM), "not return JSON")
s = socket.socket()
s.bind(("127.0.0.1", 0))
dead = s.getsockname()[1]
s.close()
raises("nothing listening", lambda: nl.understand(
    "walk 2 m", nl.make_backend("ollama", "http://127.0.0.1:{}".format(dead), "m", 2),
    TRAINED, LIM), "could not reach")


print("\n== the ask command, end to end (dry run) ==")


def cli(argv):
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        code = N.main(argv)
    return code, buf.getvalue() + err.getvalue()


log = os.path.join(os.environ.get("TMPDIR", "/tmp"), "r1_ask_test_{}.jsonl".format(os.getpid()))
code, out = cli(["ask", "Walk forward 2 meters, then turn left.", "--dry-run",
                 "--llm-url", URL, "--model", "m", "--log", log])
check("ask --dry-run executes the parsed plan", code == 0 and "plan complete" in out, out[-600:])
check("the operator sees what was understood", "understood:" in out
      and "1. walk forward 2 m" in out, out)
check("the reply is the template, with its labels", "robot:" in out and "by my IMU" in out
      and "not measured" in out, out[-400:])
rec = json.loads(open(log).read().splitlines()[-1])
check("the log has the model's raw answer and the reply", rec["understood"]["raw"]
      and rec["reply"].endswith("Done.") and rec["exit"] == 0, rec.keys())
code, out = cli(["ask", "strafe left 2 meters.", "--dry-run", "--llm-url", URL,
                 "--model", "m", "--log", log])
check("a refused instruction exits 2 and moves nothing",
      code == 2 and "nothing was executed" in out and "plan (" not in out, out)
code, out = cli(["ask", "walk 2 m", "--llm-url", URL, "--model", "broken"])
check("a dead model exits 3 with the restart hint", code == 3
      and "ollama_ctl.sh restart" in out, out)
code, out = cli(["ask-check", "--llm-url", URL, "--model", "m"])
check("ask-check passes against a right answer", code == 0 and "OK:" in out, out)
check("ask-check on the shipped config names the GPU default and sends no placement",
      "Ollama's default (the GPU)" in out and "num_gpu" not in SEEN[-1]["options"], out)
# The health check in llm/ollama_ctl.sh is `ask-check --once`. It must send exactly
# what `ask` sends: Ollama reloads the model when a request asks for another
# placement (stage C: a bare health generate would have put a CPU-placed model back
# on the GPU, and the next ask would have moved it again).
cpu_cfg = os.path.join(os.environ.get("TMPDIR", "/tmp"), "r1_cpu_cfg_{}.yaml".format(os.getpid()))
with open(cpu_cfg, "w") as f:
    f.write("".join(line for line in open(str(N.DEFAULT_CFG))
                    if not line.startswith(("llm_num_gpu:", "llm_num_thread:")))
            + "llm_num_gpu: 0\nllm_num_thread: 4\n")
n0 = len(SEEN)
code, out = cli(["ask-check", "--once", "--config", cpu_cfg, "--llm-url", URL, "--model", "m"])
check("ask-check --once: one request, with mission.yaml's placement",
      code == 0 and len(SEEN) == n0 + 1 and SEEN[-1]["options"].get("num_gpu") == 0
      and SEEN[-1]["options"].get("num_thread") == 4
      and 'placement {"num_gpu": 0, "num_thread": 4}' in out, (out, SEEN[-1].get("options")))
health_req = SEEN[-1]
code, out = cli(["ask", "Walk forward 2 meters, then turn left.", "--dry-run",
                 "--config", cpu_cfg, "--llm-url", URL, "--model", "m"])
check("ask sends the health check's model, placement and prompt",
      code == 0 and SEEN[-1]["options"] == health_req["options"]
      and SEEN[-1]["model"] == health_req["model"]
      and SEEN[-1]["messages"][:-1] == health_req["messages"][:-1], SEEN[-1].get("options"))
os.remove(cpu_cfg)
code, out = cli(["ask-check", "--llm-url", "http://127.0.0.1:{}".format(dead)])
check("ask-check says when nothing is listening", code == 3 and "NOT REACHABLE" in out, out)
code, out = cli(["capability", "--json"])
cap = json.loads(out)
check("capability --json carries the LLM prompt id and schema",
      cap["llm"]["prompt_id"] == nl.prompt_id(LIM) and "anyOf" in json.dumps(cap["llm"]))
try:
    os.remove(log)
except OSError:
    pass

print("\n== the shipped eval sets agree with the shipped config ==")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
import run_nl_eval as E   # noqa: E402

env = N.resolve_envelope()
for name in ("nl_dev.jsonl", "nl_test.jsonl", "nl_test2.jsonl", "nl_test3.jsonl"):
    path = os.path.join(os.path.dirname(HERE), "eval", name)
    bad = E.check_cases(E.load_cases(path), env, LIM)
    check("{} is consistent".format(name), not bad, bad)

srv.shutdown()
print("\npass {}  fail {}".format(len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
