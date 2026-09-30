# eval/ · the offline English eval for `ask`

`run_nl_eval.py` sends every case through exactly what `r1_mission_cli.py ask` runs
(normalize → model → translate → `plan_from_json`), and scores **what the robot would
do**. Results, method and the model choice: [docs/stageC_nl_eval.md](../../docs/stageC_nl_eval.md).

| file | cases | role |
|---|---|---|
| `nl_dev.jsonl` | 20 | prompt design, and nothing else |
| `nl_test.jsonl` | 40 | the plan's four categories; first scored in run 1 |
| `nl_test2.jsonl` | 20 | frozen before run 2 |
| `nl_test3.jsonl` | 20 | frozen before run 3 |

```bash
python3 run_nl_eval.py --check --set all                 # no model: every case vs the shipped config
python3 run_nl_eval.py --model qwen3:1.7b                # the 80 held-out cases (default --set heldout)
python3 run_nl_eval.py --model qwen3:1.7b --set dev -v   # prompt work; -v prints the wrong answers
python3 run_nl_eval.py --model qwen3:1.7b --options '{"num_gpu": 0}' --tag cpu   # CPU only
python3 run_nl_eval.py --table results/*.json            # markdown comparison
```

A case lists acceptable transcriptions (`expect`, a list of alternatives) and the
outcome they lead to under the shipped config (`outcome`). A case whose ideal answer the
deterministic layer cannot produce carries a `limit` note: it is scored as usual, so it
shows up as a miss, and `--check` skips it. **Do not tune the prompt against the test
sets.** `tests/test_nl.py` fails if the shipped prompt id is not the one the eval ran.
