"""The gateway config: every Groq model exists once per API key, and a key that hits its limit hands the call to the
same model on the next key before anything falls back to a different model."""
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = tomllib.loads((ROOT / "tensorzero" / "tensorzero.toml").read_text(encoding="utf-8"))
KEYS = ["env::GROQ_API_KEY", "env::GROQ_API_KEY_2", "env::GROQ_API_KEY_3"]


def test_every_model_tries_each_key_on_the_same_model_before_falling_back():
    for name, model in CONFIG["models"].items():
        providers = model["providers"]
        assert set(model["routing"]) == set(providers), name
        served = [(providers[p]["model_name"], providers[p]["api_key_location"]) for p in model["routing"]]
        # routing comes in groups: one model on every key, then the next model on every key
        for i in range(0, len(served), len(KEYS)):
            group = served[i:i + len(KEYS)]
            assert len({m for m, _ in group}) == 1, (name, group)
            assert [k for _, k in group] == KEYS, (name, group)


def test_benchmark_writer_stays_on_one_model_across_keys():
    strict = CONFIG["models"]["research_model_strict"]
    assert {p["model_name"] for p in strict["providers"].values()} == {"openai/gpt-oss-120b"}


def test_every_key_the_gateway_reads_is_passed_to_the_task_and_checked_on_resume():
    terraform = (ROOT / "terraform" / "main.tf").read_text(encoding="utf-8")
    resume = (ROOT / "scripts" / "resume_safe.sh").read_text(encoding="utf-8")
    for key in (k.removeprefix("env::") for k in KEYS):
        assert re.search(rf'name = "{key}",\s*valueFrom = .*:{key}::"', terraform), key
        assert f"'{key}'" in resume, key
