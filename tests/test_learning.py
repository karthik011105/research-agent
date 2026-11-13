import random
from app.learning import (
    CriticTrust, DebatePolicy, WriterBandit, beta_mean, beta_update, report_reward, thompson_choice, win_probabilities,
)


def test_trust_moves_away_from_the_critic_that_is_wrong_and_stays_normalised():
    t = CriticTrust()
    for _ in range(10):
        t.update("number", {"fact_checker": True, "skeptic": False})     # skeptic keeps missing number errors
    ws = t.weights["number"]
    assert ws["fact_checker"] > 0.9 and abs(sum(ws.values()) - 1) < 1e-9
    assert ws["skeptic"] >= 0.05 / (1 + 0.05) - 1e-9                       # floor: can win trust back
    assert t.weights["date"] == {"fact_checker": 0.5, "skeptic": 0.5}       # other claim types untouched
    assert t.updates["number"] == 10


def test_trust_decides_a_disagreement_by_weight():
    t = CriticTrust()
    for _ in range(6):
        t.update("attribution", {"fact_checker": False, "skeptic": True})
    # skeptic says problem, fact-checker says supported: the trusted skeptic wins
    assert t.weighted_problem_score("attribution", {"fact_checker": False, "skeptic": True}) > 0.5
    assert t.weighted_problem_score("general", {"fact_checker": False, "skeptic": True}) == 0.5   # untrained


def test_trust_state_round_trips():
    t = CriticTrust()
    t.update("date", {"fact_checker": False, "skeptic": True})
    t2 = CriticTrust(t.state())
    assert t2.weights == t.weights and t2.updates["date"] == 1


def test_beta_helpers():
    arm = {}
    beta_update(arm, 1.0)
    beta_update(arm, 0.25)
    assert (arm["a"], arm["b"], arm["n"]) == (2.25, 1.75, 2) and round(beta_mean(arm), 4) == 0.5625
    beta_update(arm, 7)                              # clipped to 1
    assert arm["a"] == 3.25


def test_debate_policy_learns_when_debate_pays_off():
    p = DebatePolicy(call_cost=0.15)
    for _ in range(40):
        # numbers: debating fixes the verdict (wrong after 0 rounds, right after 1 or 2)
        p.learn("number", {0: False, 1: True, 2: True})
        # general claims: already right without debate, so rounds only cost
        p.learn("general", {0: True, 1: True, 2: True})
    assert p.best("number") == 1           # 1 round gets it right; a 2nd round only adds cost
    assert p.best("general") == 0
    rng = random.Random(3)
    picks = [p.choose("general", rng) for _ in range(200)]
    assert picks.count(0) > 150 and len(set(picks)) > 1          # mostly greedy, still explores


def test_report_reward_combines_signals_and_skips_missing_ones():
    assert report_reward({"claim_pass_rate": 1.0, "citation_validity": 1.0, "first_draft_ok": 1, "user_rating": 1}) == 1.0
    assert report_reward({"claim_pass_rate": 0.5}) == 0.5                      # only one signal known
    assert report_reward({"claim_pass_rate": 1.0, "user_rating": -1}) == round(0.4 / 0.6, 4)
    assert report_reward({"claim_pass_rate": 1.0, "latency_s": 600}) < 1.0    # slow reports lose a little
    assert report_reward({}) == 0.5


def test_writer_bandit_converges_to_the_best_variant_with_low_regret():
    rng = random.Random(11)
    true_quality = {"primary": 0.70, "writer_20b": 0.55, "cite_numbers": 0.80}
    bandit = WriterBandit(explore=0.1)
    regret, picks = 0.0, []
    for _ in range(400):
        v = bandit.choose(rng)
        picks.append(v)
        reward = 1.0 if rng.random() < true_quality[v] else 0.0
        bandit.learn(v, reward)
        regret += max(true_quality.values()) - true_quality[v]
    late = picks[-100:]
    assert late.count("cite_numbers") > 70                      # traffic shifted to the best writer
    assert regret < 0.25 * 400 * 0.1                            # far below always-random (avg gap ~0.13/step)
    summary = bandit.summary()
    assert max(summary["win_probability"], key=summary["win_probability"].get) == "cite_numbers"


def test_win_probabilities_and_exploration_floor():
    arms = {"a": {"a": 50, "b": 10}, "b": {"a": 10, "b": 50}}
    probs = win_probabilities(arms)
    assert probs["a"] > 0.99 and abs(sum(probs.values()) - 1) < 1e-9
    rng = random.Random(0)
    picks = [thompson_choice(arms, rng, explore=0.2) for _ in range(1000)]
    assert 60 < picks.count("b") < 160                          # ~10% from exploration only
