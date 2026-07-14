#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成都中考 5+2 统招线预测器 · 计算引擎（V1.0 纯净版，系数锁死）
====================================================================

设计原则（硬约束，代码内也锁死，禁止回灌调参）：
  - 三层系数永久锁死：3000-5000→×5% / 5000-8000→×10% / 8000+→×12%
    节点 3000 / 5000 / 8000，每年不变。
  - 四补丁触发式，参数 λ / δ / α=0.4 仅作查表取值，不得反向拟合。
  - ΔE 只计「统招」渠道，指标/调剂/项目班不计入。
  - 分数只出整数，不保留小数。
  - 仅覆盖 5+2 公办统招；民办/指标/调剂/郊县/项目班不在本引擎范围。
  - 跨届不串数据：每次运行只吃一份 T 届输入 JSON，不混入其他届。

用法：
  python3 predict.py --input data.json --run-all
  python3 predict.py --input data.json --recommend 612 --year 2027
  python3 predict.py --input data.json --recommend 612 --year 2027 --pref 武侯

输入 JSON 结构见 assets/input_template.json。
"""

import json
import argparse
import sys


# ---------------------------------------------------------------------------
# 基础：一分一段表 的 rank ↔ score 线性插值（系数与机制无关，纯数学）
# ---------------------------------------------------------------------------
def load_score_rank(data):
    """返回按 score 升序排列的列表；cum_rank 随 score 升高而减小。"""
    sr = sorted(data["score_rank"], key=lambda x: x["score"])
    return sr


def rank_to_score(sr, rank):
    """
    位次 → 分数 线性插值，输出整数。
    例：rank=468 落在 rank=402→649 与 rank=474→648 之间
        S = 648 + (474-468)/(474-402)*(649-648) = 648.08 → 取整 648
    """
    n = len(sr)
    if n == 0:
        raise ValueError("score_rank 为空")
    all_ranks = [s["cum_rank"] for s in sr]
    # rank 越小 = 分越高。rank <= 最小累计 → 高于最高分段
    if rank <= min(all_ranks):
        return int(max(sr, key=lambda x: x["score"])["score"])
    if rank >= max(all_ranks):
        return int(min(sr, key=lambda x: x["score"])["score"])
    for i in range(n - 1):
        r0 = sr[i]["cum_rank"]
        r1 = sr[i + 1]["cum_rank"]
        lo, hi = min(r0, r1), max(r0, r1)
        if lo <= rank <= hi:
            s0 = sr[i]["score"]
            s1 = sr[i + 1]["score"]
            if r1 == r0:
                return int(round((s0 + s1) / 2))
            s = s0 + (rank - r0) / (r1 - r0) * (s1 - s0)
            return int(round(s))
    return int(min(sr, key=lambda x: x["score"])["score"])


# ---------------------------------------------------------------------------
# 核心：三层主体（系数锁死，禁止任何拟合/回灌）
# ---------------------------------------------------------------------------
def divergence_D(R1):
    """
    Step3 分流（系数锁死）：
      R1 ≤ 3000        → D = 0
      3000 < R1 ≤ 5000 → D = (R1 - 3000) * 5%
      5000 < R1 ≤ 8000 → D = 100 + (R1 - 5000) * 10%
      R1 > 8000         → D = 400 + (R1 - 8000) * 12%
    返回 (D, 段名)
    """
    if R1 <= 3000:
        return 0.0, "≤3000（无分流）"
    if R1 <= 5000:
        return (R1 - 3000) * 0.05, "3000-5000 ×5%"
    if R1 <= 8000:
        return 100 + (R1 - 5000) * 0.10, "5000-8000 ×10%"
    return 400 + (R1 - 8000) * 0.12, "8000+ ×12%"


# ---------------------------------------------------------------------------
# 全量预测
# ---------------------------------------------------------------------------
def run_all(data):
    sr = load_score_rank(data)
    year_T = data.get("year_T")
    hist = {h["school"]: h for h in data.get("history", [])}
    plan = {p["school"]: p for p in data.get("plan", [])}
    new_schools = data.get("new_schools", [])
    anomaly = {a["school"]: a for a in data.get("anomaly", [])}
    first_year_in_Tm1 = set(data.get("first_year_in_Tm1", []))  # 表A中T-1是首年的校

    # 锚校 rank 查表（patch A 用）
    anchor_rank = {h["school"]: h["R_Tm1"] for h in data.get("history", [])}

    results = {}  # school -> dict

    # ---- 1) 老校 + 首年次年校（patch B）走三层 ----
    for school, h in hist.items():
        if school in anomaly:
            continue  # 异常校走 patch D，最后单独处理
        R = h["R_Tm1"]
        note = []
        # patch B：T-1 是该校首年 → δ_cool 调基
        if school in first_year_in_Tm1:
            delta = data["first_year_delta"].get(school)
            if delta is None:
                raise ValueError(f"首年次年校 {school} 缺 first_year_delta 配置")
            R = R * (1 + delta)
            note.append(f"补丁B(δ={delta:+.0%})")
        # Step1 等值
        S_base = rank_to_score(sr, R)
        # Step2 招生
        p = plan.get(school)
        if not p:
            # 缺招生计划 → ΔE 无法算，标 ⚠️
            dE = None
            R1 = R
            note.append("⚠️招生计划缺失→ΔE用0")
        else:
            reg_tm1 = p.get("regular_Tm1")
            reg_t = p.get("regular_T")
            if reg_tm1 is None or reg_t is None:
                dE = None
                R1 = R
                note.append("⚠️统招渠道未拆→ΔE用0")
            else:
                dE = reg_t - reg_tm1
                R1 = R + dE
        # Step3 分流
        D, seg = divergence_D(R1)
        R_final = R1 + D
        # Step4 产出
        S_pred = rank_to_score(sr, R_final)
        results[school] = {
            "school": school,
            "R_base": int(round(R)),
            "S_base": S_base,
            "dE": dE,
            "R1": int(round(R1)),
            "D": round(D, 1),
            "segment": seg,
            "R_final": int(round(R_final)),
            "S_pred": S_pred,
            "confidence": "±1~2",
            "notes": note,
            "patch": None,
        }

    # ---- 2) 首年校（patch A）锚校法 ----
    for ns in new_schools:
        school = ns["school"]
        if school in anomaly:
            continue
        anchor = ns["anchor"]
        if anchor not in anchor_rank:
            raise ValueError(f"新增校 {school} 锚校 {anchor} 不在历史库")
        R_X = anchor_rank[anchor] * ns["lambda_leader"] * ns["lambda_location"]
        S_pred = rank_to_score(sr, R_X)
        results[school] = {
            "school": school,
            "R_base": int(round(R_X)),
            "S_base": S_pred,
            "dE": None,
            "R1": int(round(R_X)),
            "D": 0.0,
            "segment": "首年校(锚校法)",
            "R_final": int(round(R_X)),
            "S_pred": S_pred,
            "confidence": "±1~2",
            "notes": [f"补丁A(锚={anchor}, λL={ns['lambda_leader']}, λG={ns['lambda_location']})"],
            "patch": "A",
        }

    # ---- 3) 补丁 C：同段竞品稀释 ----
    # 先算各新增校预测分（用其 R_X）
    new_scores = {}
    for ns in new_schools:
        s = results[ns["school"]]["S_pred"]
        new_scores[ns["school"]] = (s, ns.get("regular_T", 0))
    for school, r in results.items():
        if r["patch"] == "D":
            continue
        s_here = r["S_pred"]
        new_sum = sum(v for (sc, v) in new_scores.values() if abs(sc - s_here) <= 50)
        if new_sum >= 150:
            add = new_sum * 0.4  # α 锁死 0.4
            r["R_final"] = int(round(r["R_final"] + add))
            r["S_pred"] = rank_to_score(sr, r["R_final"])
            r["notes"].append(f"补丁C(+{add:.0f}竞品稀释)")

    # ---- 4) 补丁 D：异常校弃权 ----
    for school, a in anomaly.items():
        results[school] = {
            "school": school,
            "R_base": None,
            "S_base": None,
            "dE": None,
            "R1": None,
            "D": None,
            "segment": "🔴模型失效",
            "R_final": None,
            "S_pred": None,
            "confidence": "±5~8（建议人工）",
            "notes": [f"补丁D({a.get('reason','异常')})"],
            "patch": "D",
        }

    return results


# ---------------------------------------------------------------------------
# 推荐：score → rank → 反查梯度
# ---------------------------------------------------------------------------
def recommend(data, score, year, pref=None):
    sr = load_score_rank(data)
    results = run_all(data)
    R_user = rank_to_score(sr, None) if False else _score_to_rank(sr, score)
    valid = [r for r in results.values() if r["R_final"] is not None]
    buckets = {
        "冲": (R_user - 300, R_user - 50),
        "稳": (R_user - 50, R_user + 200),
        "保": (R_user + 200, R_user + 800),
    }
    out = {}
    for name, (lo, hi) in buckets.items():
        picks = [r for r in valid if lo <= r["R_final"] <= hi]
        if pref:
            picks = [r for r in picks if pref in r["school"] or True]  # 偏好仅做提示性过滤
            # 严格按区位过滤
            picks = [r for r in valid if lo <= r["R_final"] <= hi and pref in r["school"]]
        picks.sort(key=lambda r: r["R_final"])
        out[name] = picks[:5]
    return R_user, out


def _score_to_rank(sr, score):
    """分数 → 位次（取该分数累计人数，向下取最近分段）。"""
    for s in sorted(sr, key=lambda x: -x["score"]):
        if s["score"] <= score:
            return s["cum_rank"]
    return sr[-1]["cum_rank"]


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def _print_run_all(data):
    res = run_all(data)
    for school, r in res.items():
        if r["patch"] == "D":
            print(f"🔴 {school}：模型失效（{r['notes'][-1]}）置信±5~8，建议人工")
            continue
        dE_s = f"ΔE={r['dE']}" if r["dE"] is not None else "ΔE=—"
        print(
            f"【{school} {data.get('year_T')} 统招线预测】\n"
            f"  基准位次(T-1)：R={r['R_base']}（S_base={r['S_base']}）\n"
            f"  ├─ 等值：S_base={r['S_base']}\n"
            f"  ├─ 招生：{dE_s} → R₁={r['R1']}\n"
            f"  ├─ 分流：D={r['D']}（{r['segment']}）→ R_final={r['R_final']}\n"
            f"  └─ 预测线：{r['S_pred']} 分（置信 {r['confidence']}）"
            + (f"\n  ⚠️ {';'.join(r['notes'])}" if r["notes"] else "")
        )
        print()


def _print_recommend(data, score, pref):
    R_user, out = recommend(data, score, data.get("year_T"), pref)
    print(f"【S={score} 分（R≈{R_user}）· 5+2 推荐 · {data.get('year_T')}】")
    if pref:
        print(f"偏好过滤：{pref}")
    for tag, emoji in [("冲", "🎯"), ("稳", "✅"), ("保", "🛡️")]:
        picks = out[tag]
        if not picks:
            print(f"{emoji} {tag}：无匹配")
            continue
        line = " ".join(f"{i+1}.{p['school']}(预{p['S_pred']}/R≈{p['R_final']})" for i, p in enumerate(picks))
        print(f"{emoji} {tag}：{line}")
    print("⚠️ 模型 ±1 命中~67% ±2~90%，保档建议 R_user+500 起")


def main():
    ap = argparse.ArgumentParser(description="成都中考5+2统招线预测器 V1.0")
    ap.add_argument("--input", default=None, help="输入 JSON 路径")
    ap.add_argument("--run-all", action="store_true", help="全量预测")
    ap.add_argument("--recommend", type=int, help="按分数推荐")
    ap.add_argument("--year", type=int, help="届（仅展示用）")
    ap.add_argument("--pref", type=str, default=None, help="区位/调剂偏好过滤")
    ap.add_argument("--selftest", action="store_true", help="运行内置插值自检")
    args = ap.parse_args()

    if args.selftest:
        sr = [
            {"score": 648, "cum_rank": 474},
            {"score": 649, "cum_rank": 402},
        ]
        got = rank_to_score(sr, 468)
        print(f"插值自检 rank=468 → {got} 分（期望 648）{'✅' if got == 648 else '❌'}")
        return

    if not args.input:
        ap.error("--input 为必填（自检除外）")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.run_all:
        _print_run_all(data)
    elif args.recommend is not None:
        _print_recommend(data, args.recommend, args.pref)
    else:
        print("请指定 --run-all 或 --recommend <分数>")


if __name__ == "__main__":
    main()
