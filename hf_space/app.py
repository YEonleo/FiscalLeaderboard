import json
import re
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


APP_TITLE = "Fiscal Leaderboard"
PASS_K_OPTIONS = [1, 3, 5]

# Top 10 차트용 파스텔 색상 (진한 파스텔 톤)
TOP10_PASTEL_COLORS = [
    "#5eb8e8", "#6bcb7d", "#e6d84a", "#e8a54b", "#b88dd4",
    "#5eb5a6", "#e8957a", "#9d7bc9", "#6ba3d4", "#8bc34a",
]

# CPA 과목 목록
CPA_SUBJECTS = {
    "세법": "CPA 세법",
    "경제원론": "CPA 경제원론",
    "경영학": "CPA 경영학",
    "회계학": "CPA 회계학",
    "상법": "CPA 상법",
}

# 개정세법 연도 목록 (연도별 QA)
TAX_YEARS = [2021, 2022, 2023, 2025]

# 개정세법 객관식 500문항 연도 (tax500)
TAX500_YEARS = [2023, 2024, 2025]


def pass_label(k: int) -> str:
    return f"Pass@{k}"


def find_data_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, *here.parents]
    for base in candidates:
        if (base / "results_yearly" / "summary").exists() or (
            base / "results_cpa" / "summary"
        ).exists() or (base / "results_tax500" / "summary").exists():
            return base
    return here.parent


def normalize_answer(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    # 공백 제거 (일반 공백, 줄바꿈, 유니코드 공백 \u202f, \u200b 등)
    s = text.strip().lower()
    for c in (" ", "\n", "\r", "\t", "\u202f", "\u200b", "\u200c", "\u200d"):
        s = s.replace(c, "")
    # 마크다운 강조 제거 (gpt-oss 등 모델 출력에 ** 포함되는 경우)
    s = s.replace("*", "")
    return s


def extract_final_answer(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    if "최종정답:" in text:
        answer = text.split("최종정답:")[-1].strip()
        return answer.split("\n")[0].strip()
    return text.strip()


def _answer_match(gt_normalized: str, pred_normalized: str) -> bool:
    if not pred_normalized:
        return False
    if gt_normalized in pred_normalized or pred_normalized in gt_normalized:
        return True
    # 정답이 "~입니다." / "~입니다" 형태일 때, 핵심만으로도 매칭 (gpt-oss 등 형식 차이 대응)
    gt_core = gt_normalized.rstrip(".").removesuffix("입니다").removesuffix("입니다.")
    if gt_core and (gt_core in pred_normalized or pred_normalized in gt_core):
        return True
    return False


def compute_pass_for_row(row: pd.Series, k: int) -> bool:
    gt_normalized = normalize_answer(row.get("ground_truth", ""))
    if not gt_normalized:
        return False
    for i in range(1, k + 1):
        pred_col = f"prediction_{i}"
        pred_text = row.get(pred_col)
        if pred_text is None or pd.isna(pred_text):
            continue
        pred_final = extract_final_answer(pred_text)
        pred_normalized = normalize_answer(pred_final)
        if _answer_match(gt_normalized, pred_normalized):
            return True
    return False


def matching_samples(row: pd.Series, k: int) -> list[int]:
    matches = []
    gt_normalized = normalize_answer(row.get("ground_truth", ""))
    if not gt_normalized:
        return matches
    for i in range(1, k + 1):
        pred_col = f"prediction_{i}"
        pred_text = row.get(pred_col)
        if pred_text is None or pd.isna(pred_text):
            continue
        pred_final = extract_final_answer(pred_text)
        pred_normalized = normalize_answer(pred_final)
        if _answer_match(gt_normalized, pred_normalized):
            matches.append(i)
    return matches


def parse_safe_name(filename: str) -> tuple[str, int | None]:
    base = filename
    if base.endswith("_summary.csv"):
        base = base[: -len("_summary.csv")]
    if base.endswith("_cpa"):
        return base[: -len("_cpa")], None
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return base, None


def parse_safe_name_raw_evaluated(filename: str) -> tuple[str, int | None]:
    """예: openai_gpt-oss-120b_2021_raw_evaluated.csv -> (openai_gpt-oss-120b, 2021)"""
    if not filename.endswith("_raw_evaluated.csv"):
        return filename, None
    base = filename[: -len("_raw_evaluated.csv")]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return base, None


def parse_safe_name_tax500(filename: str) -> tuple[str, int | None]:
    """예: Qwen_Qwen2.5-7B-Instruct_tax500_2023_summary.csv -> (Qwen_Qwen2.5-7B-Instruct, 2023)"""
    if not filename.endswith("_summary.csv") or "_tax500_" not in filename:
        return filename, None
    base = filename[: -len("_summary.csv")]
    if "_tax500_" in base:
        head, tail = base.split("_tax500_", 1)
        if tail.isdigit():
            return head, int(tail)
    return base, None


def parse_year(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group(0))
    match = re.search(r"(\d{2})\D*$", text)
    if match:
        yy = int(match.group(1))
        return 2000 + yy if yy <= 30 else 1900 + yy
    return None


@st.cache_data(show_spinner=False)
def load_metadata(data_root: str) -> dict:
    path = Path(data_root) / "model_metadata.json"
    if not path.exists():
        return {"models": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def attach_metadata(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    if df.empty:
        return df
    models = metadata.get("models", [])
    meta_by_id = {m.get("model_id"): m for m in models}
    meta_by_safe = {m.get("safe_name"): m for m in models}

    def pick(row: pd.Series, key: str):
        meta = meta_by_id.get(row.get("model")) or meta_by_safe.get(row.get("safe_name"))
        return meta.get(key) if meta else None

    df = df.copy()
    for key in [
        "model_id",
        "params_b",
        "params_note",
        "korean_pretrained",
        "organization",
        "note",
        "safe_name",
    ]:
        if key == "model_id":
            df["model_id"] = df.get("model")
            continue
        df[key] = df.apply(lambda r, k=key: pick(r, k), axis=1)

    df["params_b"] = pd.to_numeric(df["params_b"], errors="coerce")
    return df


def _aggregate_raw_evaluated(df_raw: pd.DataFrame) -> pd.DataFrame:
    """raw_evaluated CSV (question_id, sample_id, Judge_Score) -> 문항당 1행, pass_at_1/3/5 from Judge_Score.
    주관식 오답분석용으로 prediction(모델답변), Judge_Reason(판별 이유) 포함."""
    if df_raw.empty or "Judge_Score" not in df_raw.columns:
        return pd.DataFrame()
    # sample_id가 없으면 1로 간주 (1샘플만 있는 경우)
    if "sample_id" not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["sample_id"] = 1
    pass_map = df_raw["Judge_Score"].str.strip().str.lower().isin(("pass",))
    df_raw = df_raw.copy()
    df_raw["_is_pass"] = pass_map

    rows = []
    for (qid, model, year), grp in df_raw.groupby(
        ["question_id", "model", "target_year"], dropna=False
    ):
        grp = grp.sort_values("sample_id")
        samples = grp["_is_pass"].tolist()
        pass_at_1 = samples[0] if len(samples) >= 1 else False
        pass_at_3 = any(samples[:3]) if len(samples) >= 3 else any(samples)
        pass_at_5 = any(samples[:5]) if len(samples) >= 5 else any(samples)
        first = grp.iloc[0]
        row = {
            "question_id": qid,
            "model": first["model"],
            "target_year": year,
            "instruction": first.get("instruction", ""),
            "ground_truth": first.get("ground_truth", ""),
            "pass_at_1": bool(pass_at_1),
            "pass_at_3": bool(pass_at_3),
            "pass_at_5": bool(pass_at_5),
        }
        if "prediction" in first.index:
            row["prediction"] = first.get("prediction", "")
        if "Judge_Score" in first.index:
            row["Judge_Score"] = first.get("Judge_Score", "")
        for reason_col in ("Judge_Reason", "Judge_Feedback", "judge_reason", "judge_feedback"):
            if reason_col in first.index and pd.notna(first.get(reason_col)):
                row["Judge_Reason"] = first.get(reason_col, "")
                break
        else:
            row["Judge_Reason"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_yearly(data_root: str) -> pd.DataFrame:
    """
    개정세법(연도별) 데이터 로드.
    평가 방식: 오직 raw/*_raw_evaluated.csv 의 Judge_Score(Pass/Fail) 기준만 사용합니다.
    - results_yearly/raw/*_raw_evaluated.csv 가 있는 (모델, 연도)만 리더보드에 포함됩니다.
    - summary는 사용하지 않습니다. (evaluated가 없는 모델·연도는 개정세법 점수에 포함되지 않음)
    """
    root = Path(data_root)
    raw_dir = root / "results_yearly" / "raw"
    frames = []

    if not raw_dir.exists():
        return pd.DataFrame()

    for file in sorted(raw_dir.glob("*_raw_evaluated.csv")):
        safe_name, year = parse_safe_name_raw_evaluated(file.name)
        df_raw = pd.read_csv(file)
        agg = _aggregate_raw_evaluated(df_raw)
        if agg.empty:
            continue
        agg["file"] = file.name
        agg["safe_name"] = safe_name
        agg["year_from_file"] = year
        frames.append(agg)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "target_year" in df.columns:
        df["target_year"] = df["target_year"].apply(parse_year)
    return df


@st.cache_data(show_spinner=False)
def load_cpa(data_root: str) -> pd.DataFrame:
    data_dir = Path(data_root) / "results_cpa" / "summary"
    if not data_dir.exists():
        return pd.DataFrame()
    frames = []
    for file in sorted(data_dir.glob("*_summary.csv")):
        df = pd.read_csv(file)
        safe_name, _ = parse_safe_name(file.name)
        df["file"] = file.name
        df["safe_name"] = safe_name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "year" in df.columns:
        df["year"] = df["year"].apply(parse_year)
    for k in PASS_K_OPTIONS:
        col = f"pass_at_{k}"
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_tax500(data_root: str) -> pd.DataFrame:
    """
    개정세법 객관식 500문항(tax500) 결과 로드.
    results_tax500/summary/*_tax500_*_summary.csv 기준.
    """
    data_dir = Path(data_root) / "results_tax500" / "summary"
    if not data_dir.exists():
        return pd.DataFrame()
    frames = []
    for file in sorted(data_dir.glob("*_tax500_*_summary.csv")):
        df = pd.read_csv(file)
        safe_name, year_from_file = parse_safe_name_tax500(file.name)
        df["file"] = file.name
        df["safe_name"] = safe_name
        if "target_year" not in df.columns and year_from_file is not None:
            df["target_year"] = year_from_file
        df["target_year"] = df["target_year"].apply(parse_year)
        # pass_at_3이 없으면 extracted_1~3 vs ground_truth로 계산
        if "pass_at_3" not in df.columns:
            gt = df["ground_truth"].astype(str).str.strip().str.upper()
            e1 = df.get("extracted_1", pd.Series([""] * len(df))).astype(str).str.strip().str.upper()
            e2 = df.get("extracted_2", pd.Series([""] * len(df))).astype(str).str.strip().str.upper()
            e3 = df.get("extracted_3", pd.Series([""] * len(df))).astype(str).str.strip().str.upper()
            df["pass_at_3"] = (e1 == gt) | (e2 == gt) | (e3 == gt)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for k in PASS_K_OPTIONS:
        col = f"pass_at_{k}"
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def shorten(text: str, limit: int = 140) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "you",
    "are",
    "but",
    "또한",
    "그리고",
    "또는",
    "있다",
    "없다",
    "한다",
    "합니다",
    "경우",
    "대한",
    "관련",
    "해당",
    "정답",
    "답",
}


def top_keywords(texts: list[str], top_n: int = 15) -> pd.DataFrame:
    counter = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        tokens = re.findall(r"[A-Za-z]+|[0-9]+|[가-힣]+", text)
        for token in tokens:
            t = token.lower()
            if t in STOPWORDS or len(t) < 2:
                continue
            if t.isdigit():
                continue
            counter[t] += 1
    if not counter:
        return pd.DataFrame(columns=["keyword", "count"])
    data = counter.most_common(top_n)
    return pd.DataFrame(data, columns=["keyword", "count"])


def styled_table(df: pd.DataFrame) -> pd.DataFrame:
    if "pass_rate" in df.columns:
        df = df.copy()
        df["pass_rate"] = df["pass_rate"].round(2)
    return df


def render_metric_cards(title: str, items: list[tuple[str, str]]):
    st.markdown(f"**{title}**")
    cards = ["<div class='metric-grid'>"]
    for label, value in items:
        cards.append(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div>"
            "</div>"
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_aihub_shell():
    st.markdown(
        """
        <div class="info-notice">
            <span class="notice-icon">ℹ️</span>
            <span class="notice-text">
                Fiscal Leaderboard는 개정세법(연도별 QA·<strong>개정세법 객관식 500문항</strong>) 및 CPA 시험 데이터를 활용하여 
                한국어 LLM의 세법·회계 전문 지식 수준을 평가하는 벤치마크 플랫폼입니다.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_model_cards(leaderboard: pd.DataFrame, pass_k: int, top_n: int = 5):
    if leaderboard.empty:
        return
    top_df = leaderboard.head(top_n).copy()
    palette = [
        "#38bdf8",
        "#a3e635",
        "#facc15",
        "#fb7185",
        "#a78bfa",
        "#22c55e",
        "#f97316",
        "#60a5fa",
    ]
    cards = ["<div class='model-grid'>"]
    for i, (_, row) in enumerate(top_df.iterrows()):
        rank = int(row.get("rank", 0))
        score = row.get(f"pass_rate_{pass_k}", row.get("score", 0))
        if pd.isna(score):
            score = 0.0
        model = row.get("display_model", "")
        org = row.get("organization") or "-"
        params = row.get("params_note") or row.get("params_b") or "-"
        p1 = row.get("pass_rate_1")
        p3 = row.get("pass_rate_3")
        p5 = row.get("pass_rate_5")
        accent = palette[i % len(palette)]
        chips = []
        if pd.notna(p1):
            chips.append(
                f"<span class='stat-chip' style='--chip:#e0f2fe;'>P@1 {p1:.3f}</span>"
            )
        if pd.notna(p3):
            chips.append(
                f"<span class='stat-chip' style='--chip:#fef9c3;'>P@3 {p3:.3f}</span>"
            )
        if pd.notna(p5):
            chips.append(
                f"<span class='stat-chip' style='--chip:#ffe4e6;'>P@5 {p5:.3f}</span>"
            )
        chips_html = "".join(chips)
        cards.append(
            f"<div class='model-card' style='--accent: {accent};'>"
            f"<div class='model-rank'>Rank {rank}</div>"
            f"<div class='model-name'>{model}</div>"
            f"<div class='metric-value'>{score:.3f}</div>"
            f"<div class='stat-row'>{chips_html}</div>"
            f"<div class='model-meta'>플랫폼: {org}</div>"
            f"<div class='model-meta'>파라미터: {params}</div>"
            "</div>"
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_header():
    st.markdown(
        """
        <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
        
        /* 기본 폰트: Pretendard Bold */
        html, body, [class*="css"] {
            font-family: "Pretendard", "Pretendard Variable", -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 700;
            color: #1a1a1a;
        }
        
        /* 전체 배경 및 메인 스크롤 영역 중앙 정렬 */
        [data-testid="stAppViewContainer"] {
            background: #f8f9fa;
        }
        [data-testid="stAppViewContainer"] > section > div {
            max-width: 100%;
            margin-left: auto;
            margin-right: auto;
        }
        
        /* 메인 컨테이너 - 가로 폭 제한, 중앙 정렬 */
        .block-container {
            padding: 2rem 2rem 3rem 2rem;
            max-width: 1100px;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            box-sizing: border-box;
        }
        /* Top 10 / 리더보드 영역: 컬럼 행 중앙 정렬 */
        [data-testid="stHorizontalBlock"] {
            display: flex;
            justify-content: center;
            align-items: stretch;
            gap: 1rem;
            flex-wrap: wrap;
        }
        [data-testid="stHorizontalBlock"] [data-testid="column"] {
            min-width: 0;
        }
        /* 차트 2열 행: 각 컬럼 최대 너비 제한해 블록 전체가 중앙에 오도록 */
        [data-testid="stHorizontalBlock"]:has(div[data-testid="stVegaLiteChart"]) [data-testid="column"] {
            flex: 0 1 420px;
            max-width: 100%;
        }
        /* 차트 컨테이너 자체도 중앙 정렬 */
        div[data-testid="stVegaLiteChart"] {
            margin-left: auto;
            margin-right: auto;
        }
        
        /* Hero 섹션 - AI허브 스타일 */
        .hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.25);
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.02em;
        }
        .hero p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.05rem;
            margin: 0;
            line-height: 1.6;
        }
        
        /* 네비게이션 - AI허브 스타일 */
        .aihub-nav {
            background: #ffffff;
            border-radius: 12px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }
        .nav-links {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .nav-link {
            text-decoration: none;
            padding: 0.5rem 1.2rem;
            border-radius: 8px;
            color: #495057;
            background: #f8f9fa;
            font-size: 0.95rem;
            font-weight: 500;
            transition: all 0.2s ease;
            border: 1px solid #e9ecef;
        }
        .nav-link.active {
            background: #667eea;
            color: #ffffff;
            border-color: #667eea;
            font-weight: 600;
        }
        .nav-link:hover {
            background: #e9ecef;
        }
        .nav-link.active:hover {
            background: #5568d3;
        }
        
        /* 알림 박스 */
        .info-notice {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 0.8rem;
        }
        .notice-icon {
            font-size: 1.2rem;
            flex-shrink: 0;
        }
        .notice-text {
            color: #1565c0;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* 메트릭 카드 */
        div[data-testid="stMetric"] {
            background: #ffffff;
            padding: 1.2rem 1.4rem;
            border-radius: 12px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="stMetric"] label {
            font-size: 0.85rem !important;
            color: #6c757d !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #667eea !important;
            font-weight: 700 !important;
        }
        
        /* 차트 컨테이너 - 블록 침범 방지, y축 공간 상쇄로 시각적 중앙 정렬 */
        div[data-testid="stVegaLiteChart"] {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e9ecef;
            padding: 1rem 1.25rem 1rem 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            max-width: 100%;
            overflow: hidden;
            box-sizing: border-box;
        }
        div[data-testid="stVegaLiteChart"] > div {
            max-width: 100% !important;
            overflow: hidden !important;
        }
        div[data-testid="stVegaLiteChart"] svg {
            max-width: 100% !important;
            height: auto !important;
        }
        /* 컬럼 내 차트 래퍼 */
        [data-testid="column"] div[data-testid="stVegaLiteChart"] {
            width: 100%;
        }
        
        /* 데이터프레임 */
        div[data-testid="stDataFrame"] {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e9ecef;
            padding: 0.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }
        
        /* 섹션 헤더 - 중앙 정렬 */
        h3 {
            color: #2d3436;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-size: 1.4rem;
            text-align: center;
        }
        
        /* 캡션 */
        .caption, [data-testid="stCaptionContainer"] {
            color: #6c757d;
            font-size: 0.85rem;
            line-height: 1.5;
        }
        
        /* 입력 필드 */
        input[type="text"] {
            border-radius: 8px !important;
            border: 1px solid #dee2e6 !important;
            padding: 0.6rem 1rem !important;
        }
        input[type="text"]:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* 탭 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 0.3rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        }
        
        /* 컬럼 내 콘텐츠 오버플로우 방지 */
        [data-testid="column"] {
            min-width: 0;
            overflow: hidden;
        }
        [data-testid="column"] > div {
            max-width: 100%;
            min-width: 0;
        }
        
        /* 반응형 - 큰 화면에서도 가로 제한 유지 */
        @media (min-width: 1400px) {
            .block-container {
                max-width: 1200px;
            }
        }
        </style>
        <div class="hero">
            <h1>🏛️ Fiscal Leaderboard</h1>
            <p>한국어 LLM의 세법 및 회계 전문 지식을 평가하는 벤치마크 플랫폼</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_leaderboard(df: pd.DataFrame, pass_k: int) -> pd.DataFrame:
    pass_col = f"pass_at_{pass_k}"
    if df.empty or pass_col not in df.columns:
        return pd.DataFrame()
    grouped = build_leaderboard_all(df)
    if grouped.empty:
        return grouped
    pass_rate_col = f"pass_rate_{pass_k}"
    grouped["pass_rate"] = grouped[pass_rate_col] * 100
    grouped["display_model"] = grouped["model"].fillna(grouped["safe_name"])
    grouped = grouped.sort_values("pass_rate", ascending=False)
    return grouped


def build_leaderboard_all(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg_dict = dict(
        questions=("question_id", "count"),
        params_b=("params_b", "first"),
        params_note=("params_note", "first"),
        organization=("organization", "first"),
        korean_pretrained=("korean_pretrained", "first"),
        note=("note", "first"),
        safe_name=("safe_name", "first"),
    )
    for k in PASS_K_OPTIONS:
        col = f"pass_at_{k}"
        if col in df.columns:
            agg_dict[f"pass_rate_{k}"] = (col, "mean")
    grouped = df.groupby("model", dropna=False).agg(**agg_dict).reset_index()
    return grouped


def metric_series(df: pd.DataFrame, pass_k: int, **filters) -> pd.Series:
    subset = df
    for key, value in filters.items():
        subset = subset[subset[key] == value]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.groupby("model")[f"pass_at_{pass_k}"].mean()


def build_metric_table(
    yearly_df: pd.DataFrame,
    cpa_df: pd.DataFrame,
    tax500_df: pd.DataFrame,
    metadata: dict,
    pass_k: int,
    include_yearly_tax: bool = False,
) -> pd.DataFrame:
    metrics = {}

    tax_year_cols = []
    tax500_year_cols = []
    cpa_year_cols = []
    cpa_subject_cols = []

    # 옵션 시 개정세법(주관식·연도별) 포함
    if include_yearly_tax and not yearly_df.empty:
        metrics["개정세법 종합"] = metric_series(yearly_df, pass_k)
        for year in sorted(yearly_df["target_year"].dropna().unique()):
            col = f"개정세법_{int(year)}"
            metrics[col] = metric_series(yearly_df, pass_k, target_year=year)
            tax_year_cols.append(col)

    if not tax500_df.empty:
        metrics["개정세법 객관식 종합"] = metric_series(tax500_df, pass_k)
        for year in sorted(tax500_df["target_year"].dropna().unique()):
            col = f"개정세법 객관식_{int(year)}"
            metrics[col] = metric_series(tax500_df, pass_k, target_year=year)
            tax500_year_cols.append(col)

    if not cpa_df.empty:
        metrics["CPA 종합"] = metric_series(cpa_df, pass_k)
        for year in sorted(cpa_df["year"].dropna().unique()):
            col = f"CPA_{int(year)}"
            metrics[col] = metric_series(cpa_df, pass_k, year=year)
            cpa_year_cols.append(col)
        for subject in sorted(cpa_df["subject"].dropna().unique()):
            col = f"CPA_{subject}"
            metrics[col] = metric_series(cpa_df, pass_k, subject=subject)
            cpa_subject_cols.append(col)

    if not metrics:
        return pd.DataFrame()

    metric_df = pd.concat(metrics, axis=1)
    metric_df.index.name = "model"
    metric_df = metric_df.reset_index()

    meta_cols = [
        "model",
        "safe_name",
        "organization",
        "params_b",
        "params_note",
        "korean_pretrained",
    ]
    base_parts = []
    if not yearly_df.empty:
        b = yearly_df[[c for c in meta_cols if c in yearly_df.columns]].drop_duplicates()
        for c in meta_cols:
            if c not in b.columns:
                b[c] = None
        base_parts.append(b)
    if not cpa_df.empty:
        b = cpa_df[[c for c in meta_cols if c in cpa_df.columns]].drop_duplicates()
        for c in meta_cols:
            if c not in b.columns:
                b[c] = None
        base_parts.append(b)
    if not tax500_df.empty:
        t5 = tax500_df[["model", "safe_name"]].drop_duplicates()
        for c in meta_cols:
            if c not in t5.columns:
                t5[c] = None
        base_parts.append(t5)
    base = pd.concat(base_parts, ignore_index=True) if base_parts else pd.DataFrame()
    if base.empty:
        return pd.DataFrame()
    base = base.dropna(subset=["model"]).drop_duplicates(subset=["model"])
    base = attach_metadata(base, metadata)
    base = (
        base.groupby("model", dropna=False)
        .agg(
            safe_name=("safe_name", "first"),
            organization=("organization", "first"),
            params_b=("params_b", "first"),
            params_note=("params_note", "first"),
            korean_pretrained=("korean_pretrained", "first"),
        )
        .reset_index()
    )

    metric_df = metric_df.merge(base, on="model", how="left")
    metric_df["display_model"] = metric_df["model"].fillna(metric_df["safe_name"])

    # 전체 모델 성능 종합: 가중 평균. None/NaN은 0점. (객관식 0.6배, 주관식 포함 시 1.0배)
    WEIGHT_CPA = 1.0
    WEIGHT_TAX500 = 0.6
    WEIGHT_YEARLY = 1.0
    numer = pd.Series(0.0, index=metric_df.index)
    total_weight = 0.0
    if "CPA 종합" in metric_df.columns:
        numer = numer + metric_df["CPA 종합"].fillna(0) * WEIGHT_CPA
        total_weight += WEIGHT_CPA
    if "개정세법 객관식 종합" in metric_df.columns:
        numer = numer + metric_df["개정세법 객관식 종합"].fillna(0) * WEIGHT_TAX500
        total_weight += WEIGHT_TAX500
    if include_yearly_tax and "개정세법 종합" in metric_df.columns:
        numer = numer + metric_df["개정세법 종합"].fillna(0) * WEIGHT_YEARLY
        total_weight += WEIGHT_YEARLY
    metric_df["전체 모델 성능 종합"] = (
        numer / total_weight if total_weight > 0 else pd.Series(float("nan"), index=metric_df.index)
    )

    metric_order = []
    for col in ["전체 모델 성능 종합", "CPA 종합", "개정세법 종합", "개정세법 객관식 종합"]:
        if col in metric_df.columns:
            metric_order.append(col)
    metric_order.extend(tax_year_cols)
    metric_order.extend(tax500_year_cols)
    metric_order.extend(cpa_year_cols)
    metric_order.extend(cpa_subject_cols)
    metric_df.attrs["metric_order"] = metric_order
    return metric_df


def pass_heatmap(df: pd.DataFrame, pass_k: int) -> pd.DataFrame:
    pass_col = f"pass_at_{pass_k}"
    if df.empty or pass_col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(["model", "target_year"], dropna=False)[pass_col]
        .mean()
        .reset_index()
        .assign(pass_rate=lambda d: d[pass_col] * 100)
    )


def make_top10_chart(
    df: pd.DataFrame, pass_k: int, title: str, height: int = 220
) -> alt.Chart | None:
    leaderboard = build_leaderboard(df, pass_k)
    if leaderboard.empty:
        return None
    top_df = leaderboard.head(10).copy()
    top_df["score"] = top_df["pass_rate"] / 100
    top_df["label"] = top_df["display_model"].apply(
        lambda x: (x[:12] + "…") if isinstance(x, str) and len(x) > 13 else x
    )
    # 툴팁 첫 줄에 필드명 없이 모델명만 보이게 (제로폭 스페이스 컬럼 사용)
    _z = "\u200b"
    top_df[_z] = top_df["display_model"]
    hover = alt.selection_point(fields=["display_model"], on="mouseover", empty="none")
    bars = (
        alt.Chart(top_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X(
                "label:N",
                sort="-y",
                title="",
                scale=alt.Scale(paddingInner=0.28),
                axis=alt.Axis(
                    labelAngle=-25,
                    labelLimit=90,
                    labelFontSize=10,
                    labelColor="#000000",
                    labelFontWeight="bold",
                ),
            ),
            y=alt.Y("score:Q", title="", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "display_model:N",
                legend=None,
                scale=alt.Scale(range=TOP10_PASTEL_COLORS),
            ),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.7)),
            tooltip=[
                alt.Tooltip(f"{_z}:N", title=""),
                alt.Tooltip("score:Q", format=".3f", title="점수"),
            ],
        )
        .add_params(hover)
    )
    text = (
        alt.Chart(top_df)
        .mark_text(dy=-6, color="#374151", fontSize=11, align="center")
        .encode(
            x=alt.X("label:N", sort="-y"),
            y=alt.Y("score:Q"),
            text=alt.Text("score:Q", format=".3f"),
        )
    )
    chart = (bars + text).properties(
        height=height,
        width=400,
        title=title,
        padding={"left": 10, "right": 42, "top": 10, "bottom": 10},
    )
    return (
        chart.configure_axis(gridOpacity=0.4, gridDash=[2, 2])
        .configure_view(strokeWidth=0)
        .configure_title(align="center", anchor="middle")
    )


def make_metric_top10_chart(
    metric_df: pd.DataFrame, metric_col: str, title: str, height: int = 300
) -> alt.Chart | None:
    if metric_df.empty or metric_col not in metric_df.columns:
        return None
    df = metric_df[["display_model", metric_col]].dropna()
    if df.empty:
        return None
    df = df.sort_values(metric_col, ascending=False).head(10).copy()
    df["score"] = df[metric_col]
    
    def smart_label(name: str) -> str:
        """모델명을 간결하게 표시하되 구분 가능하게"""
        if not isinstance(name, str):
            return ""
        # 전체 모델 ID 사용
        parts = name.split("/")
        if len(parts) == 2:
            # org/model 형태
            org, model_name = parts
            # 모델명에서 주요 정보 추출
            if "Qwen" in model_name:
                # Qwen2.5-7B, Qwen3-8B 등 버전과 크기 모두 표시
                match = re.search(r"(Qwen\d+(?:\.\d+)?)-?(\d+\.?\d*B)", model_name)
                if match:
                    return f"{match.group(1)} {match.group(2)}"
                return model_name[:20]
            elif "EXAONE" in model_name:
                # EXAONE-4.0-32B 형태
                match = re.search(r"(EXAONE-[\d.]+)-?(\d+\.?\d*B)", model_name)
                if match:
                    return f"{match.group(1)} {match.group(2)}"
                return model_name[:20]
            else:
                # 기타 모델
                return model_name[:18]
        else:
            # org가 없는 경우
            return name[:18]
    
    df["label"] = df["display_model"].apply(smart_label)
    # 툴팁 첫 줄에 필드명 없이 모델명만 보이게 (제로폭 스페이스 컬럼 사용)
    _z = "\u200b"
    df[_z] = df["display_model"]
    hover = alt.selection_point(fields=["display_model"], on="mouseover", empty="none")
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X(
                "label:N",
                sort="-y",
                title="",
                scale=alt.Scale(paddingInner=0.28),
                axis=alt.Axis(
                    labelAngle=-35,
                    labelLimit=140,
                    labelFontSize=9.5,
                    labelOverlap=False,
                    labelColor="#000000",
                    labelFontWeight="bold",
                ),
            ),
            y=alt.Y("score:Q", title="", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "display_model:N",
                legend=None,
                scale=alt.Scale(range=TOP10_PASTEL_COLORS),
            ),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.7)),
            tooltip=[
                alt.Tooltip(f"{_z}:N", title=""),
                alt.Tooltip("score:Q", format=".3f", title="점수"),
            ],
        )
        .add_params(hover)
    )
    text = (
        alt.Chart(df)
        .mark_text(dy=-6, color="#374151", fontSize=11, align="center")
        .encode(
            x=alt.X("label:N", sort="-y"),
            y=alt.Y("score:Q"),
            text=alt.Text("score:Q", format=".3f"),
        )
    )
    chart_width = 380
    # y축 레이블이 왼쪽 공간을 차지하므로 오른쪽 패딩을 넉넉히 해서 시각적 중앙 정렬
    chart = (bars + text).properties(
        height=height,
        width=chart_width,
        title=title,
        padding={"left": 10, "right": 42, "top": 10, "bottom": 10},
    )
    return (
        chart.configure_axis(gridOpacity=0.4, gridDash=[2, 2])
        .configure_view(strokeWidth=0)
        .configure_title(align="center", anchor="middle")
    )


def render_leaderboard_section(
    yearly_df: pd.DataFrame,
    cpa_df: pd.DataFrame,
    tax500_df: pd.DataFrame,
    pass_k: int,
    metadata: dict,
    include_yearly_tax: bool = False,
):
    metric_table = build_metric_table(
        yearly_df, cpa_df, tax500_df, metadata, pass_k, include_yearly_tax
    )
    if metric_table.empty:
        st.info("표시할 리더보드 데이터가 없습니다.")
        return

    metric_order = metric_table.attrs.get("metric_order", [])
    primary_metric = (
        "전체 모델 성능 종합"
        if "전체 모델 성능 종합" in metric_table.columns
        else (metric_order[0] if metric_order else None)
    )

    if not primary_metric:
        st.warning("평가 지표가 없습니다.")
        return

    table_sorted = metric_table.sort_values(primary_metric, ascending=False)
    table_sorted["rank"] = range(1, len(table_sorted) + 1)
    top_model = table_sorted.iloc[0]
    avg_score = table_sorted[primary_metric].mean()
    total_models = table_sorted.shape[0]
    total_questions = int(len(yearly_df) + len(cpa_df) + len(tax500_df))

    # 상단 통계 카드
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("모델 수", f"{total_models}")
    col2.metric("문항 수", f"{total_questions:,}")
    col3.metric("평균 점수", f"{avg_score:.3f}")
    col4.metric("Top Model", f"{top_model['display_model']}")

    # Top 10 차트 섹션
    st.markdown("### 📊 Top 10 모델")
    chart_height = st.session_state.get("lb_chart_height", 300)

    main_metrics = [
        col
        for col in [
            "전체 모델 성능 종합",
            "CPA 종합",
            "개정세법 종합",
            "개정세법 객관식 종합",
        ]
        if col in metric_table.columns
    ]
    metric_charts = []
    for col in main_metrics:
        chart = make_metric_top10_chart(
            table_sorted, col, col, height=chart_height
        )
        if chart:
            metric_charts.append(chart)

    # 2열로 차트 배치 (차트가 1개여도 반칸만 차지)
    if metric_charts:
        for i in range(0, len(metric_charts), 2):
            cols = st.columns(2)
            if i + 1 < len(metric_charts):
                with cols[0]:
                    st.altair_chart(metric_charts[i], width="stretch")
                with cols[1]:
                    st.altair_chart(metric_charts[i + 1], width="stretch")
            else:
                # 차트가 1개만 있어도 왼쪽 반칸만 차지
                with cols[0]:
                    st.altair_chart(metric_charts[i], width="stretch")

    # 리더보드 테이블 (행 hover 시 회색 음영)
    st.markdown("### 📋 전체 리더보드")
    st.markdown(
        """
        <style>
        div[data-testid="stDataFrame"] tbody tr:hover {
            background-color: #eef0f2 !important;
        }
        div[data-testid="stDataFrame"] tbody tr {
            transition: background-color 0.15s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    table_search = st.text_input(
        "🔍 모델 검색", 
        key="table_search", 
        placeholder="모델명으로 검색하세요"
    )

    table_df = table_sorted.copy()
    if table_search:
        mask = table_df["display_model"].str.contains(
            table_search, case=False, na=False
        )
        table_df = table_df[mask]

    # 테이블 컬럼 구성: 순위, 모델명, 각 세부 점수
    table_display = table_df.copy()
    table_display["순위"] = range(1, len(table_display) + 1)
    table_display["모델명"] = table_display["display_model"]
    
    # 테이블 컬럼: 순위, 모델명, 세부 점수들
    table_cols = ["순위", "모델명"]
    for col in metric_order:
        if col in table_display:
            table_display[col] = (table_display[col] * 100).round(1)  # 백분율로 표시
            table_cols.append(col)
    
    # 점수가 있는 행만 표시
    display_df = table_display[table_cols].dropna(subset=[c for c in table_cols if c not in ["순위", "모델명"]], how="all")
    
    st.dataframe(
        display_df, 
        width='stretch',
        height=400,
        hide_index=True
    )
    
    caption_parts = [
        f"💡 점수는 정확도(%)로 표시됩니다. 총 {len(display_df)}개 모델이 평가되었습니다. "
    ]
    if "개정세법 종합" in metric_table.columns:
        caption_parts.append(
            "전체 모델 성능 종합 = CPA(1.0) + 개정세법 주관식(1.0) + 개정세법 객관식(0.6) 가중 평균. "
        )
    else:
        caption_parts.append(
            "전체 모델 성능 종합 = CPA(1.0) + 개정세법 객관식(0.6) 가중 평균. "
        )
    caption_parts.append("(객관식은 GPT 기반 데이터로 0.6배 적용)")
    st.caption("".join(caption_parts))
    
    # 상세 설정
    with st.expander("⚙️ 차트 설정"):
        st.session_state["lb_chart_height"] = st.slider(
            "Top10 차트 높이",
            min_value=220,
            max_value=400,
            value=st.session_state.get("lb_chart_height", 300),
        )


def render_error_section(df: pd.DataFrame, pass_k: int, dataset: str):
    """오답 분석 페이지"""
    pass_col = f"pass_at_{pass_k}"
    if df.empty or pass_col not in df.columns:
        st.info("오답 분석에 필요한 데이터가 없습니다.")
        return

    model_list = sorted(df["model"].dropna().unique().tolist())
    if not model_list:
        st.info("선택 가능한 모델이 없습니다.")
        return

    st.markdown("### 🔍 오답 분석")
    
    # 모델 선택
    model_choice = st.selectbox(
        "분석할 모델 선택", 
        model_list,
        format_func=lambda x: x.split('/')[-1] if '/' in x else x
    )
    df_model = df[df["model"] == model_choice].copy()
    if df_model.empty:
        st.info("선택한 모델 데이터가 없습니다.")
        return

    incorrect = df_model[~df_model[pass_col]]
    correct = df_model[df_model[pass_col]]
    accuracy = (len(correct) / len(df_model) * 100) if len(df_model) > 0 else 0

    # 통계 카드
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("정답률", f"{accuracy:.1f}%")
    col2.metric("정답 수", f"{len(correct):,}개")
    col3.metric("오답 수", f"{len(incorrect):,}개")
    col4.metric("전체 문항", f"{len(df_model):,}개")

    if incorrect.empty:
        st.success("🎉 완벽한 정답률입니다!")
        return

    # 개정세법 객관식: 난이도별 정답/오답 (오답 분석 상단)
    if "difficulty" in df_model.columns and df_model["difficulty"].notna().any():
        st.markdown("#### 📊 난이도별 정답·오답")
        diff_stats = []
        for diff in sorted(df_model["difficulty"].dropna().unique()):
            sub = df_model[df_model["difficulty"] == diff]
            n = len(sub)
            c = sub[pass_col].sum()
            diff_stats.append({
                "난이도": diff,
                "문항 수": int(n),
                "정답 수": int(c),
                "오답 수": int(n - c),
                "정답률(%)": round(c / n * 100, 1) if n else 0,
            })
        diff_df = pd.DataFrame(diff_stats)
        col_diff1, col_diff2 = st.columns(2)
        with col_diff1:
            st.dataframe(
                diff_df.set_index("난이도"),
                use_container_width=True,
                hide_index=True,
            )
        with col_diff2:
            err_by_diff = incorrect["difficulty"].value_counts().reset_index()
            err_by_diff.columns = ["난이도", "오답수"]
            if not err_by_diff.empty:
                diff_chart = (
                    alt.Chart(err_by_diff)
                    .mark_bar(color="#764ba2", cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("난이도:N", title="난이도"),
                        y=alt.Y("오답수:Q", title="오답 수"),
                        tooltip=["난이도", "오답수"],
                    )
                    .properties(height=220)
                )
                st.altair_chart(diff_chart, use_container_width=True)
        st.markdown("---")

    # 2단 레이아웃
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 📊 오답 키워드 분석")
        keyword_df = top_keywords(incorrect["instruction"].tolist(), top_n=10)
        if keyword_df.empty:
            st.caption("분석 가능한 키워드가 없습니다.")
        else:
            keyword_chart = (
                alt.Chart(keyword_df)
                .mark_bar(color="#f97316", cornerRadiusEnd=4)
                .encode(
                    x=alt.X("count:Q", title="빈도"),
                    y=alt.Y("keyword:N", sort="-x", title="키워드"),
                    tooltip=[
                        alt.Tooltip("keyword:N", title="키워드"),
                        alt.Tooltip("count:Q", title="빈도")
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(keyword_chart, width='stretch')
    
    with col_right:
        st.markdown("#### 📈 오답 분포")
        if dataset == "cpa" and "subject" in incorrect.columns:
            subject_counts = incorrect["subject"].value_counts().reset_index()
            subject_counts.columns = ["과목", "오답수"]
            subject_chart = (
                alt.Chart(subject_counts)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("오답수:Q"),
                    color=alt.Color("과목:N", scale=alt.Scale(scheme="category10")),
                    tooltip=["과목", "오답수"]
                )
                .properties(height=300)
            )
            st.altair_chart(subject_chart, width='stretch')
        elif "target_year" in incorrect.columns:
            year_counts = incorrect["target_year"].value_counts().reset_index()
            year_counts.columns = ["연도", "오답수"]
            year_chart = (
                alt.Chart(year_counts)
                .mark_bar(color="#764ba2", cornerRadiusEnd=4)
                .encode(
                    x=alt.X("연도:O", title="연도"),
                    y=alt.Y("오답수:Q", title="오답 수"),
                    tooltip=["연도", "오답수"]
                )
                .properties(height=300)
            )
            st.altair_chart(year_chart, width='stretch')

    # 오답 목록
    st.markdown("#### 📝 오답 목록")
    
    id_col = "unique_id" if "unique_id" in incorrect.columns else "question_id"
    preview_cols = [id_col, "instruction", "ground_truth", "prediction", "prediction_1", "extracted_1", "Judge_Reason"]
    preview_cols = [c for c in preview_cols if c in incorrect.columns]
    preview_df = incorrect[preview_cols].copy()
    
    preview_df["정답"] = preview_df["ground_truth"].apply(extract_final_answer)
    if "prediction" in preview_df.columns:
        preview_df["모델답변"] = preview_df["prediction"].astype(str).str.strip()
    elif "prediction_1" in preview_df.columns:
        preview_df["모델답변"] = preview_df["prediction_1"].apply(extract_final_answer)
    elif "extracted_1" in preview_df.columns:
        preview_df["모델답변"] = preview_df["extracted_1"].astype(str).str.strip()
    else:
        preview_df["모델답변"] = ""
    if "Judge_Reason" in incorrect.columns:
        preview_df["판별 이유"] = incorrect["Judge_Reason"].astype(str).str.strip()
    if "correct_answer" in incorrect.columns:
        preview_df["정답번호"] = incorrect["correct_answer"]
    
    # 텍스트 축약
    preview_df["문제"] = preview_df["instruction"].apply(lambda x: shorten(x, 100))
    
    # 표시 컬럼: 문항, 문제, 정답, 모델답변, 판별 이유(주관식)
    display_cols = [id_col, "문제", "정답", "모델답변"]
    if "정답번호" in preview_df.columns:
        display_cols.insert(2, "정답번호")
    if "판별 이유" in preview_df.columns:
        display_cols.append("판별 이유")
    
    st.dataframe(
        preview_df[display_cols], 
        width='stretch', 
        height=350,
        hide_index=True
    )

    col_dl, col_space = st.columns([1, 3])
    with col_dl:
        st.download_button(
            "📥 오답 CSV 다운로드",
            incorrect.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_choice.replace('/', '_')}_errors_pass{pass_k}.csv",
            mime="text/csv",
        )

    # 오답 상세 보기
    with st.expander("🔎 오답 상세 보기"):
        selection_labels = [
            f"{row[id_col]} | {shorten(row['instruction'], 70)}"
            for _, row in incorrect.iterrows()
        ]
        selected_label = st.selectbox("문항 선택", selection_labels, key="detail_select")
        selected_idx = selection_labels.index(selected_label)
        row = incorrect.iloc[selected_idx]

        st.markdown("**📋 문제**")
        st.info(row.get("instruction", ""))
        
        st.markdown("**✅ 정답**")
        st.success(row.get("ground_truth", ""))

        extra_fields = []
        if dataset == "cpa":
            for field in ["year", "subject", "question_number", "correct_answer"]:
                if field in row and pd.notna(row[field]):
                    extra_fields.append(f"**{field}**: {row[field]}")
        if "target_year" in row and pd.notna(row.get("target_year")):
            extra_fields.append(f"**연도**: {row['target_year']}")
        if extra_fields:
            st.markdown(" | ".join(extra_fields))

        # 주관식: 모델 답변(prediction) + Judge 판별 이유
        if "prediction" in row.index and pd.notna(row.get("prediction")) and str(row.get("prediction", "")).strip():
            st.markdown("**🤖 모델 답변**")
            st.write(str(row["prediction"]).strip())
        if "Judge_Reason" in row.index and pd.notna(row.get("Judge_Reason")) and str(row.get("Judge_Reason", "")).strip():
            st.markdown("**⚖️ Judge 판별 이유**")
            st.info(str(row["Judge_Reason"]).strip())
        if "Judge_Score" in row.index and pd.notna(row.get("Judge_Score")):
            st.caption(f"Judge_Score: {row['Judge_Score']}")

        st.markdown("**🤖 모델 응답** (샘플별)")
        for i in range(1, 6):
            col = f"prediction_{i}"
            ext_col = f"extracted_{i}"
            if col in row and pd.notna(row[col]):
                final_ans = extract_final_answer(row[col])
                with st.expander(f"샘플 {i}: {final_ans}"):
                    st.write(row[col])
            elif ext_col in row and pd.notna(row[ext_col]):
                with st.expander(f"샘플 {i}: {row[ext_col]}"):
                    st.write(f"추출 답: {row[ext_col]}")

        matches = matching_samples(row, pass_k)
        if matches:
            st.success(f"✓ Pass@{pass_k} 매칭 샘플: {matches}")


def render_analysis_section(df: pd.DataFrame, pass_k: int, bench_main: str):
    """상세 분석 페이지"""
    if df.empty:
        st.info("분석에 필요한 데이터가 없습니다.")
        return
    pass_col = f"pass_at_{pass_k}"
    if pass_col not in df.columns:
        st.info("Pass@k 데이터가 없습니다.")
        return

    st.markdown("### 📈 성능 분석")

    # 모델별 선택: 선택 시 해당 모델만 상세 분석란에 반영
    model_list = sorted(df["model"].dropna().unique().tolist())
    options = ["전체 (모델 비교)"] + model_list
    model_choice = st.selectbox(
        "분석할 모델",
        options,
        index=0,
        format_func=lambda x: x if x == "전체 (모델 비교)" else (x.split("/")[-1] if "/" in str(x) else x),
        key="analysis_model_select",
    )
    if model_choice == "전체 (모델 비교)":
        df_analysis = df
    else:
        df_analysis = df[df["model"] == model_choice].copy()
        if df_analysis.empty:
            st.warning("선택한 모델 데이터가 없습니다.")
            return
        st.caption(f"선택 모델: **{model_choice}** ({len(df_analysis):,}문항)")

    # 2단 레이아웃
    col_left, col_right = st.columns(2)
    
    with col_left:
        if bench_main == "CPA" and "year" in df_analysis.columns:
            st.markdown("#### 연도별 평균 정확도")
            year_trend = (
                df_analysis.groupby("year", dropna=False)[pass_col]
                .mean()
                .reset_index()
                .assign(accuracy=lambda d: d[pass_col] * 100)
            )
            year_chart = (
                alt.Chart(year_trend)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#667eea")
                .encode(
                    x=alt.X("year:O", title="연도"),
                    y=alt.Y("accuracy:Q", title="정확도 (%)", scale=alt.Scale(domain=[0, 100])),
                    tooltip=[
                        alt.Tooltip("year:O", title="연도"),
                        alt.Tooltip("accuracy:Q", format=".1f", title="정확도(%)")
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(year_chart, width='stretch')
        elif bench_main == "개정세법" and "target_year" in df_analysis.columns:
            st.markdown("#### 연도별 평균 정확도")
            year_trend = (
                df_analysis.groupby("target_year", dropna=False)[pass_col]
                .mean()
                .reset_index()
                .assign(accuracy=lambda d: d[pass_col] * 100)
            )
            year_chart = (
                alt.Chart(year_trend)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#764ba2")
                .encode(
                    x=alt.X("target_year:O", title="연도"),
                    y=alt.Y("accuracy:Q", title="정확도 (%)", scale=alt.Scale(domain=[0, 100])),
                    tooltip=[
                        alt.Tooltip("target_year:O", title="연도"),
                        alt.Tooltip("accuracy:Q", format=".1f", title="정확도(%)")
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(year_chart, width='stretch')
    
    with col_right:
        st.markdown("#### 파라미터 수 vs 정확도")
        leaderboard = build_leaderboard(df, pass_k)
        scatter_df = leaderboard.dropna(subset=["params_b"])
        if scatter_df.empty:
            st.caption("파라미터 정보가 부족합니다.")
        else:
            scatter = (
                alt.Chart(scatter_df)
                .mark_circle(size=120, opacity=0.7)
                .encode(
                    x=alt.X("params_b:Q", title="파라미터 수 (B)", scale=alt.Scale(type="log")),
                    y=alt.Y("pass_rate:Q", title="정확도 (%)"),
                    color=alt.Color("organization:N", legend=alt.Legend(title="플랫폼")),
                    tooltip=[
                        alt.Tooltip("display_model:N", title="모델"),
                        alt.Tooltip("params_b:Q", format=".1f", title="파라미터(B)"),
                        alt.Tooltip("pass_rate:Q", format=".2f", title="정확도(%)"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(scatter, width='stretch')

    # 모델별 상세 추이 (개정세법의 경우) — 선택 모델이 있으면 해당 모델만 표시
    if bench_main == "개정세법" and "target_year" in df_analysis.columns:
        st.markdown("#### 연도별 추이")
        trend = (
            df_analysis.groupby(["target_year", "model"], dropna=False)[pass_col]
            .mean()
            .reset_index()
            .assign(accuracy=lambda d: d[pass_col] * 100)
        )
        trend["model_short"] = trend["model"].apply(lambda x: x.split('/')[-1][:30] if isinstance(x, str) else x)
        
        line = (
            alt.Chart(trend)
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("target_year:O", title="연도"),
                y=alt.Y("accuracy:Q", title="정확도 (%)"),
                color=alt.Color("model_short:N", legend=alt.Legend(title="모델", labelLimit=200)),
                tooltip=[
                    alt.Tooltip("model_short:N", title="모델"),
                    alt.Tooltip("target_year:O", title="연도"),
                    alt.Tooltip("accuracy:Q", format=".1f", title="정확도(%)")
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(line, width='stretch')
    
    # CPA 과목별 분석 — 세로 막대(가독성)
    elif bench_main == "CPA" and "subject" in df_analysis.columns:
        st.markdown("#### 과목별 성능 비교")
        subject_perf = (
            df_analysis.groupby("subject", dropna=False)[pass_col]
            .mean()
            .reset_index()
            .assign(accuracy=lambda d: d[pass_col] * 100)
            .sort_values("accuracy", ascending=False)
        )
        # 도표 위에 과목별 점수 바로 표시
        n_subj = len(subject_perf)
        score_cols = st.columns(max(1, n_subj))
        for i, (_, row) in enumerate(subject_perf.iterrows()):
            with score_cols[i]:
                st.metric(label=row["subject"], value=f"{row['accuracy']:.1f}%")
        subject_chart = (
            alt.Chart(subject_perf)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X(
                    "subject:N",
                    sort="-y",
                    title="과목",
                    axis=alt.Axis(labelAngle=-25, labelLimit=120),
                ),
                y=alt.Y(
                    "accuracy:Q",
                    title="정확도 (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                color=alt.Color("subject:N", legend=None, scale=alt.Scale(scheme="category10")),
                tooltip=[
                    alt.Tooltip("subject:N", title="과목"),
                    alt.Tooltip("accuracy:Q", format=".1f", title="정확도(%)"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(subject_chart, width='stretch')


def render_model_cards(metadata: dict):
    models = metadata.get("models", [])
    if not models:
        st.info("모델 메타데이터가 없습니다.")
        return
    meta_df = pd.DataFrame(models)
    st.dataframe(meta_df, width='stretch')


def render_data_and_eval_page():
    """데이터 구성·예시·판별 방식을 한 페이지로 안내"""
    st.markdown("### 📄 데이터 및 평가 방식 안내")

    st.markdown("---")
    st.markdown("#### 1. 데이터 구성 (어떤 식으로 만들었는지)")

    st.markdown("""
| 벤치마크 | 데이터 출처 | 구성 방식 |
|----------|-------------|-----------|
| **CPA 종합** | 공인회계사 시험 기출 (2016~2025) | 문항별 user/assistant 대화형 QA. 과목(세법, 회계학, 경영학 등)·연도·문항번호 메타데이터 포함. |
| **개정세법 객관식** | 연도별 개정세법 해설 기반 | 2023·2024·2025년 개정세법 기준 4지선다(500문항/연도). instruction + choices(A/B/C/D) + 정답. GPT로 문항 생성·정답 라벨 부여. |
| **개정세법 주관식** (옵션) | 연도별 세법 QA | 2021·2022·2023·2025 연도별 주관식 QA. 추론 후 **Judge API**로 Pass/Fail 채점된 결과만 리더보드에 반영. |
""")

    st.markdown("---")
    st.markdown("#### 2. 데이터 예시")

    st.markdown("**CPA (대화형)**")
    st.code("""{
  "conversation": [
    {"role": "user", "content": "다음 문제를 풀어주세요: ..."},
    {"role": "assistant", "content": "정답: ③"}
  ],
  "metadata": {"year": 2023, "subject": "세법", "question_number": 1}
}""", language="json")

    st.markdown("**개정세법 객관식 (4지선다)**")
    st.code("""{
  "instruction": "2023년 개정세법 기준: 다음 중 ... 옳은 것은?",
  "context": "국세기본법 (차례 기반)",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "C",
  "difficulty": "easy"
}""", language="json")

    st.markdown("---")
    st.markdown("#### 3. 판별 방식")

    st.markdown("""
- **Pass@k**: 문항당 k개 답변 생성 후, **하나라도 정답이면** 해당 문항 정답 처리. (k=1, 3, 5 중 선택)
- **CPA**: 모델 출력에서 `최종정답:` / `정답:` 뒤 문자·번호(①②③ 또는 1~5) 추출 → 정답 라벨과 **문자/번호 정규화 후 비교**.
- **개정세법 객관식**: 출력에서 A/B/C/D 추출(정답: A, (B), 문장 내 첫 A~D 등) → **정답 문자와 대소문자 무시 비교**.
- **개정세법 주관식**: **Judge API**로 (instruction, ground_truth, prediction) 전달 후 Pass/Fail 판정. 리더보드는 evaluated CSV의 Judge_Score만 사용(문자열 직접 비교 없음).
- **전체 모델 성능 종합**: CPA 1.0배 + 개정세법 객관식 0.6배(고정). 옵션으로 주관식 포함 시 주관식 1.0배 추가. 없으면 0점으로 포함해 가중 평균.
""")

    st.markdown("---")
    st.caption("리더보드 데이터·평가 방식 요약. 상세 스크립트는 docs/ 및 results_* 폴더 참고.")


def main():
    st.set_page_config(
        page_title=APP_TITLE, 
        layout="wide",
        page_icon="🏛️",
        initial_sidebar_state="collapsed"
    )
    render_header()

    data_root = find_data_root()
    metadata = load_metadata(str(data_root))

    # 상단 필터
    render_aihub_shell()

    col1, col2 = st.columns([1, 1])
    with col1:
        pass_k = st.selectbox("📈 평가지표", PASS_K_OPTIONS, index=2, format_func=pass_label)
    with col2:
        korean_only = st.checkbox("한국어 사전학습", value=False)

    # 데이터 로드
    yearly_all = load_yearly(str(data_root))
    yearly_all = attach_metadata(yearly_all, metadata)
    if "target_year" not in yearly_all.columns and "year_from_file" in yearly_all.columns:
        yearly_all["target_year"] = yearly_all["year_from_file"]

    cpa_all = load_cpa(str(data_root))
    cpa_all = attach_metadata(cpa_all, metadata)

    tax500_all = load_tax500(str(data_root))
    tax500_all = attach_metadata(tax500_all, metadata)

    # 고급 필터
    with st.expander("🔧 고급 필터"):
        filter_cols = st.columns(3)

        with filter_cols[0]:
            combined_meta = pd.concat(
                [yearly_all, cpa_all, tax500_all], ignore_index=True
            )
            orgs = sorted(combined_meta["organization"].dropna().unique().tolist())
            org_filter = st.multiselect("제출 플랫폼", orgs)
            include_yearly_tax = st.checkbox(
                "리더보드에 개정세법(주관식·연도별) 포함",
                value=False,
                key="include_yearly_tax",
                help="체크 시 전체 모델 성능 종합·테이블·차트에 개정세법 연도별 QA가 포함됩니다.",
            )

        with filter_cols[1]:
            years_yearly = sorted(yearly_all["target_year"].dropna().unique().tolist())
            years_tax500 = sorted(
                tax500_all["target_year"].dropna().unique().tolist()
            )
            years_cpa = sorted(cpa_all["year"].dropna().unique().tolist())
            subjects_cpa = sorted(cpa_all["subject"].dropna().unique().tolist())
            yearly_filter = st.multiselect("개정세법 연도", years_yearly, default=years_yearly)
            tax500_year_filter = st.multiselect(
                "개정세법 객관식 연도", years_tax500, default=years_tax500
            )
            cpa_year_filter = st.multiselect("CPA 연도", years_cpa, default=years_cpa)
            cpa_subject_filter = st.multiselect("CPA 과목", subjects_cpa, default=subjects_cpa)

        with filter_cols[2]:
            model_list = sorted(combined_meta["model"].dropna().unique().tolist())
            model_filter = st.multiselect(
                "모델 선택",
                model_list,
                format_func=lambda x: x.split('/')[-1] if '/' in x else x
            )

    def apply_meta_filters(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        if korean_only:
            frame = frame[frame["korean_pretrained"] == True]
        if org_filter:
            frame = frame[frame["organization"].isin(org_filter)]
        if model_filter:
            frame = frame[frame["model"].isin(model_filter)]
        return frame

    yearly_filtered = yearly_all.copy()
    cpa_filtered = cpa_all.copy()
    tax500_filtered = tax500_all.copy()

    if yearly_filter:
        yearly_filtered = yearly_filtered[
            yearly_filtered["target_year"].isin(yearly_filter)
        ]
    if tax500_year_filter:
        tax500_filtered = tax500_filtered[
            tax500_filtered["target_year"].isin(tax500_year_filter)
        ]
    if cpa_year_filter:
        cpa_filtered = cpa_filtered[cpa_filtered["year"].isin(cpa_year_filter)]
    if cpa_subject_filter:
        cpa_filtered = cpa_filtered[cpa_filtered["subject"].isin(cpa_subject_filter)]

    yearly_filtered = apply_meta_filters(yearly_filtered)
    cpa_filtered = apply_meta_filters(cpa_filtered)
    tax500_filtered = apply_meta_filters(tax500_filtered)

    if (
        yearly_filtered.empty
        and cpa_filtered.empty
        and tax500_filtered.empty
    ):
        st.warning("⚠️ 필터 조건에 맞는 데이터가 없습니다. 필터를 조정해주세요.")
        return

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 리더보드", "🔍 오답 분석", "📈 상세 분석", "📄 데이터·평가 안내"]
    )

    with tab1:
        render_leaderboard_section(
            yearly_filtered,
            cpa_filtered,
            tax500_filtered,
            pass_k,
            metadata,
            include_yearly_tax=include_yearly_tax,
        )

    with tab2:
        err_ds = st.selectbox(
            "오답 분석 데이터",
            ["CPA", "개정세법(연도별)", "개정세법 객관식"],
            key="err_ds",
        )
        if err_ds == "CPA":
            df_err = cpa_filtered
            render_error_section(df_err, pass_k, "cpa")
        elif err_ds == "개정세법 객관식":
            df_err = tax500_filtered
            render_error_section(df_err, pass_k, "yearly")
        else:
            df_err = yearly_filtered
            render_error_section(df_err, pass_k, "yearly")

    with tab3:
        ana_ds = st.selectbox(
            "분석 데이터",
            ["CPA", "개정세법(연도별)", "개정세법 객관식"],
            key="ana_ds",
        )
        if ana_ds == "CPA":
            df_ana = cpa_filtered
            render_analysis_section(df_ana, pass_k, "CPA")
        elif ana_ds == "개정세법 객관식":
            df_ana = tax500_filtered
            render_analysis_section(df_ana, pass_k, "개정세법")
        else:
            df_ana = yearly_filtered
            render_analysis_section(df_ana, pass_k, "개정세법")

    with tab4:
        render_data_and_eval_page()


if __name__ == "__main__":
    main()
