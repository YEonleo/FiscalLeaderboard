# Fiscal Leaderboard

한국어 LLM의 **세법·회계** 전문 지식을 평가하는 벤치마크 리더보드입니다.  
개정세법(연도별 QA, 객관식 500문항) 및 CPA 시험 데이터를 활용해 모델 성능을 비교합니다.

---

## 폴더 구조

```
Fiscal_leaderboard/
├── README.md
├── requirements.txt          # (선택) 로컬 실행 시 pip install -r hf_space/requirements.txt 권장
├── model_metadata.json       # 모델 메타정보 (조직, 파라미터 등)
├── data/                     # 벤치마크 원본 데이터
│   ├── cpa_2016_2025_combined.jsonl   # CPA 문항
│   ├── tax_law_qa_2021.jsonl         # 개정세법 연도별 QA
│   ├── tax_law_qa_2022.jsonl
│   ├── tax_law_qa_2023.jsonl
│   ├── tax_law_qa_2025.jsonl
│   └── ...
├── hf_space/                 # Streamlit 앱 (리더보드 UI)
│   ├── app.py
│   ├── requirements.txt
│   └── .streamlit/
├── results_cpa/              # CPA 평가 결과 (summary/raw)
│   ├── summary/              # *_cpa_summary.csv, *_metadata.json
│   └── raw/                  # *_cpa_raw.csv
├── results_tax500/           # 개정세법 객관식 500문항 결과
│   └── summary/              # *_tax500_*_summary.csv
└── results_yearly/           # 개정세법 연도별(주관식) 결과
    └── raw/                  # *_raw_evaluated.csv (Judge_Score 포함)
```

- **앱**은 `hf_space/app.py` 한 곳에서 실행됩니다.
- **데이터 루트**는 `results_cpa`, `results_tax500`, `results_yearly` 중 하나가 있는 디렉터리로 자동 탐색됩니다 (앱 기준 상위 폴더 포함).

---

## Git 제외 (대용량 결과물)

용량이 큰 결과물 폴더는 Git에 올리지 않도록 `.gitignore`에 추가해 두었습니다.

- 제외 대상: `results_cpa/raw/`, `results_cpa/summary/`, `results_tax500/`

필요 시 추가하는 방법:

1. `.gitignore`에서 위 라인을 삭제(또는 주석 처리)한 뒤 `git add` 합니다.
2. 이미 ignore된 상태에서 강제로 올리려면 `git add -f results_cpa/raw results_cpa/summary results_tax500` 를 사용합니다.

---

## 실행 방법

### 로컬

```bash
cd Fiscal_leaderboard/hf_space
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

### Hugging Face Space

- Space 생성 후 `hf_space/` 내용을 루트로 두거나, 앱이 `results_*` 폴더를 찾을 수 있도록 데이터를 업로드합니다.
- 앱 진입점: `app.py` (Streamlit).

---

## 데이터 요약

| 구분 | 설명 |
|------|------|
| **CPA** | 공인회계사 시험 기출 (2016~2025), 과목·연도·문항별. 객관식 ①②③④⑤ 정답 비교. |
| **개정세법 연도별** | 2021·2022·2023·2025 연도별 주관식 QA. Judge API로 Pass/Fail 채점된 결과 사용. |
| **개정세법 객관식 500** | tax500 연도별 객관식. `*_tax500_*_summary.csv` 기준. |

---

## 기술 스택

- **Streamlit** – 리더보드 UI
- **Pandas** – 테이블·집계
- **Altair** – 차트

---

## 라이선스

저장소 소유자의 라이선스 정책을 따릅니다.
