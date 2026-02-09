# Fiscal Leaderboard (앱)

Streamlit 기반 리더보드 UI. 프로젝트 루트의 `README.md`와 폴더 구조를 참고하세요.

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

- 앱은 상위 폴더에서 `results_cpa/`, `results_tax500/`, `results_yearly/` 를 찾습니다.  
- 루트에서 실행할 경우: `streamlit run hf_space/app.py` (작업 디렉터리는 루트로 두면 됨).

## Hugging Face Space

- 이 디렉터리(`hf_space/`)를 Space 루트로 두고 `app.py`를 진입점으로 설정하면 됩니다.
- 결과 데이터가 필요하면 `results_*` 폴더를 Space에 업로드하거나, 루트 README의 데이터 구조를 참고해 배치하세요.

## 의존성

- streamlit
- pandas
- numpy
- altair
