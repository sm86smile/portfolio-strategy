import math
import os
from datetime import date

import pandas as pd
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta


# ============================================================
# ETF 자산배분 앱
# - VAA 공격형 / LAA / 오리지널 듀얼 모멘텀
# - 국내상장 ETF 대체 매핑
# - 화면에서 KRX 키를 입력받지 않음
# - 전략별 최근/다음 리밸런싱 일정 표시
# - Appendix: 매수 전략 및 리밸런싱 계산 방식 표 제공
# ============================================================

st.set_page_config(page_title="ETF 자산배분", layout="wide")

KRX_URL = "http://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"

ETF_MAP = [
    {"original": "SPY", "asset": "미국 S&P500", "code": "360750", "name": "TIGER 미국S&P500", "strategy": "VAA, ODM"},
    {"original": "QQQ", "asset": "미국 나스닥100", "code": "133690", "name": "TIGER 미국나스닥100", "strategy": "LAA"},
    {"original": "IWD", "asset": "미국 대형 가치주", "code": "0127M0", "name": "ACE 미국대형가치주액티브", "strategy": "LAA"},
    {"original": "GLD", "asset": "금", "code": "411060", "name": "ACE KRX금현물", "strategy": "LAA"},
    {"original": "IEF", "asset": "미국 중기국채", "code": "305080", "name": "TIGER 미국채10년선물", "strategy": "VAA, LAA"},
    {"original": "SHY", "asset": "미국 단기채", "code": "329750", "name": "TIGER 미국달러단기채권액티브", "strategy": "VAA, LAA"},
    {"original": "BIL", "asset": "미국 초단기채/현금성 비교 기준", "code": "329750", "name": "TIGER 미국달러단기채권액티브", "strategy": "ODM"},
    {"original": "EFA", "asset": "미국 제외 선진국", "code": "195970", "name": "PLUS 선진국MSCI(합성 H)", "strategy": "VAA, ODM"},
    {"original": "EEM", "asset": "신흥국", "code": "195980", "name": "PLUS 신흥국MSCI(합성 H)", "strategy": "VAA"},
    {"original": "AGG", "asset": "미국 종합채권", "code": "437080", "name": "KODEX 미국종합채권ESG액티브(H)", "strategy": "VAA, ODM"},
    {"original": "LQD", "asset": "미국 투자등급 회사채", "code": "458260", "name": "TIGER 미국투자등급회사채액티브(H)", "strategy": "VAA"},
]

VAA_ATTACK = ["SPY", "EFA", "EEM", "AGG"]
VAA_SAFE = ["LQD", "IEF", "SHY"]
LAA_FIXED = ["IWD", "GLD", "IEF"]
ODM_ASSETS = ["SPY", "EFA", "BIL", "AGG"]


def get_data_key() -> str:
    """화면 입력 없이 Streamlit Secrets 또는 환경변수에서 연결값을 읽는다."""
    value = ""

    try:
        value = st.secrets.get("KRX_AUTH_KEY", "")
    except Exception:
        value = ""

    if not value:
        value = os.getenv("KRX_AUTH_KEY", "")

    return str(value).strip()


def to_number(x):
    if x is None:
        return math.nan

    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).replace(",", "").replace("%", "").strip()

    if s in ("", "-", "N/A", "nan", "None"):
        return math.nan

    try:
        return float(s)
    except ValueError:
        return math.nan


def find_rows(payload):
    if not isinstance(payload, dict):
        return []

    for key in ["OutBlock_1", "OutBlock1", "OutBlock", "output", "result", "data"]:
        if isinstance(payload.get(key), list):
            return payload[key]

    for value in payload.values():
        if isinstance(value, list):
            return value

    return []


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_one_day(data_key: str, ymd: str) -> pd.DataFrame:
    headers = {"AUTH_KEY": data_key} if data_key else {}

    r = requests.get(
        KRX_URL,
        params={"basDd": ymd},
        headers=headers,
        timeout=20,
    )
    r.raise_for_status()

    df = pd.DataFrame(find_rows(r.json()))

    if df.empty:
        return df

    required = ["BAS_DD", "ISU_CD", "ISU_NM", "TDD_CLSPRC"]

    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    keep = [
        c
        for c in [
            "BAS_DD",
            "ISU_CD",
            "ISU_NM",
            "TDD_CLSPRC",
            "ACC_TRDVAL",
            "ACC_TRDVOL",
            "NAV",
            "FLUC_RT",
        ]
        if c in df.columns
    ]

    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["BAS_DD"].astype(str), format="%Y%m%d", errors="coerce")
    df["ISU_CD"] = df["ISU_CD"].astype(str).str.strip()
    df["ISU_NM"] = df["ISU_NM"].astype(str).str.strip()
    df["close"] = df["TDD_CLSPRC"].apply(to_number)

    if "ACC_TRDVAL" in df.columns:
        df["trading_value"] = df["ACC_TRDVAL"].apply(to_number)
    else:
        df["trading_value"] = math.nan

    return df.dropna(subset=["date", "close"])


def fetch_history(data_key: str, codes: list[str], start_dt, end_dt) -> pd.DataFrame:
    codes = set(str(c).strip() for c in codes if str(c).strip())
    days = pd.date_range(start=start_dt, end=end_dt, freq="B")
    rows = []

    if len(days) == 0:
        return pd.DataFrame()

    bar = st.progress(0, text="ETF 가격 데이터를 조회하는 중입니다.")

    for i, d in enumerate(days):
        ymd = d.strftime("%Y%m%d")

        try:
            day_df = fetch_one_day(data_key, ymd)

            if not day_df.empty:
                filtered = day_df[day_df["ISU_CD"].isin(codes)].copy()

                if not filtered.empty:
                    rows.append(filtered)
        except Exception:
            pass

        bar.progress(
            (i + 1) / len(days),
            text=f"ETF 가격 데이터 조회 중... {ymd}",
        )

    bar.empty()

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["ISU_CD", "date"])
    out = out.drop_duplicates(["ISU_CD", "date"], keep="last")

    return out


def make_map(map_df: pd.DataFrame) -> dict:
    result = {}

    for _, r in map_df.iterrows():
        original = str(r["original"]).strip().upper()
        code = str(r["code"]).strip()

        if original and code:
            result[original] = {
                "code": code,
                "name": str(r["name"]).strip(),
                "asset": str(r["asset"]).strip(),
            }

    return result


def price_on_or_before(hist: pd.DataFrame, code: str, target_dt):
    sub = hist[
        (hist["ISU_CD"] == code)
        & (hist["date"] <= pd.Timestamp(target_dt))
    ].sort_values("date")

    if sub.empty:
        return None

    row = sub.iloc[-1]

    return {
        "date": row["date"],
        "price": float(row["close"]),
        "name": row["ISU_NM"],
    }


def calc_return(hist: pd.DataFrame, code: str, eval_dt, months: int):
    now = price_on_or_before(hist, code, eval_dt)
    past = price_on_or_before(
        hist,
        code,
        pd.Timestamp(eval_dt) - relativedelta(months=months),
    )

    if now is None or past is None:
        return None

    if past["price"] == 0 or math.isnan(past["price"]):
        return None

    return {
        "eval_date": now["date"],
        "base_date": past["date"],
        "return": now["price"] / past["price"] - 1,
    }


def vaa_table(hist: pd.DataFrame, original_map: dict, eval_dt) -> pd.DataFrame:
    rows = []

    for original in VAA_ATTACK + VAA_SAFE:
        info = original_map[original]

        row = {
            "구분": "공격자산" if original in VAA_ATTACK else "안전자산",
            "원 ETF": original,
            "국내 ETF 코드": info["code"],
            "국내 ETF명": info["name"],
            "자산군": info["asset"],
        }

        ok = True

        for m in [1, 3, 6, 12]:
            r = calc_return(hist, info["code"], eval_dt, m)

            row[f"{m}개월 수익률"] = None if r is None else r["return"]
            row[f"{m}개월 기준일"] = None if r is None else r["base_date"].date()

            if r is None:
                ok = False

        if ok:
            row["모멘텀 스코어"] = (
                12 * row["1개월 수익률"]
                + 4 * row["3개월 수익률"]
                + 2 * row["6개월 수익률"]
                + row["12개월 수익률"]
            )
        else:
            row["모멘텀 스코어"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def odm_table(hist: pd.DataFrame, original_map: dict, eval_dt) -> pd.DataFrame:
    rows = []

    for original in ODM_ASSETS:
        info = original_map[original]
        r = calc_return(hist, info["code"], eval_dt, 12)

        rows.append(
            {
                "원 ETF": original,
                "국내 ETF 코드": info["code"],
                "국내 ETF명": info["name"],
                "자산군": info["asset"],
                "12개월 수익률": None if r is None else r["return"],
                "12개월 기준일": None if r is None else r["base_date"].date(),
                "평가 기준일": None if r is None else r["eval_date"].date(),
            }
        )

    return pd.DataFrame(rows)


def next_rebalance_date(last_date, cycle: str):
    """최근 리밸런싱일과 주기에 따라 다음 리밸런싱일 계산."""
    if cycle == "연 1회":
        return last_date + relativedelta(years=1)

    if cycle == "월 1회":
        return last_date + relativedelta(months=1)

    return None


def rebalance_status(next_date, base_date):
    if next_date is None:
        return "-"

    if pd.Timestamp(base_date).date() >= next_date:
        return "리밸런싱 필요"

    return "대기"


def alloc_rows(
    strategy: str,
    strategy_weight: float,
    inner_alloc: dict,
    original_map: dict,
    rebalance_info_by_original: dict,
    base_date,
):
    rows = []

    for original, inner_weight in inner_alloc.items():
        info = original_map[original]
        rb = rebalance_info_by_original.get(original, {})
        cycle = rb.get("cycle", "-")
        last_date = rb.get("last_date", None)
        next_date = next_rebalance_date(last_date, cycle) if last_date else None

        rows.append(
            {
                "하위전략": strategy,
                "원 ETF": original,
                "국내 ETF 코드": info["code"],
                "국내 ETF명": info["name"],
                "자산군": info["asset"],
                "하위전략 내 비중": inner_weight,
                "강환국 전략 전체 비중": strategy_weight * inner_weight,
                "리밸런싱 주기": cycle,
                "최근 리밸런싱일": last_date,
                "다음 리밸런싱일": next_date,
                "리밸런싱 상태": rebalance_status(next_date, base_date),
            }
        )

    return rows


def pct(x):
    if x is None or pd.isna(x):
        return "-"

    return f"{x * 100:.2f}%"


def pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(pct)

    return out


def date_cols_to_string(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: "-" if pd.isna(x) or x is None else str(x))

    return out


def appendix_buy_strategy_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "전략": "LAA",
                "구분": "고정자산",
                "대상 원 ETF": "IWD, GLD, IEF",
                "국내 대체 ETF": "ACE 미국대형가치주액티브, ACE KRX금현물, TIGER 미국채10년선물",
                "매수/보유 규칙": "각 25%씩 투자 및 보유",
                "판정 지표": "없음",
                "선택 결과": "항상 보유",
            },
            {
                "전략": "LAA",
                "구분": "변동자산",
                "대상 원 ETF": "QQQ 또는 SHY",
                "국내 대체 ETF": "TIGER 미국나스닥100 또는 TIGER 미국달러단기채권액티브",
                "매수/보유 규칙": "나머지 25%를 QQQ 또는 SHY에 투자",
                "판정 지표": "수동 O/X: S&P500 < 200일선 AND 미국 실업률 > 12개월 평균",
                "선택 결과": "O이면 SHY, X이면 QQQ",
            },
            {
                "전략": "VAA 공격형",
                "구분": "공격/안전자산 선택",
                "대상 원 ETF": "공격자산: SPY, EFA, EEM, AGG / 안전자산: LQD, IEF, SHY",
                "국내 대체 ETF": "앱의 ETF 매핑표 기준",
                "매수/보유 규칙": "공격자산 4개 모두 모멘텀 스코어가 양호하면 최고 공격자산에 100% 투자. 하나라도 방어 신호면 최고 안전자산에 100% 투자",
                "판정 지표": "(12×1개월 수익률)+(4×3개월 수익률)+(2×6개월 수익률)+(1×12개월 수익률)",
                "선택 결과": "선택 ETF 1개에 VAA 비중 전액 투입",
            },
            {
                "전략": "오리지널 듀얼 모멘텀",
                "구분": "주식/채권 선택",
                "대상 원 ETF": "SPY, EFA, BIL, AGG",
                "국내 대체 ETF": "TIGER 미국S&P500, PLUS 선진국MSCI(합성 H), TIGER 미국달러단기채권액티브, KODEX 미국종합채권ESG액티브(H)",
                "매수/보유 규칙": "SPY 12개월 수익률이 BIL보다 높으면 SPY/EFA 중 강한 자산 선택. SPY가 BIL보다 낮거나 같으면 AGG 선택",
                "판정 지표": "최근 12개월 수익률",
                "선택 결과": "선택 ETF 1개에 ODM 비중 전액 투입",
            },
        ]
    )


def appendix_rebalance_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "전략": "LAA",
                "대상": "IWD, GLD, IEF",
                "주기": "연 1회",
                "최근 리밸런싱일 입력": "LAA 고정자산 최근 리밸런싱일",
                "다음 리밸런싱일 계산": "최근 리밸런싱일 + 1년",
                "상태 계산": "평가 기준일 >= 다음 리밸런싱일이면 리밸런싱 필요, 아니면 대기",
            },
            {
                "전략": "LAA",
                "대상": "QQQ 또는 SHY",
                "주기": "월 1회",
                "최근 리밸런싱일 입력": "LAA 변동자산 최근 리밸런싱일",
                "다음 리밸런싱일 계산": "최근 리밸런싱일 + 1개월",
                "상태 계산": "평가 기준일 >= 다음 리밸런싱일이면 리밸런싱 필요, 아니면 대기",
            },
            {
                "전략": "VAA 공격형",
                "대상": "선택 ETF 1개",
                "주기": "월 1회",
                "최근 리밸런싱일 입력": "VAA 공격형 최근 리밸런싱일",
                "다음 리밸런싱일 계산": "최근 리밸런싱일 + 1개월",
                "상태 계산": "평가 기준일 >= 다음 리밸런싱일이면 리밸런싱 필요, 아니면 대기",
            },
            {
                "전략": "오리지널 듀얼 모멘텀",
                "대상": "선택 ETF 1개",
                "주기": "월 1회",
                "최근 리밸런싱일 입력": "오리지널 듀얼 모멘텀 최근 리밸런싱일",
                "다음 리밸런싱일 계산": "최근 리밸런싱일 + 1개월",
                "상태 계산": "평가 기준일 >= 다음 리밸런싱일이면 리밸런싱 필요, 아니면 대기",
            },
        ]
    )


# ============================================================
# 화면
# ============================================================

st.title("강환국 전략 ETF 자산배분")
st.caption("VAA 공격형 · LAA · 오리지널 듀얼 모멘텀을 국내상장 ETF로 대체해 최종 매수 비중과 리밸런싱 일정을 계산합니다.")

data_key = get_data_key()

today = date.today()
default_monthly_last = today - relativedelta(months=1)
default_annual_last = today - relativedelta(years=1)

with st.sidebar:
    st.header("1) 조회 기준")
    eval_date = st.date_input("평가 기준일", value=today)
    lookback_months = st.slider(
        "데이터 조회기간",
        min_value=13,
        max_value=36,
        value=15,
        help="12개월 수익률 계산을 위해 최소 13개월 이상 필요합니다.",
    )

    st.header("2) LAA 수동 조건")
    laa_defensive = st.radio(
        "S&P500 200일선 하회 + 미국 실업률 12개월 평균 상회 여부",
        options=[False, True],
        format_func=lambda x: "X: 조건 미충족 → QQQ 대체 ETF" if not x else "O: 조건 충족 → SHY 대체 ETF",
        index=0,
    )

    st.header("3) 최근 리밸런싱일")
    laa_annual_last = st.date_input(
        "LAA 고정자산 최근 리밸런싱일",
        value=default_annual_last,
        help="IWD, GLD, IEF 대체 ETF의 최근 연간 리밸런싱일",
    )
    laa_monthly_last = st.date_input(
        "LAA 변동자산 최근 리밸런싱일",
        value=default_monthly_last,
        help="QQQ 또는 SHY 대체 ETF의 최근 월간 리밸런싱일",
    )
    vaa_monthly_last = st.date_input(
        "VAA 공격형 최근 리밸런싱일",
        value=default_monthly_last,
        help="VAA 공격형 전략의 최근 월간 리밸런싱일",
    )
    odm_monthly_last = st.date_input(
        "오리지널 듀얼 모멘텀 최근 리밸런싱일",
        value=default_monthly_last,
        help="오리지널 듀얼 모멘텀 전략의 최근 월간 리밸런싱일",
    )

    st.header("4) 하위전략 비중")
    w_laa = st.number_input(
        "LAA 비중",
        min_value=0.0,
        max_value=1.0,
        value=1 / 3,
        step=0.01,
        format="%.4f",
    )
    w_vaa = st.number_input(
        "VAA 공격형 비중",
        min_value=0.0,
        max_value=1.0,
        value=1 / 3,
        step=0.01,
        format="%.4f",
    )
    w_odm = st.number_input(
        "오리지널 듀얼 모멘텀 비중",
        min_value=0.0,
        max_value=1.0,
        value=1 / 3,
        step=0.01,
        format="%.4f",
    )

    st.header("5) VAA 판정 기준")
    zero_is_defensive = st.checkbox(
        "모멘텀 스코어 0점은 방어 신호로 처리",
        value=True,
    )

weight_sum = w_laa + w_vaa + w_odm

if weight_sum <= 0:
    st.error("하위전략 비중 합계가 0입니다. 비중을 입력하세요.")
    st.stop()

if abs(weight_sum - 1.0) > 1e-8:
    st.warning(f"하위전략 비중 합계가 {weight_sum:.4f}입니다. 계산 시 100% 기준으로 자동 정규화합니다.")
    w_laa = w_laa / weight_sum
    w_vaa = w_vaa / weight_sum
    w_odm = w_odm / weight_sum


with st.expander("국내상장 ETF 대체 매핑 확인/수정", expanded=False):
    st.write("같은 자산군의 다른 ETF를 쓰고 싶으면 종목코드와 ETF명을 수정하세요.")
    map_df = st.data_editor(
        pd.DataFrame(ETF_MAP),
        num_rows="dynamic",
        use_container_width=True,
    )

original_map = make_map(map_df)

required_originals = sorted(
    set(
        VAA_ATTACK
        + VAA_SAFE
        + LAA_FIXED
        + ["QQQ", "SHY"]
        + ODM_ASSETS
    )
)

missing = [
    x
    for x in required_originals
    if x not in original_map or not original_map[x]["code"]
]

if missing:
    st.error(f"필수 ETF 매핑이 없습니다: {', '.join(missing)}")
    st.stop()

st.subheader("입력된 리밸런싱 일정")
schedule_preview = pd.DataFrame(
    [
        {
            "구분": "LAA 고정자산",
            "대상": "IWD, GLD, IEF",
            "주기": "연 1회",
            "최근 리밸런싱일": laa_annual_last,
            "다음 리밸런싱일": next_rebalance_date(laa_annual_last, "연 1회"),
            "상태": rebalance_status(next_rebalance_date(laa_annual_last, "연 1회"), eval_date),
        },
        {
            "구분": "LAA 변동자산",
            "대상": "QQQ 또는 SHY",
            "주기": "월 1회",
            "최근 리밸런싱일": laa_monthly_last,
            "다음 리밸런싱일": next_rebalance_date(laa_monthly_last, "월 1회"),
            "상태": rebalance_status(next_rebalance_date(laa_monthly_last, "월 1회"), eval_date),
        },
        {
            "구분": "VAA 공격형",
            "대상": "선택 ETF 1개",
            "주기": "월 1회",
            "최근 리밸런싱일": vaa_monthly_last,
            "다음 리밸런싱일": next_rebalance_date(vaa_monthly_last, "월 1회"),
            "상태": rebalance_status(next_rebalance_date(vaa_monthly_last, "월 1회"), eval_date),
        },
        {
            "구분": "오리지널 듀얼 모멘텀",
            "대상": "선택 ETF 1개",
            "주기": "월 1회",
            "최근 리밸런싱일": odm_monthly_last,
            "다음 리밸런싱일": next_rebalance_date(odm_monthly_last, "월 1회"),
            "상태": rebalance_status(next_rebalance_date(odm_monthly_last, "월 1회"), eval_date),
        },
    ]
)
st.dataframe(schedule_preview, use_container_width=True)

with st.expander("Appendix. 매수 전략 및 리밸런싱 일정 계산 방식", expanded=False):
    tab1, tab2 = st.tabs(["매수 전략 요약", "리밸런싱 일정 계산 방식"])

    with tab1:
        st.dataframe(appendix_buy_strategy_table(), use_container_width=True)

    with tab2:
        st.dataframe(appendix_rebalance_table(), use_container_width=True)

run = st.button("전략 비중 계산", type="primary")

if run:
    start_dt = pd.Timestamp(eval_date) - relativedelta(months=lookback_months)
    end_dt = pd.Timestamp(eval_date)

    needed_codes = sorted(set(original_map[x]["code"] for x in required_originals))

    hist = fetch_history(
        data_key,
        needed_codes,
        start_dt,
        end_dt,
    )

    if hist.empty:
        st.error("ETF 가격 데이터를 가져오지 못했습니다. 기준일, 종목코드, 데이터 연결 상태를 확인하세요.")
        st.stop()

    actual_eval_dt = hist["date"].max()

    st.success(f"계산 기준일: {actual_eval_dt.date()}")

    with st.expander("계산 기준일 ETF 가격 확인", expanded=False):
        latest = (
            hist[hist["date"] == actual_eval_dt]
            .sort_values("ISU_CD")[["ISU_CD", "ISU_NM", "close", "trading_value"]]
            .rename(
                columns={
                    "ISU_CD": "국내 ETF 코드",
                    "ISU_NM": "국내 ETF명",
                    "close": "종가",
                    "trading_value": "거래대금",
                }
            )
        )
        st.dataframe(latest, use_container_width=True)

    rows = []

    laa_variable = "SHY" if laa_defensive else "QQQ"

    laa_rebalance_info = {
        "IWD": {"cycle": "연 1회", "last_date": laa_annual_last},
        "GLD": {"cycle": "연 1회", "last_date": laa_annual_last},
        "IEF": {"cycle": "연 1회", "last_date": laa_annual_last},
        laa_variable: {"cycle": "월 1회", "last_date": laa_monthly_last},
    }

    rows += alloc_rows(
        "LAA",
        w_laa,
        {
            "IWD": 0.25,
            "GLD": 0.25,
            "IEF": 0.25,
            laa_variable: 0.25,
        },
        original_map,
        laa_rebalance_info,
        eval_date,
    )

    vt = vaa_table(hist, original_map, actual_eval_dt)

    attack = vt[vt["원 ETF"].isin(VAA_ATTACK)].copy()
    safe = vt[vt["원 ETF"].isin(VAA_SAFE)].copy()

    if zero_is_defensive:
        attack_ok = (attack["모멘텀 스코어"] > 0).all()
    else:
        attack_ok = (attack["모멘텀 스코어"] >= 0).all()

    if attack["모멘텀 스코어"].isna().any() or safe["모멘텀 스코어"].isna().any():
        st.warning("일부 ETF는 모멘텀 계산에 필요한 가격 데이터가 부족합니다.")

    if attack_ok:
        vaa_pool = attack
        vaa_reason = "공격자산 4개가 모두 양호하여 공격자산 중 최고 점수 ETF 선택"
    else:
        vaa_pool = safe
        vaa_reason = "공격자산 중 방어 신호가 있어 안전자산 중 최고 점수 ETF 선택"

    vaa_pool_valid = vaa_pool.dropna(subset=["모멘텀 스코어"])

    if vaa_pool_valid.empty:
        st.error("VAA 선택 자산을 계산할 수 없습니다. ETF 매핑 또는 조회기간을 확인하세요.")
        st.stop()

    vaa_selected = (
        vaa_pool_valid
        .sort_values("모멘텀 스코어", ascending=False)
        .iloc[0]["원 ETF"]
    )

    rows += alloc_rows(
        "VAA 공격형",
        w_vaa,
        {vaa_selected: 1.0},
        original_map,
        {vaa_selected: {"cycle": "월 1회", "last_date": vaa_monthly_last}},
        eval_date,
    )

    ot = odm_table(hist, original_map, actual_eval_dt)

    def get_r(original):
        s = ot[ot["원 ETF"] == original]["12개월 수익률"]

        if s.empty or pd.isna(s.iloc[0]):
            return None

        return float(s.iloc[0])

    spy_r = get_r("SPY")
    efa_r = get_r("EFA")
    bil_r = get_r("BIL")

    if spy_r is None or efa_r is None or bil_r is None:
        st.error("오리지널 듀얼 모멘텀 계산에 필요한 12개월 수익률 데이터가 부족합니다.")
        st.stop()

    if spy_r > bil_r:
        odm_selected = "SPY" if spy_r >= efa_r else "EFA"
        odm_reason = "SPY 12개월 수익률이 BIL 대체 ETF보다 높아 SPY/EFA 중 강한 ETF 선택"
    else:
        odm_selected = "AGG"
        odm_reason = "SPY 12개월 수익률이 BIL 대체 ETF보다 낮거나 같아 AGG 선택"

    rows += alloc_rows(
        "오리지널 듀얼 모멘텀",
        w_odm,
        {odm_selected: 1.0},
        original_map,
        {odm_selected: {"cycle": "월 1회", "last_date": odm_monthly_last}},
        eval_date,
    )

    st.subheader("하위전략별 선택 결과")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "LAA 변동 25%",
        f"{laa_variable} → {original_map[laa_variable]['name']}",
    )
    c1.caption(f"다음 월간 리밸런싱일: {next_rebalance_date(laa_monthly_last, '월 1회')}")

    c2.metric(
        "VAA 공격형",
        f"{vaa_selected} → {original_map[vaa_selected]['name']}",
    )
    c2.caption(f"{vaa_reason} / 다음 리밸런싱일: {next_rebalance_date(vaa_monthly_last, '월 1회')}")

    c3.metric(
        "오리지널 듀얼 모멘텀",
        f"{odm_selected} → {original_map[odm_selected]['name']}",
    )
    c3.caption(f"{odm_reason} / 다음 리밸런싱일: {next_rebalance_date(odm_monthly_last, '월 1회')}")

    st.subheader("VAA 모멘텀 스코어")
    st.dataframe(
        pct_cols(
            vt,
            [
                "1개월 수익률",
                "3개월 수익률",
                "6개월 수익률",
                "12개월 수익률",
                "모멘텀 스코어",
            ],
        ),
        use_container_width=True,
    )

    st.subheader("오리지널 듀얼 모멘텀 12개월 수익률")
    st.dataframe(
        pct_cols(ot, ["12개월 수익률"]),
        use_container_width=True,
    )

    detail = pd.DataFrame(rows)

    st.subheader("하위전략별 투입 비중 및 리밸런싱 일정")
    detail_display = pct_cols(
        detail,
        [
            "하위전략 내 비중",
            "강환국 전략 전체 비중",
        ],
    )
    detail_display = date_cols_to_string(detail_display, ["최근 리밸런싱일", "다음 리밸런싱일"])
    st.dataframe(detail_display, use_container_width=True)

    final = (
        detail
        .groupby(["국내 ETF 코드", "국내 ETF명", "자산군"], as_index=False)
        .agg(
            {
                "강환국 전략 전체 비중": "sum",
                "다음 리밸런싱일": lambda x: min(v for v in x if v is not None),
                "리밸런싱 상태": lambda x: "리밸런싱 필요" if "리밸런싱 필요" in list(x) else "대기",
            }
        )
        .sort_values("강환국 전략 전체 비중", ascending=False)
    )

    st.subheader("최종 매수 비중")
    final_display = pct_cols(final, ["강환국 전략 전체 비중"])
    final_display = date_cols_to_string(final_display, ["다음 리밸런싱일"])
    st.dataframe(final_display, use_container_width=True)

    st.bar_chart(
        final.set_index("국내 ETF명")["강환국 전략 전체 비중"]
    )

    csv = final.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        "최종 비중 CSV 다운로드",
        data=csv,
        file_name=f"kang_portfolio_allocation_{actual_eval_dt.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

else:
    st.info("조회 기준, LAA 조건, 최근 리밸런싱일을 선택한 뒤 '전략 비중 계산'을 누르세요.")

    st.subheader("기본 국내 ETF 매핑")
    st.dataframe(
        pd.DataFrame(ETF_MAP),
        use_container_width=True,
    )
