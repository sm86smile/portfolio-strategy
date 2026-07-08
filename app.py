import math
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import requests
import streamlit as st


# ============================================================
# 강환국 전략 ETF 자산배분 앱
# - KRX Open API: ETF 일별매매정보
# - 국내상장 ETF 대체 매핑 기반
# - LAA / VAA 공격형 / 오리지널 듀얼 모멘텀 계산
# ============================================================

DEFAULT_KRX_URL = "http://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"

DEFAULT_ETF_MAP = [
    {
        "original": "SPY",
        "asset": "미국 S&P500",
        "krx_code": "360750",
        "krx_name": "TIGER 미국S&P500",
        "used_in": "VAA, ODM",
    },
    {
        "original": "QQQ",
        "asset": "미국 나스닥100",
        "krx_code": "133690",
        "krx_name": "TIGER 미국나스닥100",
        "used_in": "LAA",
    },
    {
        "original": "IWD",
        "asset": "미국 대형 가치주",
        "krx_code": "0127M0",
        "krx_name": "ACE 미국대형가치주액티브",
        "used_in": "LAA",
    },
    {
        "original": "GLD",
        "asset": "금",
        "krx_code": "411060",
        "krx_name": "ACE KRX금현물",
        "used_in": "LAA",
    },
    {
        "original": "IEF",
        "asset": "미국 중기국채",
        "krx_code": "305080",
        "krx_name": "TIGER 미국채10년선물",
        "used_in": "VAA, LAA",
    },
    {
        "original": "SHY",
        "asset": "미국 단기채",
        "krx_code": "329750",
        "krx_name": "TIGER 미국달러단기채권액티브",
        "used_in": "VAA, LAA",
    },
    {
        "original": "BIL",
        "asset": "미국 초단기채/현금성 비교 기준",
        "krx_code": "329750",
        "krx_name": "TIGER 미국달러단기채권액티브",
        "used_in": "ODM",
    },
    {
        "original": "EFA",
        "asset": "미국 제외 선진국",
        "krx_code": "195970",
        "krx_name": "PLUS 선진국MSCI(합성 H)",
        "used_in": "VAA, ODM",
    },
    {
        "original": "EEM",
        "asset": "신흥국",
        "krx_code": "195980",
        "krx_name": "PLUS 신흥국MSCI(합성 H)",
        "used_in": "VAA",
    },
    {
        "original": "AGG",
        "asset": "미국 종합채권",
        "krx_code": "437080",
        "krx_name": "KODEX 미국종합채권ESG액티브(H)",
        "used_in": "VAA, ODM",
    },
    {
        "original": "LQD",
        "asset": "미국 투자등급 회사채",
        "krx_code": "458260",
        "krx_name": "TIGER 미국투자등급회사채액티브(H)",
        "used_in": "VAA",
    },
]


VAA_ATTACK = ["SPY", "EFA", "EEM", "AGG"]
VAA_SAFE = ["LQD", "IEF", "SHY"]
LAA_FIXED = ["IWD", "GLD", "IEF"]
ODM_ASSETS = ["SPY", "EFA", "BIL", "AGG"]


def clean_number(value):
    """KRX 문자열 숫자를 float로 변환."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace(",", "").replace("%", "").strip()
    if s in ("", "-", "N/A", "nan", "None"):
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def normalize_krx_rows(json_data):
    """KRX 응답 구조가 조금 달라도 OutBlock 목록을 찾아 DataFrame으로 변환."""
    if not isinstance(json_data, dict):
        return pd.DataFrame()

    rows = None
    for key in ["OutBlock_1", "OutBlock1", "OutBlock", "output", "result"]:
        if key in json_data and isinstance(json_data[key], list):
            rows = json_data[key]
            break

    if rows is None:
        for value in json_data.values():
            if isinstance(value, list):
                rows = value
                break

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_krx_etf_one_day(auth_key, krx_url, ymd):
    """기준일자 1일의 KRX ETF 전체 일별매매정보를 조회."""
    response = requests.get(
        krx_url,
        params={"basDd": ymd},
        headers={"AUTH_KEY": auth_key},
        timeout=20,
    )
    response.raise_for_status()

    try:
        json_data = response.json()
    except Exception as exc:
        raise RuntimeError(f"KRX 응답을 JSON으로 해석하지 못했습니다. 응답 일부: {response.text[:200]}") from exc

    df = normalize_krx_rows(json_data)
    if df.empty:
        return df

    # 필요한 표준 컬럼이 없는 경우 대비
    required_cols = ["BAS_DD", "ISU_CD", "ISU_NM", "TDD_CLSPRC"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"KRX 응답에 필요한 컬럼이 없습니다: {missing}. 실제 컬럼: {list(df.columns)}")

    keep_cols = [
        col for col in [
            "BAS_DD", "ISU_CD", "ISU_NM", "TDD_CLSPRC", "ACC_TRDVOL", "ACC_TRDVAL",
            "NAV", "FLUC_RT", "IDX_IND_NM", "OBJ_STKPRC_IDX"
        ] if col in df.columns
    ]

    df = df[keep_cols].copy()
    df["ISU_CD"] = df["ISU_CD"].astype(str).str.strip()
    df["ISU_NM"] = df["ISU_NM"].astype(str).str.strip()
    df["close"] = df["TDD_CLSPRC"].apply(clean_number)

    if "ACC_TRDVAL" in df.columns:
        df["trading_value"] = df["ACC_TRDVAL"].apply(clean_number)
    else:
        df["trading_value"] = math.nan

    df["date"] = pd.to_datetime(df["BAS_DD"].astype(str), format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["date", "close"])


def fetch_price_history(auth_key, krx_url, codes, start_dt, end_dt):
    """기간 내 영업일을 돌면서 ETF 일별매매정보를 받아 필요한 코드만 저장."""
    codes = set(str(c).strip() for c in codes if str(c).strip())
    all_rows = []

    days = pd.date_range(start=start_dt, end=end_dt, freq="B")
    progress = st.progress(0, text="KRX ETF 데이터를 조회하는 중입니다.")

    for i, d in enumerate(days):
        ymd = d.strftime("%Y%m%d")
        try:
            day_df = fetch_krx_etf_one_day(auth_key, krx_url, ymd)
        except Exception as exc:
            st.warning(f"{ymd} 조회 중 오류: {exc}")
            day_df = pd.DataFrame()

        if not day_df.empty:
            filtered = day_df[day_df["ISU_CD"].isin(codes)].copy()
            if not filtered.empty:
                all_rows.append(filtered)

        progress.progress((i + 1) / len(days), text=f"KRX ETF 데이터 조회 중... {ymd}")

    progress.empty()

    if not all_rows:
        return pd.DataFrame()

    hist = pd.concat(all_rows, ignore_index=True)
    hist = hist.sort_values(["ISU_CD", "date"]).drop_duplicates(["ISU_CD", "date"], keep="last")
    return hist


def build_original_to_krx_map(etf_map_df):
    """원 ETF 티커 -> 국내 ETF 코드/명 매핑."""
    out = {}
    for _, row in etf_map_df.iterrows():
        original = str(row["original"]).strip().upper()
        out[original] = {
            "code": str(row["krx_code"]).strip(),
            "name": str(row["krx_name"]).strip(),
            "asset": str(row["asset"]).strip(),
        }
    return out


def latest_available_date(hist):
    if hist.empty:
        return None
    return hist["date"].max()


def price_on_or_before(hist, code, target_dt):
    """특정 코드의 target_dt 이하 가장 가까운 거래일 종가."""
    sub = hist[(hist["ISU_CD"] == code) & (hist["date"] <= pd.Timestamp(target_dt))].sort_values("date")
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return {
        "date": row["date"],
        "price": float(row["close"]),
        "name": row["ISU_NM"],
    }


def calc_return(hist, code, eval_dt, months):
    now = price_on_or_before(hist, code, eval_dt)
    prev_target = pd.Timestamp(eval_dt) - relativedelta(months=months)
    prev = price_on_or_before(hist, code, prev_target)

    if now is None or prev is None or prev["price"] == 0 or math.isnan(prev["price"]):
        return None

    return {
        "code": code,
        "name": now["name"],
        "eval_date": now["date"],
        "base_date": prev["date"],
        "eval_price": now["price"],
        "base_price": prev["price"],
        "return": now["price"] / prev["price"] - 1,
    }


def calc_momentum_table(hist, original_map, originals, eval_dt):
    """1/3/6/12개월 수익률과 VAA 모멘텀 스코어 계산."""
    rows = []
    for original in originals:
        code = original_map[original]["code"]
        item = {
            "original": original,
            "code": code,
            "name": original_map[original]["name"],
            "asset": original_map[original]["asset"],
        }

        ok = True
        for m in [1, 3, 6, 12]:
            r = calc_return(hist, code, eval_dt, m)
            value = None if r is None else r["return"]
            item[f"r{m}m"] = value
            item[f"base_date_{m}m"] = None if r is None else r["base_date"].date()
            if value is None:
                ok = False

        if ok:
            item["momentum_score"] = (
                12 * item["r1m"] +
                4 * item["r3m"] +
                2 * item["r6m"] +
                1 * item["r12m"]
            )
        else:
            item["momentum_score"] = None

        rows.append(item)

    return pd.DataFrame(rows)


def calc_12m_return_table(hist, original_map, originals, eval_dt):
    rows = []
    for original in originals:
        code = original_map[original]["code"]
        r = calc_return(hist, code, eval_dt, 12)
        rows.append({
            "original": original,
            "code": code,
            "name": original_map[original]["name"],
            "asset": original_map[original]["asset"],
            "r12m": None if r is None else r["return"],
            "base_date": None if r is None else r["base_date"].date(),
            "eval_date": None if r is None else r["eval_date"].date(),
        })
    return pd.DataFrame(rows)


def allocation_rows(strategy_name, strategy_weight, selected_allocations, original_map):
    """하위전략 내 비중을 강환국 전략 전체 비중으로 환산."""
    rows = []
    for original, inner_weight in selected_allocations.items():
        code = original_map[original]["code"]
        rows.append({
            "하위전략": strategy_name,
            "원 ETF": original,
            "국내 ETF 코드": code,
            "국내 ETF명": original_map[original]["name"],
            "자산군": original_map[original]["asset"],
            "하위전략 내 비중": inner_weight,
            "강환국 전략 전체 비중": strategy_weight * inner_weight,
        })
    return rows


def format_pct(x):
    if x is None or pd.isna(x):
        return "-"
    return f"{x * 100:.2f}%"


st.set_page_config(page_title="강환국 전략 ETF 자산배분", layout="wide")

st.title("강환국 전략 ETF 자산배분 앱")
st.caption("LAA · VAA 공격형 · 오리지널 듀얼 모멘텀을 국내상장 ETF로 대체해 비중을 계산합니다.")

with st.sidebar:
    st.header("1) KRX Open API 설정")
    auth_key = st.text_input("KRX Open API 인증키", type="password")
    krx_url = st.text_input("ETF 일별매매정보 URL", value=DEFAULT_KRX_URL)

    st.header("2) 조회 기준")
    today = date.today()
    eval_date = st.date_input("평가 기준일", value=today)
    lookback_months = st.slider("데이터 조회기간", min_value=13, max_value=36, value=15, help="12개월 수익률 계산을 위해 최소 13개월 이상 필요합니다.")

    st.header("3) LAA 수동 조건")
    laa_defensive = st.radio(
        "S&P500 < 200일선 AND 미국 실업률 > 12개월 평균?",
        options=[False, True],
        format_func=lambda x: "O: 방어 조건 충족 → SHY" if x else "X: 방어 조건 아님 → QQQ",
        index=0,
    )

    st.header("4) 하위전략 비중")
    w_laa = st.number_input("LAA 비중", min_value=0.0, max_value=1.0, value=1/3, step=0.01, format="%.4f")
    w_vaa = st.number_input("VAA 공격형 비중", min_value=0.0, max_value=1.0, value=1/3, step=0.01, format="%.4f")
    w_odm = st.number_input("오리지널 듀얼 모멘텀 비중", min_value=0.0, max_value=1.0, value=1/3, step=0.01, format="%.4f")

    st.header("5) VAA 0점 처리")
    zero_is_defensive = st.checkbox("모멘텀 스코어 0점은 방어 신호로 처리", value=True)

weight_sum = w_laa + w_vaa + w_odm
if abs(weight_sum - 1.0) > 1e-6:
    st.warning(f"하위전략 비중 합계가 {weight_sum:.4f}입니다. 앱에서는 자동으로 100% 기준으로 정규화합니다.")
    if weight_sum > 0:
        w_laa, w_vaa, w_odm = w_laa / weight_sum, w_vaa / weight_sum, w_odm / weight_sum

st.subheader("국내상장 ETF 대체 매핑")
st.write("필요하면 종목코드나 ETF명을 직접 수정할 수 있습니다. 예: 같은 자산군의 다른 운용사 ETF로 교체")
etf_map_df = st.data_editor(
    pd.DataFrame(DEFAULT_ETF_MAP),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "original": "원 ETF",
        "asset": "자산군",
        "krx_code": "국내 ETF 코드",
        "krx_name": "국내 ETF명",
        "used_in": "사용 전략",
    },
)

original_map = build_original_to_krx_map(etf_map_df)
needed_originals = sorted(set(["SPY", "QQQ", "IWD", "GLD", "IEF", "SHY", "BIL", "EFA", "EEM", "AGG", "LQD"]))
missing_originals = [x for x in needed_originals if x not in original_map or not original_map[x]["code"]]

if missing_originals:
    st.error(f"필수 원 ETF 매핑이 없습니다: {missing_originals}")
    st.stop()

if not auth_key:
    st.info("왼쪽 사이드바에 KRX Open API 인증키를 입력한 뒤 실행하세요.")
    st.stop()

if st.button("전략 비중 계산", type="primary"):
    start_dt = pd.Timestamp(eval_date) - relativedelta(months=lookback_months)
    end_dt = pd.Timestamp(eval_date)

    codes = [original_map[o]["code"] for o in needed_originals]
    hist = fetch_price_history(auth_key, krx_url, codes, start_dt, end_dt)

    if hist.empty:
        st.error("조회된 ETF 데이터가 없습니다. 인증키, URL, 종목코드, 평가 기준일을 확인하세요.")
        st.stop()

    actual_eval_dt = latest_available_date(hist)
    st.success(f"실제 계산 기준일: {actual_eval_dt.date()}")

    # 원 ETF와 국내 ETF 코드 중복 처리 확인
    latest_snapshot = (
        hist[hist["date"] == actual_eval_dt]
        .sort_values("ISU_CD")
        [["ISU_CD", "ISU_NM", "close", "trading_value"]]
    )
    with st.expander("계산 기준일 국내 ETF 가격 확인"):
        st.dataframe(latest_snapshot, use_container_width=True)

    all_alloc_rows = []

    # ------------------------------------------------------------
    # 1. LAA
    # ------------------------------------------------------------
    laa_variable = "SHY" if laa_defensive else "QQQ"
    laa_inner_alloc = {
        "IWD": 0.25,
        "GLD": 0.25,
        "IEF": 0.25,
        laa_variable: 0.25,
    }
    all_alloc_rows.extend(allocation_rows("LAA", w_laa, laa_inner_alloc, original_map))

    # ------------------------------------------------------------
    # 2. VAA 공격형
    # ------------------------------------------------------------
    vaa_table = calc_momentum_table(hist, original_map, VAA_ATTACK + VAA_SAFE, actual_eval_dt)

    if vaa_table["momentum_score"].isna().any():
        st.warning("VAA 모멘텀 스코어 계산에 필요한 1/3/6/12개월 가격 데이터가 부족한 ETF가 있습니다.")

    attack_table = vaa_table[vaa_table["original"].isin(VAA_ATTACK)].copy()
    safe_table = vaa_table[vaa_table["original"].isin(VAA_SAFE)].copy()

    if zero_is_defensive:
        attack_ok = (attack_table["momentum_score"] > 0).all()
    else:
        attack_ok = (attack_table["momentum_score"] >= 0).all()

    if attack_ok:
        vaa_pool = attack_table
        vaa_mode = "공격자산 4개 모멘텀 양호 → 공격자산 중 최고 점수 선택"
    else:
        vaa_pool = safe_table
        vaa_mode = "공격자산 중 방어 신호 발생 → 안전자산 중 최고 점수 선택"

    vaa_pool_valid = vaa_pool.dropna(subset=["momentum_score"])
    if vaa_pool_valid.empty:
        st.error("VAA 선택 자산을 계산할 수 없습니다. 조회기간 또는 ETF 매핑을 확인하세요.")
        st.stop()

    vaa_selected = vaa_pool_valid.sort_values("momentum_score", ascending=False).iloc[0]["original"]
    all_alloc_rows.extend(allocation_rows("VAA 공격형", w_vaa, {vaa_selected: 1.0}, original_map))

    # ------------------------------------------------------------
    # 3. 오리지널 듀얼 모멘텀
    # ------------------------------------------------------------
    odm_table = calc_12m_return_table(hist, original_map, ODM_ASSETS, actual_eval_dt)

    def get_odm_return(original):
        s = odm_table.loc[odm_table["original"] == original, "r12m"]
        if s.empty or pd.isna(s.iloc[0]):
            return None
        return float(s.iloc[0])

    spy_r = get_odm_return("SPY")
    efa_r = get_odm_return("EFA")
    bil_r = get_odm_return("BIL")

    if spy_r is None or efa_r is None or bil_r is None:
        st.error("오리지널 듀얼 모멘텀 계산에 필요한 SPY/EFA/BIL 12개월 수익률 데이터가 부족합니다.")
        st.stop()

    if spy_r > bil_r:
        odm_selected = "SPY" if spy_r >= efa_r else "EFA"
        odm_reason = "SPY 12개월 수익률 > BIL 대체 ETF 12개월 수익률 → SPY/EFA 중 강한 자산 선택"
    else:
        odm_selected = "AGG"
        odm_reason = "SPY 12개월 수익률 ≤ BIL 대체 ETF 12개월 수익률 → AGG 선택"

    all_alloc_rows.extend(allocation_rows("오리지널 듀얼 모멘텀", w_odm, {odm_selected: 1.0}, original_map))

    # ------------------------------------------------------------
    # 결과 출력
    # ------------------------------------------------------------
    st.subheader("하위전략별 선택 결과")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("LAA 변동 25%", f"{laa_variable} → {original_map[laa_variable]['name']}")
        st.caption("수동 조건이 O이면 SHY, X이면 QQQ")

    with col2:
        st.metric("VAA 선택", f"{vaa_selected} → {original_map[vaa_selected]['name']}")
        st.caption(vaa_mode)

    with col3:
        st.metric("오리지널 듀얼 모멘텀 선택", f"{odm_selected} → {original_map[odm_selected]['name']}")
        st.caption(odm_reason)

    st.subheader("VAA 모멘텀 스코어")
    vaa_display = vaa_table.copy()
    for col in ["r1m", "r3m", "r6m", "r12m", "momentum_score"]:
        vaa_display[col] = vaa_display[col].map(format_pct)
    st.dataframe(vaa_display, use_container_width=True)

    st.subheader("오리지널 듀얼 모멘텀 12개월 수익률")
    odm_display = odm_table.copy()
    odm_display["r12m"] = odm_display["r12m"].map(format_pct)
    st.dataframe(odm_display, use_container_width=True)

    alloc_df = pd.DataFrame(all_alloc_rows)

    st.subheader("하위전략별 투입 비중")
    display_alloc = alloc_df.copy()
    display_alloc["하위전략 내 비중"] = display_alloc["하위전략 내 비중"].map(format_pct)
    display_alloc["강환국 전략 전체 비중"] = display_alloc["강환국 전략 전체 비중"].map(format_pct)
    st.dataframe(display_alloc, use_container_width=True)

    final_alloc = (
        alloc_df
        .groupby(["국내 ETF 코드", "국내 ETF명", "자산군"], as_index=False)["강환국 전략 전체 비중"]
        .sum()
        .sort_values("강환국 전략 전체 비중", ascending=False)
    )

    st.subheader("최종 매수 비중")
    final_display = final_alloc.copy()
    final_display["강환국 전략 전체 비중"] = final_display["강환국 전략 전체 비중"].map(format_pct)
    st.dataframe(final_display, use_container_width=True)

    chart_df = final_alloc.set_index("국내 ETF명")["강환국 전략 전체 비중"]
    st.bar_chart(chart_df)

    csv = final_alloc.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "최종 비중 CSV 다운로드",
        data=csv,
        file_name=f"kang_strategy_allocation_{actual_eval_dt.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    with st.expander("리밸런싱 메모"):
        st.markdown(
            """
            - **LAA**
              - IWD, GLD, IEF 대체 ETF: 연 1회 리밸런싱
              - QQQ/SHY 대체 ETF: 월 1회 수동 조건에 따라 리밸런싱
            - **VAA 공격형**
              - 매월 말 1·3·6·12개월 수익률로 모멘텀 스코어 계산
              - 공격자산이 모두 양호하면 최고 공격자산, 아니면 최고 안전자산 선택
            - **오리지널 듀얼 모멘텀**
              - 매월 말 SPY, EFA, BIL 대체 ETF의 12개월 수익률 비교
              - 방어 신호 시 AGG 대체 ETF 선택
            """
        )
