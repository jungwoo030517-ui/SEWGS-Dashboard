import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================
# 1단계: 웹 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="SEWGS 순도 예측 서비스",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# 2단계: 최신 DNN 동적 아키텍처 및 자원 로드 (다중 출력: H2 & CO2)
# ==========================================
def build_model(input_dim=7, hidden_dims=[64, 32, 16], act_fn=nn.ReLU, output_dim=2):
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(act_fn())
        prev_dim = h
    # 👈 출력 노드 2개 (H2 순도, CO2 순도)
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

class SurrogateWrapper:
    def __init__(self, model, scaler_X, scaler_y):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def predict(self, X_raw):
        X_raw = np.atleast_2d(np.asarray(X_raw, dtype=np.float64))
        X_scaled = self.scaler_X.transform(X_raw)
        with torch.no_grad():
            y_scaled = self.model(torch.FloatTensor(X_scaled)).numpy()
        return self.scaler_y.inverse_transform(y_scaled)

@st.cache_resource
def load_model_assets():
    model_path = os.path.join('saved_models_tuned_CO2', 'sewgs_dnn_final_model.pth')
    sx_path = os.path.join('saved_models_tuned_CO2', 'scaler_X.pkl')
    sy_path = os.path.join('saved_models_tuned_CO2', 'scaler_y.pkl')

    # 다중 출력 아키텍처 모델 생성 (output_dim=2)
    model = build_model(input_dim=7, hidden_dims=[64, 32, 16], act_fn=nn.ReLU, output_dim=2)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        scaler_X = joblib.load(sx_path)
        scaler_y = joblib.load(sy_path)
    elif os.path.exists('sewgs_dnn_final_model.pth'):
        model.load_state_dict(torch.load('sewgs_dnn_final_model.pth', map_location=torch.device('cpu'), weights_only=True))
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
    else:
        # 파일 미존재 시 더미 예외 처리 (출력 2개 타깃 기준)
        from sklearn.preprocessing import MinMaxScaler
        scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
        dummy_X = np.array([[0.3, 0.05, 0.5, 0.1, 0.02, 0.07, 210], [0.4, 0.02, 0.4, 0.15, 0.01, 0.20, 410]])
        dummy_y = np.array([[90.0, 85.0], [99.0, 95.0]])
        scaler_X.fit(dummy_X)
        scaler_y.fit(dummy_y)

    model.eval()
    return SurrogateWrapper(model, scaler_X, scaler_y)

try:
    surrogate = load_model_assets()
except Exception as e:
    st.error(f"⚠️ 모델 및 스케일러 로드 실패! (.pth, .pkl 파일 위치 확인 필요): {e}")
    st.stop()

# ==========================================
# 3단계: 최적 피드시간(t_feed) 탐색 제어 엔진
# ==========================================
T_GRID = np.arange(210, 411, 1)          # t_feed 범위 (210~410s)
U_GRID = np.arange(0.070, 0.2001, 0.005) # u_rinse 범위 (0.07~0.20m/s)
_TT, _UU = np.meshgrid(T_GRID, U_GRID, indexing='ij')
CAND_T, CAND_U = _TT.ravel(), _UU.ravel()
N_CAND = len(CAND_T)

T_N = (CAND_T - CAND_T.min()) / (CAND_T.max() - CAND_T.min())
U_N = (CAND_U - CAND_U.min()) / (CAND_U.max() - CAND_U.min())

def draw_purity_card_html(title, purity_val, target_val, delta_val, is_h2=True):
    if is_h2:
        is_pass = purity_val >= target_val
        bar_color = "#28a745" if is_pass else "#dc3545"
        status_text = "✅ 정상운전 중" if is_pass else "🚨 비정상 탐지"
        status_bg = "rgba(40, 167, 69, 0.2)" if is_pass else "rgba(220, 53, 69, 0.2)"
        status_border = "#28a745" if is_pass else "#dc3545"
        caption_text = f"ℹ️ 고순도 수소 생산 기준: {target_val:.1f}%"
        delta_color = "#28a745" if delta_val >= 0 else "#dc3545"
    else:
        is_pass = purity_val >= target_val
        bar_color = "#28a745" if is_pass else "#ffc107"
        status_text = "🍃 포집 규격 만족" if is_pass else "⚠️ 포집 성능 주시"
        status_bg = "rgba(40, 167, 69, 0.2)" if is_pass else "rgba(255, 193, 7, 0.2)"
        status_border = "#28a745" if is_pass else "#ffc107"
        caption_text = f"ℹ️ CO₂ 포집/저장 가이드라인: {target_val:.1f}%"
        delta_color = "#28a745" if delta_val >= 0 else "#ffc107"

    # Y축 범위를 둘 다 80~100%로 완전 동일하게 고정 (95선이 90선보다 바르게 위에 그어짐)
    y_min = 70
    fill_percent = max(0, min(100, (purity_val - y_min) / (100.0 - y_min) * 100))
    target_percent = max(0, min(100, (target_val - y_min) / (100.0 - y_min) * 100))

    html_code = f"""<div style="display: flex; align-items: center; justify-content: space-between; padding: 5px 0 15px 0;">
<div>
<div style="font-size: 0.95rem; font-weight: 600; opacity: 0.7; margin-bottom: 2px; font-family: 'Arial', sans-serif;">{'H₂' if is_h2 else 'CO₂'} Dry Purity</div>
<div style="font-family: 'Arial', sans-serif; font-size: 2.6rem; font-weight: 800; line-height: 1.1; letter-spacing: -1px;">{purity_val:.2f} %</div>
<div style="font-size: 0.85rem; font-weight: bold; color: {delta_color}; margin-top: 6px; font-family: 'Arial', sans-serif;">{delta_val:+.2f} % ({'목표' if is_h2 else '기준'} {target_val:.1f}% 대비)</div>
</div>
<div style="position: relative; width: 75px; height: 85px;">
<div style="position: absolute; top: 0%; left: 0; right: 0; display: flex; align-items: center; transform: translateY(-50%);">
<span style="width: 24px; font-family: 'Arial', sans-serif; font-size: 11px; font-weight: bold; color: #64748B; text-align: right; padding-right: 4px;">100</span>
<div style="width: 5px; height: 1.5px; background-color: #64748B;"></div>
</div>
<div style="position: absolute; bottom: {target_percent}%; left: 0; right: 0; display: flex; align-items: center; transform: translateY(50%);">
<span style="width: 24px; font-family: 'Arial', sans-serif; font-size: 11px; font-weight: bold; color: #64748B; text-align: right; padding-right: 4px;">{int(target_val)}</span>
<div style="width: 5px; height: 1.5px; background-color: #64748B;"></div>
</div>
<div style="position: absolute; top: 0; bottom: 0; left: 29px; width: 1.5px; background-color: #64748B;"></div>
<div style="position: absolute; bottom: 0; left: 34px; width: 40px; height: {fill_percent}%; background-color: {bar_color}; border-radius: 3px 3px 0 0; transition: height 0.4s ease-in-out;"></div>
</div>
</div>
<div style="background-color: {status_bg}; border: 1px solid {status_border}; border-radius: 8px; padding: 12px; text-align: center; color: {status_border}; font-weight: bold; font-size: 1.35rem;">{status_text}</div>
<div style="font-size: 0.8rem; opacity: 0.6; margin-top: 6px;">{caption_text}</div>"""

    return html_code
    
def optimize_operation(comp, h2_spec=95.0, u_ref=0.07, w_u=0.2, margin=0.65):
    comp = np.asarray(comp, float).ravel()
    Xc = np.empty((N_CAND, 7))
    Xc[:, :5] = comp
    Xc[:, 5] = CAND_U
    Xc[:, 6] = CAND_T

    Y = surrogate.predict(Xc)
    
    # 안전 여유(margin) 포함 H2 순도 조건 만족 여부
    feas = Y[:, 0] >= (h2_spec + margin)

    tier1 = None # 기존 tier1은 그대로 유지하되 안 써도 무방

    if feas.any():
        # 📌 스코어링: 피드시간(T_N) 극대화 + 린스유속(U_N) 최적화 트레이드오프
        # w_u 값을 조절하여 린스 유속 절감과 피드시간 연장의 우위 설정 가능
        score = np.where(feas, T_N - w_u * U_N, -np.inf)
        k2 = int(np.argmax(score))
        
        tier2 = {
            't_feed': float(CAND_T[k2]), 
            'u_rinse': float(CAND_U[k2]), 
            'h2_pred': float(Y[k2, 0]),
            'co2_pred': float(Y[k2, 1]) if Y.shape[1] > 1 else 0.0
        }
    else:
        # 제약 만족 조건이 없을 경우 H2 순도 최고점 선택
        k2 = int(np.argmax(Y[:, 0]))
        tier2 = {
            't_feed': float(CAND_T[k2]), 
            'u_rinse': float(CAND_U[k2]), 
            'h2_pred': float(Y[k2, 0]),
            'co2_pred': float(Y[k2, 1]) if Y.shape[1] > 1 else 0.0,
            'infeasible': True
        }

    return tier1, tier2, 0.0, 0.0

# ==========================================
# 4단계: 사이드바 - 실시간 입력 파라미터
# ==========================================
st.sidebar.header("⚙️ 운전 파라미터 설정")

PRESETS = {
    "기본 운전 조건 (Base)": {
        "y_H2": 0.3962, "y_CO": 0.0514, "y_H2O": 0.4743, "y_CO2": 0.0666, "y_CH4": 0.0116,
        "u_rinse": 0.1991, "t_feed": 228.03
    },
    "고농도 수소 피드 (High-H2)": {
        "y_H2": 0.4500, "y_CO": 0.0300, "y_H2O": 0.4400, "y_CO2": 0.0700, "y_CH4": 0.0100,
        "u_rinse": 0.2200, "t_feed": 300.00
    },
    "고농도 CO 피드 (High-CO)": {
        "y_H2": 0.3500, "y_CO": 0.1000, "y_H2O": 0.4500, "y_CO2": 0.0800, "y_CH4": 0.0200,
        "u_rinse": 0.1800, "t_feed": 300.00
    }
}

if "y_H2" not in st.session_state:
    for k, v in PRESETS["기본 운전 조건 (Base)"].items():
        st.session_state[k] = v

def apply_preset(preset_name):
    for k, v in PRESETS[preset_name].items():
        st.session_state[k] = v

st.sidebar.caption("📌 **대표 피드 조성 프리셋 선택**")
for name in PRESETS.keys():
    if st.sidebar.button(f"🔹 {name}", key=f"btn_{name}", use_container_width=True):
        apply_preset(name)
        st.rerun()

st.sidebar.markdown("---")

st.sidebar.subheader("1. 피드 가스 조성 (Mole Fraction)")
y_H2  = st.sidebar.number_input("y_H2 :gray[0.28~0.42]", min_value=0.0, max_value=1.0, step=0.005, format="%.4f", key="y_H2")
y_CO  = st.sidebar.number_input("y_CO :gray[0.01~0.06]", min_value=0.0, max_value=1.0, step=0.005, format="%.4f", key="y_CO")
y_H2O = st.sidebar.number_input("y_H2O :gray[0.4~0.6]", min_value=0.0, max_value=1.0, step=0.005, format="%.4f", key="y_H2O")
y_CO2 = st.sidebar.number_input("y_CO2 :gray[0.06~0.18]", min_value=0.0, max_value=1.0, step=0.005, format="%.4f", key="y_CO2")
y_CH4 = st.sidebar.number_input("y_CH4 :gray[0.01~0.025]", min_value=0.0, max_value=1.0, step=0.001, format="%.4f", key="y_CH4")
current_comp = [y_H2, y_CO, y_H2O, y_CO2, y_CH4]

mole_sum = y_H2 + y_CO + y_H2O + y_CO2 + y_CH4
if abs(mole_sum - 1.0) > 0.01:
    st.sidebar.warning(f"⚠️ 조성 총합 = {mole_sum:.4f} (1.0 기준 점검 권장)")

# 📌 2. (Operating Conditions) 제거
st.sidebar.subheader("2. 조업 조건")

t_feed = st.sidebar.number_input(
    "반응 단계 시간 (s)",
    min_value=50.0,
    max_value=410.0,
    value=228.03,
    step=1.0,
    format="%.2f",
    key="t_feed"
)

st.sidebar.markdown(
    "<p style='color: #EF4444; font-size: 0.8rem; font-weight: bold; margin-top: -10px; margin-bottom: 15px;'>주의: 권장 범위는 210~410입니다.</p>", 
    unsafe_allow_html=True
)
u_rinse = st.sidebar.number_input(
    "린스 유속 (m/s)",
    min_value=0.0100,
    max_value=0.5000,
    value=0.1991,
    step=0.0050,
    format="%.4f",
    key="u_rinse"
)

# 📌 Rinse 유속 밑에 빨간색 안내 텍스트 추가
st.sidebar.markdown(
    "<p style='color: #EF4444; font-size: 0.8rem; font-weight: bold; margin-top: -10px; margin-bottom: 15px;'>주의: 권장 범위는 0.07~0.2입니다.</p>", 
    unsafe_allow_html=True
)


# ==========================================
# 5단계: 실시간 추론 및 메인 대시보드 화면
# ==========================================
raw_input = [y_H2, y_CO, y_H2O, y_CO2, y_CH4, u_rinse, t_feed]
preds = surrogate.predict(raw_input)[0]

# H2 및 CO2 순도 추출
pred_h2_purity = float(preds[0])
pred_co2_purity = float(preds[1]) if len(preds) > 1 else 0.0

# 기존 st.title("⚡ SEWGS 대리운전") 대체 
st.markdown("<h1 style='font-size: 2.5rem; font-weight: 800; margin-bottom: -10px;'>⚡ SEWGS 대리운전</h1>", unsafe_allow_html=True)
st.caption("PyTorch 기반 DNN 대리모델 실시간 수소/이산화탄소 순도 동시 예측 및 외란 응답형 제시 시스템")
st.divider()

# 레이아웃 비율: 좌측 요약표(1.1) / 우측 H2 순도(1.0) / 우측 CO2 순도(1.0)
col_table, col_h2, col_co2 = st.columns([1.1, 1.0, 1.0], gap="medium")

# 📌 [좌측] 입력 운전 조건 요약표
with col_table:
    st.subheader("📋 입력 운전 조건 및 현황")
    
    comp_str = f"H₂: {y_H2:.3f}, CO: {y_CO:.3f}, H₂O: {y_H2O:.3f}, CO₂: {y_CO2:.3f}, CH₄: {y_CH4:.3f}"
    
    df_summary = pd.DataFrame({
        "구분": ["입력 조성", "입력 피드시간", "입력 린스유속"],
        "값 (Value)": [
            comp_str,
            f"{t_feed:.2f} s",
            f"{u_rinse:.4f} m/s",
        ]
    })
    
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

# 📌 6단계 최적화 탐색 버튼 전용 CSS (메트릭 스타일과 분리)
st.markdown("""
    <style>
    /* 메인 버튼 전용 커스텀 스타일 */
    [data-testid="stMainBlockContainer"] div.stButton > button {
        height: 5rem !important;
        border-radius: 12px !important;
        background-color: #1F77B4 !important;
        border: 1px solid #1D4ED8 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }

    [data-testid="stMainBlockContainer"] div.stButton > button:hover {
        background-color: #155E75 !important;
        border-color: #38BDF8 !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="stMainBlockContainer"] div.stButton > button p {
        font-family: 'Arial', sans-serif !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.5px !important;
        margin: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 📌 [우측 1] H2 순도 카드
with col_h2:
    st.subheader("🎯 예측 H₂ 순도")
    delta_h2 = pred_h2_purity - 95.0
    st.markdown(
        draw_purity_card_html("H₂ Dry Purity", pred_h2_purity, 95.0, delta_h2, is_h2=True),
        unsafe_allow_html=True
    )

# 📌 [우측 2] CO2 순도 카드
with col_co2:
    st.subheader("🌱 예측 CO₂ 순도")
    CO2_SPEC = 90.0
    delta_co2 = pred_co2_purity - CO2_SPEC
    st.markdown(
        draw_purity_card_html("CO₂ Dry Purity", pred_co2_purity, CO2_SPEC, delta_co2, is_h2=False),
        unsafe_allow_html=True
    )

st.divider()

# ==========================================
# 6단계: 최적 피드시간 및 린스유속 제안 기능
# ==========================================
st.subheader("🛠️ 최적 운전 조건(반응 시간 & 린스 유속) 제안")
st.caption("가스 조성 변동 시 95.0% 순도 스펙을 만족하면서 생산성을 극대화하는 최적 피드시간과 린스 유속을 제어기가 탐색합니다.")

# 📌 메트릭 숫자는 건드리지 않고 '버튼 내 텍스트'만 정확히 타겟팅하는 수정 CSS
st.markdown("""
    <style>
    /* 1. 메인 버튼 스타일 지정 */
    [data-testid="stMainBlockContainer"] div.stButton > button {
        height: 5rem !important;
        border-radius: 12px !important;
        background-color: #1F77B4 !important;
        border: 1px solid #1D4ED8 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }

    [data-testid="stMainBlockContainer"] div.stButton > button:hover {
        background-color: #155E75 !important;
        border-color: #38BDF8 !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="stMainBlockContainer"] div.stButton > button p {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.5px !important;
        margin: 0 !important;
    }

    /* 2. 🚨 st.metric 테마 자동 반응 (color 고정 해제) */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        /* color 구문을 싹 제거하여 Streamlit 기본 테마 자동 전환 유지 */
    }
    /* 3. 📌 5번 & 6번 메트릭 숫자 영역 Arial Bold 스타일 강제 적용 */
    [data-testid="stMetricValue"] {
        font-family: 'Arial', 'Helvetica Neue', 'Helvetica', sans-serif !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;  /* Arial Bold 굵기 */
        letter-spacing: -1px !important; /* 숫자가 큼직하고 짱짱하게 들어오도록 자간 조절 */
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        /* color 구문을 싹 제거하여 Streamlit 기본 테마 자동 전환 유지 */
    }
    </style>
""", unsafe_allow_html=True)

# 📌 [위치/크기 복원] 가운데 컬럼 비율 (1.5 : 2 : 1.5 -> 전체 화면의 50% 중앙 배치)
col_b1, col_b2, col_b3 = st.columns([1.5, 2, 1.5])

with col_b2:
    btn_click = st.button("🚀 최적 운전 조건 자동 탐색", type="primary", use_container_width=True)

if btn_click:
    with st.spinner("후보 격자(1,107개 조건) 전수 추론 및 최적화 계산 중..."):
        t_start = time.perf_counter()
        _, tier2, _, _ = optimize_operation([y_H2, y_CO, y_H2O, y_CO2, y_CH4], h2_spec=95.0, u_ref=u_rinse)
        t_elapsed = (time.perf_counter() - t_start) * 1000

    st.success(f"⚡ 탐색 완료! (소요 시간: **{t_elapsed:.1f} ms**)")
    
    with st.expander("📌 **추천 운전 조건 결과 (클릭하여 접기/열기)**", expanded=True):
        if tier2 and 'infeasible' not in tier2:
            dt_from_current = tier2['t_feed'] - t_feed
            gain_pct_current = (dt_from_current / t_feed * 100) if t_feed > 0 else 0.0
            du_from_current = tier2['u_rinse'] - u_rinse

            st.markdown("##### 🔸 **생산량 극대화 최적 운전 모드**")
            
            res_m1, res_m2 = st.columns(2)
            
            with res_m1:
                st.metric(
                    label="추천 피드시간 (t_feed)", 
                    value=f"{tier2['t_feed']:.0f} s", 
                    delta=f"{dt_from_current:+.0f} s (현재 대비 {gain_pct_current:+.1f}%)"
                )
            with res_m2:
                st.metric(
                    label="추천 린스유속 (u_rinse)", 
                    value=f"{tier2['u_rinse']:.4f} m/s", 
                    delta=f"{du_from_current:+.4f} m/s (현재 대비)"
                )
            
            st.write(f"- **예상 H₂ 순도**: `{tier2['h2_pred']:.2f} %`")
            st.write(f"- **예상 CO₂ 순도**: `{tier2['co2_pred']:.2f} %`")
            
            st.info(
                f"🎯 **제어 추천 요약**: 피드 시간을 기존 **{t_feed:.0f}초 $\\rightarrow$ {tier2['t_feed']:.0f}초**, "
                f"린스 유속을 **{u_rinse:.4f} m/s $\\rightarrow$ {tier2['u_rinse']:.4f} m/s**로 제어 시, "
                f"제품 순도 **{tier2['h2_pred']:.2f}%**를 유지하며 공정 제어를 최적화합니다."
            )
        else:
            st.error("🚨 95% 순도 제약 조건을 만족하는 운전 영역이 없습니다.")

st.divider()
st.caption("Created by 대리운전 Team of University of ULSAN | Model Version: alpha | Web Dashboard was built by W.J.JEONG")
