import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

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

def optimize_operation(comp, h2_spec=95.0, u_ref=0.07, w_u=0.5, margin=0.65):
    comp = np.asarray(comp, float).ravel()
    Xc = np.empty((N_CAND, 7))
    Xc[:, :5] = comp
    Xc[:, 5] = CAND_U
    Xc[:, 6] = CAND_T

    Y = surrogate.predict(Xc)
    feas = Y[:, 0] >= (h2_spec + margin)

    u_ref_grid = float(U_GRID[np.argmin(np.abs(U_GRID - u_ref))])
    on_ref = np.isclose(CAND_U, u_ref_grid)
    m1 = feas & on_ref

    tier1 = None
    if m1.any():
        k1 = int(np.argmax(np.where(m1, CAND_T, -np.inf)))
        tier1 = {
            't_feed': float(CAND_T[k1]), 
            'u_rinse': float(CAND_U[k1]), 
            'h2_pred': float(Y[k1, 0]),
            'co2_pred': float(Y[k1, 1]) if Y.shape[1] > 1 else 0.0
        }

    if feas.any():
        score = np.where(feas, T_N - w_u * U_N, -np.inf)
        k2 = int(np.argmax(score))
        tier2 = {
            't_feed': float(CAND_T[k2]), 
            'u_rinse': float(CAND_U[k2]), 
            'h2_pred': float(Y[k2, 0]),
            'co2_pred': float(Y[k2, 1]) if Y.shape[1] > 1 else 0.0
        }
    else:
        k2 = int(np.argmax(Y[:, 0]))
        tier2 = {
            't_feed': float(CAND_T[k2]), 
            'u_rinse': float(CAND_U[k2]), 
            'h2_pred': float(Y[k2, 0]),
            'co2_pred': float(Y[k2, 1]) if Y.shape[1] > 1 else 0.0,
            'infeasible': True
        }

    gain_dt = (tier2['t_feed'] - tier1['t_feed']) if (tier1 and 'infeasible' not in tier2) else 0.0
    gain_pct = (gain_dt / tier1['t_feed'] * 100) if (tier1 and tier1['t_feed'] > 0) else 0.0

    return tier1, tier2, gain_dt, gain_pct

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
y_H2  = st.sidebar.number_input("y_H2", min_value=0.0, max_value=1.0, step=0.005, format="%.4f", key="y_H2")
y_CO  = st.sidebar.number_input("y_CO", min_value=0.0, max_value=1.0, step=0.005, format="%.4f", key="y_CO")
y_H2O = st.sidebar.number_input("y_H2O", min_value=0.0, max_value=1.0, step=0.005, format="%.4f", key="y_H2O")
y_CO2 = st.sidebar.number_input("y_CO2", min_value=0.0, max_value=1.0, step=0.005, format="%.4f", key="y_CO2")
y_CH4 = st.sidebar.number_input("y_CH4", min_value=0.0, max_value=1.0, step=0.001, format="%.4f", key="y_CH4")
current_comp = [y_H2, y_CO, y_H2O, y_CO2, y_CH4]

mole_sum = y_H2 + y_CO + y_H2O + y_CO2 + y_CH4
if abs(mole_sum - 1.0) > 0.01:
    st.sidebar.warning(f"⚠️ 조성 총합 = {mole_sum:.4f} (1.0 기준 점검 권장)")

st.sidebar.subheader("2. 조업 조건 (Operating Conditions)")

u_rinse = st.sidebar.number_input(
    "Rinse 유속 (m/s)",
    min_value=0.0100,
    max_value=0.5000,
    value=0.1991,
    step=0.0050,
    format="%.4f",
    key="u_rinse"
)

t_feed = st.sidebar.number_input(
    "Feed 스텝 시간 (s)",
    min_value=50.0,
    max_value=410.0,
    value=228.03,
    step=1.0,
    format="%.2f",
    key="t_feed"
)

# ==========================================
# 5단계: 실시간 추론 및 메인 대시보드 화면
# ==========================================
raw_input = [y_H2, y_CO, y_H2O, y_CO2, y_CH4, u_rinse, t_feed]
preds = surrogate.predict(raw_input)[0]

# H2 및 CO2 순도 추출
pred_h2_purity = float(preds[0])
pred_co2_purity = float(preds[1]) if len(preds) > 1 else 0.0

st.title("⚡ SEWGS 순도 예측 & 제어 시스템")
st.caption("PyTorch 기반 DNN 대리모델 실시간 수소/이산화탄소 순도 동시 예측 및 외란 응답형 제어 시스템")
st.divider()

# 레이아웃 비율: 좌측 요약표(1.1) / 우측 H2 순도(1.0) / 우측 CO2 순도(1.0)
col_table, col_h2, col_co2 = st.columns([1.1, 1.0, 1.0], gap="medium")

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

# 📌 st.metric 내부 숫자 크기 및 넙덕한 기계식 폰트 적용 CSS
st.markdown("""
    <style>
    /* 메트릭 값(숫자) 영역 타깃팅 */
    div[data-testid="stMetricValue"] {
        font-size: 3.2rem !important;                     /* 숫자 크기 확대 */
        font-family: 'Arial', monospace !important;       /* 넙덕한 폰트 */
        font-weight: 900 !important;                      /* 아주 굵게 */
        letter-spacing: -1px !important;                 /* 자간 촘촘하게 */
        color: #ffffff !important;                       /* 숫자 색상 (밝게) */
    }

    /* 메트릭 라벨 글자 크기 조절 */
    div[data-testid="stMetricLabel"] {
        font-size: 1.15rem !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# 1) H2 순도 카드 (우측 첫 번째)
with col_h2:
    st.subheader("🎯 예측 H₂ 순도")
    
    delta_h2 = pred_h2_purity - 95.0
    st.metric(
        label="H₂ Dry Purity",
        value=f"{pred_h2_purity:.2f} %",
        delta=f"{delta_h2:+.2f} % (목표 95.0% 대비)"
    )

    if pred_h2_purity >= 95.0:
        st.markdown(
            """
            <div style="background-color: rgba(40, 167, 69, 0.2); border: 1px solid #28a745; border-radius: 8px; padding: 15px; text-align: center; color: #28a745; font-weight: bold; font-size: 1.35rem;">
                ✅ 정상운전 중
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color: rgba(220, 53, 69, 0.2); border: 1px solid #dc3545; border-radius: 8px; padding: 15px; text-align: center; color: #dc3545; font-weight: bold; font-size: 1.35rem;">
                🚨 비정상 탐지
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.caption("ℹ️ 고순도 수소 생산 기준: 95.0%")

# 2) CO2 순도 카드 (우측 두 번째 - 신규 추가 ✨)
with col_co2:
    st.subheader("🌱 예측 CO₂ 순도")
    
    # 예시 CO2 목표 기준 90.0% 설정
    CO2_SPEC = 90.0
    delta_co2 = pred_co2_purity - CO2_SPEC
    st.metric(
        label="CO₂ Dry Purity",
        value=f"{pred_co2_purity:.2f} %",
        delta=f"{delta_co2:+.2f} % (기준 {CO2_SPEC:.1f}% 대비)"
    )

    if pred_co2_purity >= CO2_SPEC:
        st.markdown(
            """
            <div style="background-color: rgba(40, 167, 69, 0.2); border: 1px solid #28a745; border-radius: 8px; padding: 15px; text-align: center; color: #28a745; font-weight: bold; font-size: 1.35rem;">
                🍃 포집 규격 만족
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color: rgba(255, 193, 7, 0.2); border: 1px solid #ffc107; border-radius: 8px; padding: 15px; text-align: center; color: #ffc107; font-weight: bold; font-size: 1.35rem;">
                ⚠️ 포집 성능 주시
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.caption(f"ℹ️ CO₂ 포집/저장 가이드라인: {CO2_SPEC:.1f}%")

st.divider()

# ==========================================
# 6단계: 최적 피드시간 탐색 기능
# ==========================================
st.subheader("🛠️ 외란 응답형 최적 피드시간 제안")
st.caption("가스 조성 변동 시 95.0% 순도 스펙을 만족하면서 생산성을 극대화하는 최적 피드시간을 제어기가 탐색합니다.")

st.markdown("""
    <style>
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
        transform: translateY(-2px);
    }

    [data-testid="stMainBlockContainer"] div.stButton > button p,
    [data-testid="stMainBlockContainer"] div.stButton > button div[data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.5px !important;
    }
    </style>
""", unsafe_allow_html=True)

col_b1, col_b2, col_b3 = st.columns([1.5, 2, 1.5])

with col_b2:
    btn_click = st.button("🚀 최적 피드시간 자동 탐색", type="primary", use_container_width=True)

if btn_click:
    with st.spinner("후보 격자(1,107개 조건) 전수 추론 및 Tier 1/2 최적화 계산 중..."):
        t_start = time.perf_counter()
        tier1, tier2, gain_dt, gain_pct = optimize_operation([y_H2, y_CO, y_H2O, y_CO2, y_CH4], h2_spec=95.0, u_ref=u_rinse)
        t_elapsed = (time.perf_counter() - t_start) * 1000

    st.success(f"⚡ 탐색 완료! (소요 시간: **{t_elapsed:.1f} ms**)")
    
    with st.expander("📌 **추천 운전 조건 결과 (클릭하여 접기/열기)**", expanded=True):
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("##### 🔹 **[1안] 린스 고정 - 피드시간 연장 모드**")
            if tier1:
                st.metric("추천 t_feed", f"{tier1['t_feed']:.0f} s", delta=f"{tier1['t_feed'] - t_feed:+.0f} s (현재 대비)")
                st.write(f"- **고정 린스 유속 ($u_{{rinse}}$)**: `{tier1['u_rinse']:.4f} m/s`")
                st.write(f"- **예상 H₂ 순도 / CO₂ 순도**: `{tier1['h2_pred']:.2f} %` / `{tier1['co2_pred']:.2f} %`")
            else:
                st.error("⚠️ 현재 린스 유속으로는 95% 순도 달성이 불가능합니다.")

        with res_col2:
            st.markdown("##### 🔸 **[2안] 생산량 극대화 모드 (권장)**")
            if 'infeasible' not in tier2:
                st.metric("추천 t_feed", f"{tier2['t_feed']:.0f} s", delta=f"+{gain_dt:.0f} s (Tier 1 대비 +{gain_pct:.1f}%)")
                st.write(f"- **최적 린스 유속 ($u_{{rinse}}$)**: `{tier2['u_rinse']:.4f} m/s`")
                st.write(f"- **예상 H₂ 순도 / CO₂ 순도**: `{tier2['h2_pred']:.2f} %` / `{tier2['co2_pred']:.2f} %`")
            else:
                st.error("🚨 제약 조건을 만족하는 운전 영역이 없습니다.")

        if tier2 and 'infeasible' not in tier2:
            st.info(
                f"🎯 **제어 추천 요약**: 피드 시간을 기존 **{t_feed:.0f}초**에서 **{tier2['t_feed']:.0f}초**로 변경 시, "
                f"제품 순도 **{tier2['h2_pred']:.2f}%**를 유지하면서 **생산 시간을 약 {gain_pct:.1f}% 향상**시킬 수 있습니다."
            )

st.divider()
st.caption("Created by 대리운전 Team of University of ULSAN | Model Version: alpha | Web Dashboard was built by W.J.JEONG")