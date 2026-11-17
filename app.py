"""
app.py
제주 감귤 경매 & 정가수의 AI 가격·거래량 예측 Streamlit 플랫폼
- 품목별 전체 거래량 예측치 수치 고정 (규격 변경 시 거래량 불변 보장)
- 거래량 순위(Feature Importance) 제거 및 월별 품종별 거래 비중(%) 차트 신규 탑재
- 실행 당일 날짜 기준 동적 향후 7일 예측
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pipeline
import utils

# 1. Streamlit 페이지 설정
st.set_page_config(
    page_title="제주 감귤 AI 가격·거래량 예측 플랫폼",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일링
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {
        font-family: 'Pretendard', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.06);
    }
    .metric-title {
        font-size: 0.88rem;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-delta-pos {
        color: #e03131;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #1971c2;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-today {
        background-color: #ff922b;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .banner-box {
        background: linear-gradient(90deg, #fff4e6 0%, #ffe8cc 100%);
        border-left: 5px solid #fd7e14;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 2. 당일 날짜 동적 취득
current_date = pipeline.get_current_date()
korean_weekdays = ['월', '화', '수', '목', '금', '토', '일']
today_str = current_date.strftime('%Y-%m-%d')
today_weekday_str = korean_weekdays[current_date.dayofweek]

# 3. 캐시를 활용한 데이터 로더 함수 (캐시 버전 갱신으로 언패킹 에러 원천 차단)
@st.cache_data(show_spinner="전국 감귤 도매시장 및 기상 데이터를 불러오는 중...")
def load_all_market_and_weather_data(_version=2):
    auction_df, private_df, weather_df = pipeline.load_raw_data()
    sub_matrix = pipeline.build_substitute_price_matrix(auction_df)
    monthly_pct, monthly_vol = pipeline.get_monthly_citrus_share(auction_df)
    return auction_df, private_df, weather_df, sub_matrix, monthly_pct, monthly_vol

auction_df, private_df, weather_df, sub_matrix, monthly_pct, monthly_vol = load_all_market_and_weather_data()

# 4. 사이드바 구성
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1557800636-894a64c1696f?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.title("🍊 감귤 예측 설정")
    st.markdown("분석 대상 품목과 규격을 선택하세요.")
    
    st.subheader("1. 감귤 품목 선택")
    citrus_options = ['하우스감귤', '비가림감귤', '노지온주', '만감류']
    selected_citrus = st.selectbox(
        "품목 (Category)",
        options=citrus_options,
        index=0,
        key="select_citrus_category",
        help="분석 대상 감귤 품목을 선택합니다. (거래량은 선택 품목의 전체 출하량 기준이며 규격 변경 시에도 동일하게 유지됩니다.)"
    )
    
    st.subheader("2. 규격 선택")
    size_options = ['소형', '중형', '대형', '가공용']
    size_desc = {
        '소형': '3.5kg 이하',
        '중형': '3.5kg 초과 ~ 7.5kg 이하',
        '대형': '7.5kg 초과 ~ 12kg 이하',
        '가공용': '12kg 초과'
    }
    selected_size = st.selectbox(
        "규격 (Size Grade)",
        options=size_options,
        index=1,
        key="select_size_grade",
        format_func=lambda x: f"{x} ({size_desc[x]})",
        help="가격 분석을 위한 규격을 선택합니다."
    )

    st.subheader("3. 품질 등급")
    grade_options = ['특', '전체', '상', '중']
    selected_grade = st.selectbox(
        "품질 등급",
        options=grade_options,
        index=0,
        key="select_quality_grade",
        help="기본값 '특'은 시장의 대표 기준 규격입니다."
    )
    
    st.markdown("---")
    st.markdown(f"**📅 실행 당일 기준일자:** `{today_str} ({today_weekday_str})`")
    st.info("💡 **가격·거래량 연계 예측 안내:**\n- **1단계 (품목 전체 거래량 모델):** 선택 품목의 전체 출하 거래량을 예측합니다. (규격을 변경해도 거래량 수치는 변하지 않고 고정 유지)\n- **2단계 (품목·규격별 가격 모델):** 1단계에서 예측된 거래량(`volume_feature`)을 핵심 수급 피처로 투입하고, 선택된 규격의 고유 시계열 가격을 학습하여 규격별로 차별화된 가격을 정밀 예측합니다.")

# 모델 실행 및 캐싱
@st.cache_data(show_spinner="품목 전체 거래량 연계 규격별 가격 LightGBM 모델 학습 및 7일 예측 수행 중...")
def run_citrus_forecast_pipeline(citrus, size, grade, execution_date_str, _version=4):
    # 1. 경매 데이터 및 모델 (품목·규격별 예측 가격을 거래량 모델의 핵심 피처로 투입)
    auc_res = pipeline.run_auction_pipeline(
        auction_df, citrus, size, grade, weather_df, sub_matrix
    )

    # 2. 거래량 팩 (정가수의용)
    vol_pack = auc_res['vol_pack'] if auc_res and 'vol_pack' in auc_res else pipeline.get_citrus_volume_data_and_model(
        auction_df, citrus, weather_df, sub_matrix, target_size=size, grade=grade
    )

    # 3. 정가수의 데이터 및 모델
    priv_res = pipeline.run_private_pipeline(
        private_df, citrus, size, grade, weather_df, sub_matrix, vol_pack
    )

    # 4. 7일 통합 예측
    combined_7d = pipeline.get_combined_7day_forecast(auc_res, priv_res)

    # 5. 4대 품목 7일 가격 비교 (선택 품목 결과 재활용)
    comp_4cats = pipeline.get_all_citrus_7day_comparison(
        auction_df, citrus, auc_res, size, grade, weather_df, sub_matrix
    )

    warn_msg = auc_res.get('warning') if auc_res else None
    return auc_res, priv_res, combined_7d, comp_4cats, warn_msg

auc_res, priv_res, combined_7d, comp_4cats, warn_msg = run_citrus_forecast_pipeline(
    selected_citrus, selected_size, selected_grade, today_str
)

# 메인 헤더 영역
st.title("🍊 제주 감귤 도매시장 AI 가격 & 거래량 예측 시스템")
st.caption(f"선택 조건: **{selected_citrus}** (전체 거래량 기준) | 규격: **{selected_size} ({size_desc[selected_size]})** | 등급: **{selected_grade}** | 분석 엔진: LightGBM Regressor")

if warn_msg:
    st.warning(warn_msg)

# 5. 탭 구성
tab_forecast, tab_auction, tab_private, tab_market = st.tabs([
    f"🔮 [당일 기준: {today_str}] 향후 7일 예측",
    "🔨 경매 예측 (품목 전체 거래량 & 가격)",
    "🤝 정가수의 예측 (가격 중심 분석)",
    "📊 감귤 시장 통계 & 상관관계"
])

# ==============================================================================
# TAB 1: [첫 번째 페이지] 실행 당일 날짜 기준 향후 7일간의 예측
# ==============================================================================
with tab_forecast:
    st.markdown(f"""
    <div class="banner-box">
        <span class="badge-today">실행 당일 기준: {today_str} ({today_weekday_str})</span>
        <h3 style="margin-top: 8px; margin-bottom: 4px; color: #d9480f;">
            오늘({today_str}) 기준 향후 7일간의 경매 및 정가수의 예측 리포트
        </h3>
        <p style="margin: 0; color: #495057; font-size: 0.93rem;">
            도매시장 거래 이력과 제주 기상(기온·강수량), 4대 대체재 가격, 시장 영업 캘린더를 종합 분석하여 앞으로 7일간의 
            <b>{selected_citrus} 전체 거래량</b>과 <b>{selected_size} 규격 예상 가격</b>을 실시간 제공합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if combined_7d is not None and not combined_7d.empty:
        vol_col = '경매_품목총거래량' if '경매_품목총거래량' in combined_7d.columns else '경매_예상거래량'

        today_match = combined_7d[combined_7d['date'] == today_str]
        today_row = today_match.iloc[0] if not today_match.empty else combined_7d.iloc[0]
        
        # 내일 날짜 행 찾기 (오늘 날짜 다음 행)
        if not today_match.empty:
            today_idx = today_match.index[0]
            tomorrow_row = combined_7d.loc[today_idx + 1] if (today_idx + 1) in combined_7d.index else (combined_7d.iloc[1] if len(combined_7d) > 1 else today_row)
        else:
            tomorrow_row = combined_7d.iloc[1] if len(combined_7d) > 1 else combined_7d.iloc[0]

        avg_auc_price = combined_7d['경매_예상가격'].mean()
        avg_priv_price = combined_7d['정가수의_예상가격'].mean() if '정가수의_예상가격' in combined_7d.columns else 0
        total_pred_volume = combined_7d[vol_col].sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            today_auc_p = int(today_row['경매_예상가격']) if pd.notna(today_row.get('경매_예상가격')) else 0
            today_vol_val = today_row[vol_col] if pd.notna(today_row.get(vol_col)) else 0.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">오늘({today_str[5:]}) 예상 경매가 ({selected_size})</div>
                <div class="metric-value">{today_auc_p:,}원</div>
                <div style="font-size:0.82rem; color:#495057; margin-top:4px;">{selected_citrus} 전체 예상 거래량: <b>{today_vol_val:,.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            tom_auc_p = int(tomorrow_row['경매_예상가격']) if pd.notna(tomorrow_row.get('경매_예상가격')) else 0
            price_diff = tom_auc_p - today_auc_p
            diff_rate = (price_diff / today_auc_p) * 100 if today_auc_p > 0 else 0
            delta_class = "metric-delta-pos" if price_diff >= 0 else "metric-delta-neg"
            delta_sign = "+" if price_diff >= 0 else ""
            tom_date_str = str(tomorrow_row.get('date', ''))
            tom_label = tom_date_str[5:] if len(tom_date_str) >= 5 else "내일"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">내일({tom_label}) 예상 경매가 ({selected_size})</div>
                <div class="metric-value">{tom_auc_p:,}원</div>
                <div class="{delta_class}" style="margin-top:4px;">전일 대비 {delta_sign}{int(price_diff):,}원 ({delta_sign}{diff_rate:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_priv_p_int = int(avg_priv_price) if (pd.notna(avg_priv_price) and avg_priv_price > 0) else 0
            avg_auc_p_int = int(avg_auc_price) if pd.notna(avg_auc_price) else 0
            prem_diff = avg_priv_p_int - avg_auc_p_int
            if avg_priv_p_int > 0:
                prem_sign = "+" if prem_diff >= 0 else ""
                prem_html = f"경매 대비 프리미엄: <b>{prem_sign}{prem_diff:,}원</b>"
                val_html = f"{avg_priv_p_int:,}원"
            else:
                prem_html = "정가수의 계약 이력 없음"
                val_html = "-"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">7일간 평균 예상 정가수의가 ({selected_size})</div>
                <div class="metric-value">{val_html}</div>
                <div style="font-size:0.82rem; color:#2b8a3e; margin-top:4px;">{prem_html}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">향후 7일간 {selected_citrus} 전체 예상 거래량</div>
                <div class="metric-value">{total_pred_volume:,.1f}</div>
                <div style="font-size:0.82rem; color:#495057; margin-top:4px;">일평균: <b>{total_pred_volume/7:,.1f}</b> (규격 무관 품목 전체)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. 메인 7일 복합 차트 (품목 전체 거래량 + 규격별 가격)
        fig_7d = utils.create_future_forecast_chart(combined_7d, selected_citrus, selected_size)
        st.plotly_chart(fig_7d, use_container_width=True, key="fig_7d_forecast")

        # 2. 일별 상세 예측 테이블 & 4대 품목 비교 뷰
        col_table, col_comp = st.columns([1.1, 0.9])
        
        with col_table:
            st.subheader(f"📋 일자별 상세 예측 데이터 ({today_str} ~ )")
            avail_cols = [c for c in ['date', 'weekday', '경매_예상가격', vol_col, '정가수의_예상가격', '가격차이(정가-경매)', '정가수의_프리미엄(%)', 'is_holiday'] if c in combined_7d.columns]
            table_df = combined_7d[avail_cols].copy()
            
            # 포맷팅 적용 (NaN 방어)
            if '경매_예상가격' in table_df.columns:
                table_df[f'경매 예상가({selected_size})'] = table_df['경매_예상가격'].apply(
                    lambda x: f"{int(x):,}원" if pd.notna(x) else "-"
                )
            if vol_col in table_df.columns:
                table_df[f'{selected_citrus} 전체거래량'] = table_df[vol_col].apply(
                    lambda x: f"{x:,.1f}" if pd.notna(x) and x > 0 else ("0.0" if pd.notna(x) and x == 0 else "시장 휴무")
                )
            if '정가수의_예상가격' in table_df.columns:
                table_df[f'정가수의 예상가({selected_size})'] = table_df['정가수의_예상가격'].apply(
                    lambda x: f"{int(x):,}원" if pd.notna(x) and x > 0 else "-"
                )
            if '가격차이(정가-경매)' in table_df.columns:
                table_df['가격차(정가-경매)'] = table_df['가격차이(정가-경매)'].apply(
                    lambda x: f"{int(x):+,}원" if pd.notna(x) else "-"
                )
            if '정가수의_프리미엄(%)' in table_df.columns:
                table_df['프리미엄(%)'] = table_df['정가수의_프리미엄(%)'].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
                )
            if 'is_holiday' in table_df.columns:
                table_df['도매시장 휴일'] = table_df['is_holiday'].apply(
                    lambda x: "🛑 휴일" if x else "✅ 정상영업"
                )

            drop_orig = ['경매_예상가격', vol_col, '정가수의_예상가격', '가격차이(정가-경매)', '정가수의_프리미엄(%)', 'is_holiday']
            table_display = table_df.drop(columns=[c for c in drop_orig if c in table_df.columns], errors='ignore')
            table_display = table_display.rename(columns={'date': '날짜', 'weekday': '요일'})

            st.dataframe(table_display, use_container_width=True, hide_index=True)
            
            csv_data = combined_7d.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="📥 7일 예측 결과 CSV 다운로드",
                data=csv_data,
                file_name=f"{selected_citrus}_{selected_size}_7일예측_{today_str}.csv",
                mime='text/csv',
                key="btn_download_forecast_csv"
            )

        with col_comp:
            st.subheader(f"📊 4대 감귤 품목 7일 가격 비교 ({selected_size})")
            if comp_4cats is not None and not comp_4cats.empty:
                fig_comp = utils.create_all_citrus_comparison_chart(comp_4cats, selected_size)
                st.plotly_chart(fig_comp, use_container_width=True, key="fig_comp_4cats")
            else:
                st.info("비교 데이터 준비 중입니다.")
    else:
        st.error("예측 결과를 산출할 수 없습니다. 데이터 조건을 확인해주세요.")

# ==============================================================================
# TAB 2: [경매 예측] 품목 전체 거래량 & 규격별 가격 분석 + 월별 품종 점유율(%)
# ==============================================================================
with tab_auction:
    st.header(f"🔨 [{selected_citrus} - {selected_size}] 거래량 연계형 가격 & 거래량 LightGBM 모델 분석")
    st.markdown(f"""
    > **수급 연계 파이프라인 구조 (노트북 설계 기반):**
    > 1. **1단계 (품목 전체 거래량 모델):** 제주 기상(기온·강수량), 요일, 시장 영업일 캘린더, 시계열 Lag를 반영하여 **{selected_citrus} 일별 총거래량**을 예측합니다. (규격 선택에 관계없이 품목 전체 거래량 기준 불변 유지)
    > 2. **2단계 (품목·규격별 가격 모델):** 1단계에서 예측된 거래량(`volume_feature`)을 핵심 공급 설명변수로 직접 투입하고, 선택된 **{selected_citrus} ({selected_size})**의 고유 시계열 가격 및 대체재 가격을 반영하여 일별 평균가격을 정밀 예측합니다.
    """)

    if auc_res is not None:
        metrics = auc_res['metrics']
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(f"{selected_citrus} 전체 거래량 R²", f"{metrics['vol_r2']:.4f}", help="품목 전체 거래량 설명력")
        with col_m2:
            st.metric("거래량 RMSE (오차)", f"{metrics['vol_rmse']:.2f}")
        with col_m3:
            st.metric(f"{selected_size} 가격 R²", f"{metrics['price_r2']:.4f}", help="규격별 가격 설명력")
        with col_m4:
            rmse_val = int(metrics['price_rmse']) if pd.notna(metrics.get('price_rmse')) else 0
            st.metric("가격 RMSE (오차)", f"{rmse_val:,}원")

        st.markdown("---")
        
        # 1. 실제 vs 예측 비교 플롯 (거래량은 품목 전체 기준, 가격은 규격 기준)
        st.subheader("📈 테스트 데이터: 실제값 vs LightGBM 예측값 비교")
        test_df_auc = auc_res['test_data']
        fig_dual = utils.create_dual_actual_pred_chart(test_df_auc, selected_citrus, selected_size, is_auction=True)
        st.plotly_chart(fig_dual, use_container_width=True, key="fig_auction_dual")

        st.markdown("---")

        # 2. 월별 품종별 점유율 (%) (결정요인 제거 및 전면 배치)
        st.subheader("📅 월별 전체 거래량 대비 품종별 점유율 (%)")
        fig_share = utils.create_monthly_category_share_chart(monthly_pct, selected_citrus)
        st.plotly_chart(fig_share, use_container_width=True, key="fig_auction_monthly_share")

        # 상세 점유율 표 토글
        with st.expander("📋 1월~12월 품종별 거래 비중(%) 데이터 테이블 보기"):
            share_display = monthly_pct.copy()
            share_display.index = [f"{m}월" for m in share_display.index]
            for col in share_display.columns:
                share_display[col] = share_display[col].apply(lambda x: f"{x:.1f}%")
            st.dataframe(share_display, use_container_width=True)

        st.markdown("---")

        # 3. 잔차 분석
        st.subheader("📉 가격 예측 잔차 (실제값 - 예측값) 분석")
        fig_res = utils.create_residual_chart(test_df_auc, '평균가격', 'predicted_price', f"{selected_citrus} {selected_size} 경매 가격 예측 잔차 분포")
        st.plotly_chart(fig_res, use_container_width=True, key="fig_auction_res")
    else:
        st.error("경매 모델 학습 결과가 없습니다.")

# ==============================================================================
# TAB 3: [정가수의 예측] 가격 중심 분석
# ==============================================================================
with tab_private:
    st.header(f"🤝 [{selected_citrus} - {selected_size}] 정가수의 매매 계약가격 LightGBM 모델 분석")
    st.markdown("""
    > **정가수의 매매 특성 및 모델 설계:**
    > 사용자 요구에 따라 **가격 예측에 집중**하여 분석을 제공합니다.
    > 정가수의 매매는 사전 협의된 단가로 안정적인 대량 거래가 이루어지므로, 경매 시장에 비해 가격 변동 폭이 적고 대체재 가격 및 직전 계약 단가의 연속성이 강하게 작용합니다.
    """)

    if priv_res is not None:
        p_metrics = priv_res['metrics']
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("정가수의 가격 R² (설명력)", f"{p_metrics['price_r2']:.4f}")
        with col_p2:
            rmse_priv = int(p_metrics['price_rmse']) if pd.notna(p_metrics.get('price_rmse')) else 0
            st.metric("정가수의 가격 RMSE", f"{rmse_priv:,}원")
        with col_p3:
            mae_priv = int(p_metrics['price_mae']) if pd.notna(p_metrics.get('price_mae')) else 0
            st.metric("정가수의 가격 MAE (평균 오차)", f"{mae_priv:,}원")

        st.markdown("---")

        test_df_priv = priv_res['test_data']
        st.subheader("📈 정가수의 평균가격 시계열: 실제 계약단가 vs LightGBM 예측단가")
        fig_priv_price = utils.create_dual_actual_pred_chart(test_df_priv, selected_citrus, selected_size, is_auction=False)
        st.plotly_chart(fig_priv_price, use_container_width=True, key="fig_private_price")

        st.markdown("---")
        # 잔차 분석 (결정요인 제거 및 전면 배치)
        st.subheader("📉 정가수의 가격 예측 잔차 (실제값 - 예측값) 분석")
        fig_priv_res = utils.create_residual_chart(
            test_df_priv, '평균가격', 'predicted_price', 
            f"{selected_citrus} {selected_size} 정가수의 가격 잔차 분포"
        )
        st.plotly_chart(fig_priv_res, use_container_width=True, key="fig_private_res")

    else:
        st.error("정가수의 모델 학습 결과가 없습니다.")

# ==============================================================================
# TAB 4: 감귤 시장 통계 & 상관관계
# ==============================================================================
with tab_market:
    st.header(f"📊 감귤 도매시장 거래 데이터 통계 (신규 데이터셋)")
    
    # 상단에 월별 품종별 점유율 차트 전면 배치
    st.subheader("📅 월별 전체 거래량 중 품종별 점유율(%) 심층 분석")
    fig_share_market = utils.create_monthly_category_share_chart(monthly_pct, selected_citrus)
    st.plotly_chart(fig_share_market, use_container_width=True, key="fig_market_monthly_share")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("📦 품목군별(gds_sclsf_grp) 연도별 데이터 건수 분포")
        auction_df_copy = auction_df.copy()
        auction_df_copy['year'] = auction_df_copy['trd_clcln_ymd'].dt.year
        year_cat_counts = auction_df_copy.groupby(['year', 'gds_sclsf_grp']).size().unstack(fill_value=0)
        st.dataframe(year_cat_counts.style.background_gradient(cmap='YlGnBu'), use_container_width=True)

    with col_s2:
        st.subheader("🏷️ 규격군별(unit_qty_grp) 거래 건수 분포")
        size_counts = auction_df['unit_qty_grp'].value_counts().reset_index()
        size_counts.columns = ['규격', '경매 건수']
        priv_size_counts = private_df['unit_qty_grp'].value_counts().reset_index()
        priv_size_counts.columns = ['규격', '정가수의 건수']
        merged_size_dist = pd.merge(size_counts, priv_size_counts, on='규격', how='outer').fillna(0)
        st.dataframe(merged_size_dist, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("☀️ 기상 요인(기온·강수량)과 거래량/가격의 관계")
    st.markdown("""
    - 거래량이 적은 날에는 도매시장 경락가의 분산이 넓게 나타나며, 일정 거래량 이상 출하될 때 가격이 특정 범위로 안정화(수렴)되는 패턴이 뚜렷합니다.
    - 강수 계속시간과 기온 변화는 출하 작업 및 소비지 수요에 직접적인 영향을 미쳐 1~2일 시차를 두고 거래량에 반영됩니다.
    """)
