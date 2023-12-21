import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import joblib
import numpy as np

st.set_page_config(page_title="Attrition Rate", page_icon="🚀")

st.title('🚀조기퇴사 가능성 예측 모델 TEST')

selected2 = option_menu(None, ["Input Data", "Upload File"], 
    icons=['gear', "list-task"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected2 == "Input Data":
    st.header("개별 직원 데이터 입력")
    st.markdown('개별 직원의 상세 데이터를 입력할 경우 조기퇴사 가능성을 예측해볼 수 있습니다')

    df = pd.read_csv('hr_data_cleaned.csv')
    loaded_model = joblib.load('xgboost_model.pkl')

    satisfaction_mean = df['satisfaction_level'].mean()
    hours_mean = df['average_monthly_hours'].mean()

    col1, col2 = st.columns(2)
    with col1:
            # 게이지바를 통한 입력값 설정
            satisfaction_level = st.slider("회사 만족도", float(0), float(1))
            last_evaluation = st.slider("최근 업무 평가 점수", float(0), float(1))
            average_monthly_hours = st.slider("한달 평균 근무 시간", float(90), float(320), step=1.0)
            time_spend_company = st.slider("근속 연수", float(0), float(10), step=1.0)

    with col2 :
            # 옵션 선택을 통한 입력값 설정
            work_accident = st.radio("시말서 작성 여부", ["작성했다", "작성한 적 없다"])
            promotion_last_5years = st.radio("지난 5년간 승진 여부", ["승진했다", "승진하지 않았다"])
            department = st.selectbox("부서", ['sales', 'technical', 'support', 'IT', 'HR', 'marketing', 'product_mng', 'accounting', 'RandD', 'management'])
            salary = st.selectbox("월급 규모", ['높음', '중간', '낮음'])

    # 입력값 변형 함수 정의
    def transform_input(work_accident, promotion_last_5years, department, salary):

        if work_accident == '실수한 적이 있다' :
            work_accident_transformed = 1
        else :
            work_accident_transformed = 0

        if promotion_last_5years == "승진했다" :
            promotion_last_5years_transformed = 1
        else :
            promotion_last_5years_transformed = 0

        department_dict = {'sales':1, 'IT':2, 'support':3, 'technical':4, 'HR':5, 'product_mng':6, 'RandD':7, 'marketing':8,
                        'management':9, 'accounting':10}
        department_transformed = department_dict[department]

        salary_dict = {'높음':3, '중간':2, '낮음':1}
        salary_transformed = salary_dict[salary]

        return work_accident_transformed, promotion_last_5years_transformed, department_transformed, salary_transformed

    work_accident_transformed, promotion_last_5years_transformed, department_transformed, salary_transformed = transform_input(work_accident, promotion_last_5years, department, salary)

    if st.button("Predict"):
        # 모델 예측
        input_array = [satisfaction_level, last_evaluation, average_monthly_hours, time_spend_company, work_accident_transformed, promotion_last_5years_transformed, department_transformed, salary_transformed]
        instance = np.array(input_array)
        data_array = instance.reshape(1, -1)
        data_array = np.array(data_array).astype(float)
        prediction = loaded_model.predict(data_array)

        # 예측 결과 출력
        if prediction[0] == 0 :
            message = '조기퇴사 가능성이 낮습니다'
        else :
            message = '조기퇴사 가능성이 높습니다'

        st.write(f"### {message}")

        # 퇴사 가능성 높을 경우 분석 리포트 출력

        if message == '조기퇴사 가능성이 높습니다' :

            #satisfaction_level
            if satisfaction_level < satisfaction_mean : 
                st.markdown('회사 만족도 : 전체 직원 평균보다 낮음')
            else : 
                st.markdown('회사 만족도 : 전체 직원 평균보다 같거나 높음')

            #last_evaluation
            if last_evaluation < 0.5 :
                same_eval = df[df['last_evaluation'] < 0.5]
                st.markdown('지난 성과 점수 : 낮음')
            elif 0.5 <= last_evaluation < 0.8 :
                same_eval = df[(df['last_evaluation'] >= 0.5) & (df['last_evaluation'] < 0.8)]
                st.markdown('지난 성과 점수 : 중간')
            elif 0.8 <= last_evaluation :
                same_eval = df[df['last_evaluation'] >= 0.8]
                st.markdown('지난 성과 점수 : 높음')
            
            slary_mean = same_eval['salary'].mean()
            if salary_transformed < slary_mean :
                '성과 대비 월급 : 비슷한 성과 점수 직원들 대비 월급 낮음'
            else :
                '성과 대비 월급 : 비슷한 성과 점수 직원들 대비 월급 높거나 같음'
            
            #average_monthly_hours
            if average_monthly_hours < hours_mean : 
                st.markdown('월 평균 근무시간 : 전체 직원 평균보다 낮음')
            else : 
                st.markdown('월 평균 근무시간 : 전체 직원 평균보다 같거나 높음')

            same_hours =  df[(df['average_monthly_hours'] >= average_monthly_hours - 10) & (df['last_evaluation'] < average_monthly_hours +10)]
            slary_mean2 = same_hours['salary'].mean()
            if salary_transformed < slary_mean2 :
                '근무시간 대비 월급 : 비슷한 성과 점수 직원들 대비 월급 낮음'
            else :
                '근무시간 대비 월급 : 비슷한 성과 점수 직원들 대비 월급 높거나 같음'

            #time_spend_company
            same_time = df[df['time_spend_company'] == time_spend_company]
            slary_mean3 = same_time['salary'].mean()
            if salary_transformed < slary_mean3 :
                '연차 대비 월급 : 동연차 직원들 대비 월급 낮음'
            else :
                '연차 대비 월급 : 동연차 직원들 대비 월급 높거나 같음'


            
elif selected2 == "Upload File":
    st.header('CSV 파일 업로드')
    st.markdown('직원들 데이터를 파일로 한꺼번에 업로드하면 조기퇴사 위험군 직원을 예측할 수 있습니다')

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    loaded_model = joblib.load('xgboost_model.pkl')
    loaded_encoder = joblib.load('ordinal_encoder.joblib')

    if uploaded_file is not None:
        loaded_df = pd.read_csv(uploaded_file, encoding='euc-kr')
        df_dropped = loaded_df.drop(columns=['ID', 'name'])

        df_encoded = loaded_encoder.transform(df_dropped)
        df_encoded = df_encoded.drop(columns=['number_project'])
        salary_map = {'low':1, 'medium':2, 'high':3}
        df_encoded['salary'] = df_encoded['salary'].map(salary_map)

        predictions = loaded_model.predict(df_encoded)
        loaded_df['predictions'] = predictions
        selected_rows_df = loaded_df[loaded_df['predictions'] == 1]
        final_df = selected_rows_df[['ID', 'name']]

        # 결과 표시
        st.markdown('조기퇴사 위험군 직원')
        st.dataframe(final_df)

