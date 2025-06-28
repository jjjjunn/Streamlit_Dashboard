# 📊 가상 데이터 리포지토리 기반 서비스 분석 및 추천 시스템

Streamlit을 활용하여 개발된 이 프로젝트는 가상의 마케팅 활동 데이터를 바탕으로,  
오프라인 및 온라인 캠페인 성과를 분석하고 다양한 예측 모델을 제공하는 **데이터 시각화 및 머신러닝 기반 웹 애플리케이션**입니다.

🔗 **배포 링크**: [Streamlit 웹 앱 바로가기](https://apprjgroup3-dtiyavdpz8ywuhdu6nhint.Streamlit.app/)

---

## 📅 프로젝트 개요

- **프로젝트 기간**: 2025년 3월 14일 ~ 3월 27일  
- **배포일**: 2025년 3월 27일  
- **가정된 서비스 기간**: 2023년 1월 1일 ~ 2024년 12월 31일 (2년간의 마케팅 활동 데이터 기반)

이 프로젝트는 마케팅/홍보팀이 2년간 운영한 가상의 오프라인/온라인 캠페인을 기반으로 하여,  
성과 지표를 CSV로 정리하고, 다양한 조건에 따라 사용자 행동을 예측하는 **데이터 기반 마케팅 의사결정 도구**입니다.

---

## 🖥️ 웹 애플리케이션 구조

Streamlit의 `multipage` 기능을 활용하여 전체 애플리케이션을 5개의 주요 페이지로 구성하였습니다.

| 페이지명        | 설명 |
|-----------------|------|
| **Home**        | 프로젝트 소개 및 페이지 안내 역할을 수행하는 메인 화면 |
| **Summary**     | 오프라인 + 온라인 데이터를 종합적으로 보여주는 대시보드 (필터 없음) |
| **Offline**     | 캠페인 날짜, 요일, 지역 필터링을 통한 오프라인 활동 데이터 시각화 |
| **Online**      | 온라인 유입 경로, 디바이스, 날짜 필터링을 통한 온라인 캠페인 데이터 시각화 |
| **Pred Model**  | 다양한 예측/추천 모델을 통해 사용자 행동 분석 및 마케팅 전략 추천 |

---

## 📁 데이터 구성

| 구분       | 설명 |
|------------|------|
| **오프라인 데이터** | 지역별 캠페인 참여율, 참여 인구 통계, 방문 채널, 전환율 등 포함 |
| **온라인 데이터** | 유입 채널, 디바이스 종류, 시간대, 클릭/전환 데이터 등 포함 |
| **사용자 프로필** | 가상의 1,000명 사용자: 나이, 성별, 결혼 여부, 지역, 캠페인 참여 기록 등 포함 |

모든 데이터는 프로젝트 목적에 맞게 **파이썬으로 생성된 가상 데이터(fake data)**를 사용하여 구현되었습니다.

[가상 데이터 코드 보러가기](https://github.com/jjjjunn/fake_data_for_streamlit_prj)

---

## 🤖 머신러닝 모델 상세

> 모든 모델은 **scikit-learn** 라이브러리를 기반으로 구축되었으며,  
> **Random Forest, Logistic Regression, Gradient Boosting** 등의 알고리즘을 사용했습니다.

### 1. ✅ 신규 서비스 구독 예측 모델 (Classification)
- 입력값: 나이, 성별, 결혼 여부
- 출력값: 사용자가 새로운 서비스를 구독할 가능성
- 사용 알고리즘: RandomForestClassifier

### 2. 🎯 맞춤형 캠페인 추천 시스템
- 입력값: 나이, 성별, 도시, 결혼 여부
- 출력값: 가장 높은 구독률을 기록한 캠페인 추천
- 사용 알고리즘: VotingClassifier (Ensemble)

### 3. 🌐 최적의 온라인 채널 추천
- 입력값: 나이, 성별, 결혼 여부
- 출력값: 가입률이 높은 상위 3개 온라인 유입 채널
- 구현 방식: 채널별 전환율 계산 → 조건부 필터링 기반 추천

### 4. 📲 디바이스 기반 전환율 예측
- 입력값: 디바이스 종류, 유입 채널
- 출력값: 예상 전환 확률
- 사용 알고리즘: RandomForestClassifier

### 5. 📈 월별 방문자 예측 (12개월 Forecasting)
- 입력값: 지역
- 출력값: 향후 12개월간 월별 예상 방문자 수
- 사용 알고리즘: RandomForestRegressor

---

## 📊 주요 기능

- 📂 캠페인 성과 데이터 CSV 로드
- 🗺️ Folium 기반 지도 시각화 (오프라인 캠페인 분포)
- 📉 Plotly 및 Seaborn을 이용한 대화형 시각화
- 🧠 사용자 조건에 따른 실시간 예측 결과 출력
- 🧩 각 모델 결과를 기반으로 한 마케팅 전략 제안

---

## 🛠️ 사용 기술 스택

| 범주       | 기술 |
|------------|------|
| **Frontend** | `Streamlit` |
| **Backend / Logic** | `Python` |
| **데이터 처리** | `pandas`, `numpy`, `datetime`, `time` |
| **시각화 도구** | `seaborn`, `matplotlib`, `plotly`, `folium` |
| **머신러닝 라이브러리** | `scikit-learn`, `RandomForest`, `GridSearchCV`, `VotingClassifier`, `Pipeline`, `OneHotEncoder`, `StandardScaler` 등 |
| **기타** | `StratifiedKFold`, `Cross_val_score`, `train_test_split` |

---

## 📈 기대 효과

- 실제 마케팅 환경에서 사용 가능한 분석/추천 구조를 이해할 수 있음  
- 비즈니스 현장에서 데이터 기반 의사결정의 가능성을 체험  
- 다양한 모델 및 시각화 기법을 통합하여 **End-to-End ML 프로젝트 구조** 이해  
- 비개발자도 활용 가능한 **인터랙티브한 대시보드 경험** 제공

---

## 📝 개발 기록 (블로그 제작기)

| 주제 | 링크 |
|------|------|
|가상데이터 생성 | [바로가기](https://puppy-foot-it.tistory.com/722) |
|데이터 전처리, 시각화 | [바로가기](https://puppy-foot-it.tistory.com/734) |
|데이터 시각화: 캠페인별 방문&참여자 | [바로가기](https://puppy-foot-it.tistory.com/735) |
|데이터 시각화: 연령대별 분석 | [바로가기](https://puppy-foot-it.tistory.com/736) |
|머신러닝 | [바로가기](https://puppy-foot-it.tistory.com/737) |
|DB연동 | [바로가기](https://puppy-foot-it.tistory.com/738) |
|Streamlit 구현하기 | [바로가기](https://puppy-foot-it.tistory.com/739) |
|페이지 구현 | [바로가기](https://puppy-foot-it.tistory.com/740) |
|멀티페이지 | [바로가기](https://puppy-foot-it.tistory.com/741) |
|Streamlit 배포하기 | [바로가기](https://puppy-foot-it.tistory.com/742) |
|ML 구현하기 | [바로가기](https://puppy-foot-it.tistory.com/743) |
|전체 보완 및 재배포 | [바로가기](https://puppy-foot-it.tistory.com/722) |

---

## 💡 개선 및 확장 아이디어

- 실제 Google Analytics 또는 SNS API 연동을 통한 실시간 데이터 처리  
- 예측 정확도를 높이기 위한 Hyperparameter Tuning 및 딥러닝 모델 도입  
- Flask 또는 FastAPI로 백엔드 API화 및 다양한 프론트엔드 프레임워크와의 통합  
- 사용자 피드백 기반 추천 시스템 고도화

---

## 📌 참고

- 본 프로젝트의 모든 데이터는 **가상의 시뮬레이션 데이터**입니다. 실제 기업 또는 서비스와는 무관합니다.
- 모델 성능 지표 및 정확도는 Streamlit 앱 내에서 확인 가능합니다.

---

