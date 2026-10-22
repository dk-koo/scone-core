# 동작 데이터 분석기 v3.0

SCONE/OpenSim 계열 동작 데이터를 불러와 통계, 보행 phase, 좌우 대칭, 근활성, GRF/COP/COM, 리포트와 시각화를 생성하는 Tkinter GUI 애플리케이션입니다.

## 주요 기능

- CSV, STO, TXT, XLSX, XLS 파일 로드
- 파일 인코딩과 헤더 행 자동 감지
- 데이터 품질 검증과 최소 시간 길이 확인
- 전체 통계, 공통 feature, SCONE state 기반 phase 통계 계산
- 보행 주기 0-100% 정규화 그래프
- 좌우 symmetry feature 계산
- 근활성 요약, CCI, GRF/COP/COM, overview 시각화
- CSV/XLSX 결과 저장, PNG/SVG/HTML 그래프 저장
- 자동 분석 후보 추천과 전체 자동 리포트 생성

## 프로젝트 구조

```text
SCONE_analysis_v1/
├── main.py                    # GUI 실행 진입점
├── requirements.txt           # 실행과 테스트에 필요한 패키지
├── config/                    # 상수, phase 이름, 언어 메시지
├── src/                       # GUI, 분석 흐름, 분석 엔진, 저장 로직
├── utils/                     # 데이터 검증, 수학, 파일명, 단위 변환 유틸
├── visualization/             # 그래프와 대시보드 생성 로직
├── tests/                     # 자동 테스트
└── output/                    # 실행 결과물, Git 추적 제외
```

## 설치

Python 3.8 이상을 권장합니다.

```bash
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

## 테스트

```bash
python -m pytest tests -q
```

현재 테스트는 데이터 로딩, 분석 엔진, 자동 분석 후보, 결과 저장, 주요 시각화 로직을 포함합니다.

## 분석 phase 이름

SCONE state 값은 프로젝트 전체에서 아래 이름으로 통일합니다.

| state | phase |
| --- | --- |
| 0 | EarlyStance |
| 1 | LateStance |
| 2 | LiftOff |
| 3 | Swing |
| 4 | Landing |

복합 phase로 `stance`는 state 0, 1, 2를 포함하고, `swing`은 state 3, 4를 포함합니다.

## Git 관리 기준

`.gitignore`는 가상환경, 캐시, 테스트 임시 폴더, 생성된 분석 결과물(`output/`)을 Git에서 제외합니다. 소스 코드, 설정, 문서, 테스트만 GitHub에 올리는 것을 기본 원칙으로 합니다.

## 사용 흐름

1. `파일 열기`로 데이터 파일을 선택합니다.
2. 자동 후보 또는 왼쪽 변수 목록에서 분석할 변수를 선택합니다.
3. `분석 실행`, `자동 분석`, `Feature 분석`, `추천 실행` 중 목적에 맞는 버튼을 사용합니다.
4. 결과 패널에서 요약을 확인합니다.
5. CSV/XLSX 결과, 그래프, HTML 대시보드를 필요한 위치에 저장합니다.

## 주의할 점

- `time` 컬럼이 있으면 오름차순 숫자 시간으로 검증합니다.
- joint angle 변수는 내부 radian 값을 결과와 그래프에서 degree로 변환해 표시합니다.
- 부호 방향은 모델 좌표계 정의를 따르며, 앱이 임의로 해부학적 의미로 바꾸지 않습니다.
- 원본 연구 데이터와 생성 결과물은 코드 저장소와 분리해 관리하는 것을 권장합니다.
