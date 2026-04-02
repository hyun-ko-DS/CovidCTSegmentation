# CovidCTSegmentation

이 저장소는 Kaggle의 `COVID-19 CT Images Segmentation` 대회에서 `2위`를 기록한 솔루션 GitHub 입니다.  

- 대회 링크: https://www.kaggle.com/competitions/covid-segmentation
- 대회 리더보드: https://www.kaggle.com/competitions/covid-segmentation/leaderboard

## Competition Info
이 대회는 COVID-19 환자의 CT axial slice에서 다음 두 가지 병변을 세그멘테이션하는 문제입니다.

- `GGO` (Ground-glass opacity)
- `CS` (Consolidation)

평가는 pixel-wise `F1-score`를 두 클래스와 테스트 이미지들에 대해 평균하여 산출합니다. (대회 핵심 지표가 `F1`임)

## Train Details
학습은 RunPod에서 `NVIDIA A6000 (48GB VRAM)`에서 `PyTorch 2.4`로 진행했습니다.

## How to Use
아래 순서대로 실행하면 됩니다.

1. `conda` 가상환경 python `3.11`버전으로 생성
2. `requirements.txt` 설치 (`pytorch 2.4`버전 기준)
3. `kaggle.json` 파일 업로드 (kaggle 홈페이지에서 발급 후 업로드)
4. `config.json` 파일 업로드 (Contact: hyunko954@gmail.com)
5. `python main.py` 로 경로 등록
6. `python loader.py` 로 데이터셋 다운로드
7. `python train.py` 로 학습 (직접 학습 또는 best.pt 제공은 Contact: hyunko954@gmail.com)
8. `python inference.py` 로 validation 및 test 추론

## Results
- Validation F1: 0.7561 (GG: 0.8037, CS: 0.7084)
- TEST F1: 0.7362