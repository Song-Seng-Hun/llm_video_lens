<p align="center"><b>한국어</b> · <a href="README.en.md">English</a> · <a href="README.ja.md">日本語</a></p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img src="assets/banner-light.svg" width="640" alt="픽셀아트 배너. 돋보기가 컷이 하나 있는 필름 띠의 여덟 프레임을 차례로 읽으면, 프레임마다 자세 키포인트와 파랑에서 빨강으로 변하는 추적 점이 찍히고, Claude가 BRIEF에 한 줄씩 적으며, TRACK 패널에 컷에서 끊긴 점의 경로가 그려집니다.">
  </picture>
</p>

# LLM Image Lens

영상을 AI(LLM)가 읽을 수 있는 짧은 글과 확인용 그림으로 바꿔 주는 명령줄 도구입니다. 샷을 나누고, 샷마다 몸과 카메라가 어떻게 움직였는지 적으며, 모든 내용을 원본 프레임 번호로 되짚을 수 있게 남깁니다.

## 이게 뭔가요

사람이나 캐릭터가 나오는 영상을 AI 에이전트가 다룰 수 있게 정리해 주는 도구입니다. 영상 하나를 넣으면 샷(컷)마다 몸·팔·다리·카메라의 움직임을 짧은 글(`look.md`)로 적고, 확인용 그림을 함께 만듭니다. 결과 파일은 사람이 열어 읽어도 됩니다.

확실하지 않은 부분은 지어내지 않습니다. 점을 놓친 구간은 "모름"으로 적고, 도구 스스로 불안하다고 본 구간은 맨 위 "먼저 확인할 곳"에 모읍니다.

## 주요 기능

- **한 번에 요약 (`look`)**: 영상 하나를 샷별 움직임 요약 `look.md`와 그림 자료로 바꿉니다. 결과 글은 한국어입니다.
- **구간 질문 (`at`)**: "18~22초에 무슨 일이 있었지?"처럼 4초 이내 구간의 근거만 모아 줍니다.
- **샷 나누기와 목록 (`scan`, `brief`, `catalog`)**: 컷 경계를 찾고, 샷마다 원본 프레임 범위와 빠진 구간을 담은 목록을 만듭니다.
- **프레임 비교 (`compare`, `zoom`)**: 두 프레임의 차이를 그림으로 보거나, 짧은 구간을 촘촘한 프레임으로 봅니다.
- **직접 점 찍기 (`grid`, `mark`)**: 소품이나 자세 모델이 모르는 부분에 직접 점을 찍고 이름을 붙여 추적합니다.
- **닫힌 작업 공간**: 입력과 출력은 작업 공간 폴더 안에서만 읽고 씁니다. 덮어쓰기와 심볼릭 링크가 낀 경로는 거부합니다.

## 빠른 시작

1. Python 3.11 이상에서 의존성을 설치합니다. `look`은 자세 추정과 점 추적을 함께 쓰므로 두 묶음이 모두 필요합니다.

   ```sh
   pip install -r requirements-pose.txt -r requirements-tracking.txt
   ```

2. 모델 파일 세 개를 직접 준비합니다. 사람 검출(YOLOX HumanArt, ONNX), 전신 자세(RTMW 133 키포인트, ONNX), 점 추적(CoTracker 체크포인트)입니다. 저장소에 들어 있지 않고, 자동으로 내려받지도 않습니다.
3. 작업 공간 폴더(기본값은 저장소 안의 `workspace/`)에 영상을 복사하고, `workspace/models.json`에 모델 경로를 적습니다.

   ```json
   {"detector": "/models/yolox_humanart.onnx", "pose_model": "/models/rtmw133.onnx", "checkpoint": "/models/scaled_offline.pth"}
   ```

4. 저장소 폴더에서 실행합니다. 아래의 `W`는 작업 공간 폴더 경로입니다.

   ```sh
   python -m image_lens --workspace W look --video clip.mp4 --out look-clip
   ```

5. 결과 폴더의 `look.md`를 엽니다. 맨 위 "먼저 확인할 곳"부터 보세요.

## 사용 예

### 영상 하나를 요약해서 읽기

`look.md`는 추적한 구간마다 한 줄입니다. 한 줄에는 다음이 들어갑니다.

- 샷 구도: 전신, 무릎 위, 상반신, 얼굴 클로즈업
- 화면 전체(카메라)와 몸 중심의 움직임
- 팔·다리 같은 부위가 몸을 기준으로 한 움직임과 속도. 예: "속도 9→5→3→1 px/f"
- 점을 놓친 곳, 그림이 바뀐 곳, 같은 자세의 반복 같은 읽기 표시

함께 만들어지는 그림도 있습니다.

- `<window>-onion.jpg`: 첫 프레임과 마지막 프레임을 겹치고 움직임을 화살표로 그린 그림
- `frame-sheet-N.jpg`: 영상 전체에서 고르게 뽑은 프레임 모음
- `check-sheet-N.jpg`: "먼저 확인할 곳" 구간의 촘촘한 프레임

### 궁금한 구간만 묻기

요약 전체를 다시 읽지 말고 그 구간만 물어보세요. 구간은 4초 이내여야 합니다.

```bash
python -m image_lens --workspace W at --look W/look-clip --span 18-22 --out W/at-18-22
```

`at.md`에 그 구간에 걸친 줄과 표시, 열어 볼 그림 이름이 모이고, `at-frames-1.jpg`에 촘촘한 프레임 12장이 담깁니다.

### 자세 모델이 모르는 부분 추적하기

`grid`로 좌표 눈금을 그린 프레임을 보고, `mark`로 점을 찍어 추적합니다. 점은 최대 8개까지 찍을 수 있고, 이름은 ASCII로 붙입니다.

```bash
python -m image_lens --workspace W grid --video V --frame 572 --box 100,250,1000,1000 --out W/grid-f572.jpg
python -m image_lens --workspace W mark --video V --shots W/look/shots.json --frame 572 --end 599 --point hammer-stripe=430,680 --point eye=322,738 --checkpoint C --out W/mark-s19
```

## 현재 상태와 한계

- 검증한 환경은 macOS arm64의 Python 3.12입니다. Windows에서의 동작과 설치한 패키지(wheel) 실행은 검증되지 않았고, 점 추적에는 POSIX 환경이 필요합니다.
- 자세, 동작, 같은 사람인지 판별, 완급(easing)을 정확히 맞힌다고 주장하지 않습니다. "먼저 확인할 곳"에 없는 구간도 틀릴 수 있으니 프레임 시트를 함께 보세요.
- 모델 파일은 직접 준비해야 합니다. CoTracker 소스와 체크포인트에는 비상업적 이용 조건이 있고, 자세 모델 가중치의 라이선스는 확인되지 않았습니다.
- 1920×1080보다 큰 영상은 줄여서 처리합니다. 이때 보고서의 px 값은 줄인 영상 기준입니다.

## 더 보기

자세한 문서는 영어입니다.

- [전체 안내](docs/guide.md): 모든 명령과 옵션, `look.md` 읽는 법 전체, 결과 파일 목록, brief 형식, 작업(job)과 추적, 자세 시드
- [look 노트](docs/look-notes.md): `look`의 기준값과 그 근거가 된 확인 기록
- [한계](docs/limitations.md) · [검증](docs/verification.md) · [출처와 라이선스](docs/provenance.md) · [작업 계약](docs/jobs.md)
- 테스트 실행: `python -B -m unittest discover -s tests -v`
