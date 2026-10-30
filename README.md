<p align="center"><b>한국어</b> · <a href="README.en.md">English</a> · <a href="README.ja.md">日本語</a></p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img src="assets/banner-light.svg" width="640" alt="픽셀아트 배너. 돋보기가 컷이 하나 있는 필름 띠의 여덟 프레임을 차례로 읽으면, 프레임마다 자세 키포인트와 파랑에서 빨강으로 변하는 추적 점이 찍히고, Claude가 BRIEF에 한 줄씩 적으며, TRACK 패널에 컷에서 끊긴 점의 경로가 그려집니다.">
  </picture>
</p>

# LLM Image Lens

LLM이 영상을 살펴볼 수 있게 해 주는, 원본에 묶인(source-bound) 영상 검사 CLI입니다. 샷 경계를 찾고, 원본 프레임 몇 장을 모은 작은 묶음(contact)을 요청하고, 보수적인 2D 표면 움직임 보고서를 읽을 수 있습니다. 이번 릴리스에서 제한적으로 나아진 점은 간결한 샷 요약(brief)과 목록(catalog)입니다. 원본 프레임/PTS 범위, 원본 해시, 빠진 구간, 원시 산출물의 정확한 위치(locator)를 담습니다. 자세(pose), 동작, 동일인 판별, 이징(easing)의 정확도가 나아졌다고 주장하지는 않습니다.

핵심 작업 흐름은 이미 골라 둔 엔진을 그대로 씁니다. 진단용 프레임 비교, 원본 전체 기준(canonical full-source)의 하드 컷 맥락, 표면 점 타이밍, 명시적인 인물/특징 보고서, 원본 색 영역, 그리고 선택 사항인 단색 배경 전경 검사입니다. 기본 이미지 출력은 **0장**입니다. 직접 요청한 원본 프레임 묶음은 모두 합쳐 8프레임까지이고, 비교는 원본 2장과 그 차이(residual)를 보여 주는 패널 1장을 씁니다.

> `docs/` 안의 자세한 문서는 영어입니다.

## 설치 없이 실행

표준 라이브러리만 쓰는 명령은 Python 3.11 이상이 필요합니다. 디코딩·렌더링 명령에는 `requirements.txt`가 더 필요하며, 의존성·체크포인트·미디어는 들어 있지 않습니다. 추출(extraction)을 검증한 환경은 macOS arm64의 Python 3.12입니다. Windows에서의 핵심 동작과 설치한 wheel 실행은 검증되지 않았고, 실험적 추적(tracking)에는 POSIX 부모 감시(watchdog) 프로세스가 필요합니다.

저장소 안에서:

```sh
python -m image_lens --help
python -m image_lens --workspace /path/to/workspace scan --video input.mp4 --out shots.json
python -m image_lens --workspace /path/to/workspace brief --shots shots.json --out brief.json
python -m image_lens --workspace /path/to/workspace catalog --shots shots.json --offset 0 --limit 8 --format markdown --out brief.md
python -m image_lens --workspace /path/to/workspace compare --shots shots.json --frames 5,6 --evidence-dir comparison --out comparison.json
```

다른 작업 폴더에서 실행할 때는(격리된 Python 포함) 얇은 직접 실행기의 절대 경로를 씁니다.

```sh
python -I /path/to/llm_image_lens/image_lens_cli.py --workspace /path/to/workspace brief --shots shots.json --out brief.json
```

선택적으로 설치하면 `image-lens` 명령이 생깁니다. 추가 묶음(extras)은 `.[core]`, `.[tracking]`, `.[pose]`입니다. 표준 라이브러리만으로 설치하면 반드시 필요한 런타임 의존성이 없습니다. 버전을 고정한 PySceneDetect 소스는 이미 저장소에 들어 있습니다(vendored). 지원하는 진입점은 공개 CLI이고, 엔진마다 있는 main은 내부 어댑터입니다.

데이터·출력의 상대 경로는 모두 선택한 작업 공간(workspace)을 기준으로 하며, 기본값은 `<repository>/workspace`입니다. 원본 입력은 그곳에 직접 복사해 두어야 합니다. 입력과 출력은 이 경계를 벗어날 수 없고, 심볼릭 링크가 낀 경로와 덮어쓰기는 거부됩니다. 경계 밖에서 받을 수 있는 입력은 선택 사항인 체크포인트 하나뿐입니다. 작업 공간, 영상, 모델, 생성된 이미지와 결과는 Git이 무시합니다.

## 요약(brief) 읽기

`brief`와 `catalog`는 같은 제한된 스키마 `image-lens.brief/1`을 출력합니다. 기본으로 샷 8행을 돌려주고(최대 16행), `next_offset`과 생략된 개수로 페이지 나누기가 드러납니다. 원본 PTS 배열은 원시 스캔 위치(locator)에 그대로 둡니다. 검증된 요약을 내기 전에 원본 바이트와 현재 검출기 헬퍼·소스 고정값(pin)이 일치해야 합니다. 일부만 스캔한 원본은 일부라고 명시하고, 불완전한 디코드 패킷은 거부합니다. 해시 검사는 바이트와 출처를 묶어 줄 뿐, 검출된 컷이 맞다는 것까지 보장하지는 않습니다.

선택 사항인 `--report report.json`은 지원되는 캐시된 움직임 보고서 중에서, 고정된 원본 트랙과 바인딩으로부터 정확한 표준(canonical) 스캔 기준으로 다시 계산할 수 있는 것만 더합니다. 없거나, 오래되었거나, 지원되지 않거나, 컷을 넘나드는 보고서는 `not_admitted`로 표시되고 그 구간은 알 수 없음으로 남습니다. 보고서 참조는 최대 8개입니다. brief는 LLM/API 호출, 네트워크, 이미지 생성, 원시 좌표 덤프, 모델 불러오기를 하지 않습니다. 보고서 안의 라벨은 믿을 수 없는 인용 데이터로 다룹니다.

모든 샷에는 원본을 새로 관찰하고 다시 추적해야 할 의무가 남습니다. 프레임이 하나뿐인 샷은 살펴보기만 할 수 있습니다. 컷이나 디졸브를 놓칠 수 있으며, 컷이 검출되지 않았다고 해서 연속성이나 동일성이 증명되지는 않습니다. 클립 전체를 오프라인으로 추론한 좌표를 잘라 와서 샷 단위 추적이라고 부르지 마세요.

## 새 작업(job)과 선택적 추적

[작업 계약](docs/jobs.md)을 보세요. `prepare`는 새 질의(query) 패킷과 표준 전체 원본 스캔을 받아 샷 단위 작업 하나를 검증하고, 바꿀 수 없는 새 고정본(freeze)을 만듭니다. 기본값으로는 검증만 허용합니다. `validate`는 원본 바이트, 질의, vendor·헬퍼 고정값, 디코드/PTS와 컷을 **Torch를 불러오지 않고**, 체크포인트 없이, 트랙을 만들지 않고 검사합니다. 핵심 디코드 의존성이 필요합니다.

```sh
python -m image_lens --workspace /path/to/workspace prepare --queries queries.json --shots shots.json --job shot0 --out job.freeze.json
python -m image_lens --workspace /path/to/workspace validate --queries queries.json --frozen job.freeze.json --job shot0 --out unused-tracks.json
```

실험적 추론에는 `--allow-tracking`으로 따로 준비한 고정본, 명시적인 로컬 체크포인트, 고정된 실행 시간 예산이 필요합니다. 기회는 한 번뿐이며 모델을 불러오기 직전에 소모되고, 실패해도 자동으로 다시 시도하지 않습니다. 예산 시간이 지나면 CLI 부모 프로세스가 자기 자식 프로세스 그룹을 종료하고 회수합니다. 내려받기는 일어나지 않습니다. CoTracker 소스와 체크포인트에는 원저작자의 비상업적 이용 조건이 붙어 있습니다. 이번 추출을 검증하면서 신경망 추론은 실행하지 않았습니다.

```sh
python -m image_lens --workspace /path/to/workspace prepare --queries queries.json --shots shots.json --job shot0 --allow-tracking --max-seconds 120 --out tracking.freeze.json
python -m image_lens --workspace /path/to/workspace track --queries queries.json --frozen tracking.freeze.json --job shot0 --max-seconds 120 --checkpoint /path/to/scaled_offline.pth --out tracks.json
```

관찰된 표면 트랙은 실제 엔진으로 골라서 분석할 수 있습니다.

```sh
python -m image_lens --workspace /path/to/workspace analyze --tracks tracks.json --shots shots.json --point surface0 --start 0 --end 30 --out timing.json
python -m image_lens analyze --help
```

LLM이 읽을 보고서라면 `--report-bindings bindings.json --format markdown --trajectory-dir traj`를 붙이세요. 그러면 Markdown 앞부분에 굵은 획(stroke)이 나옵니다. 중앙값 필터를 거친 경로를 모서리와 멈춤에서 나눈 것으로, 가장 큰 단위의 절반 이상이면 주요(major) 획입니다. `traj/trajectory.png`에는 프레임 전체 기준의 경로(이른 쪽 파랑, 늦은 쪽 빨강), 획·타임라인 띠, 자르지 않은 프레임 최대 4장이 나옵니다. 같은 실행에서 외형 변화 후보도 나열합니다. 연속 프레임의 회색조 차이가 튀는 곳으로, `compare --frames a,b`로 확인할 대상입니다. 주요 획이 후보와 겹치면 표시해 둡니다. 잔상(echo)이나 다시 그리기 때문에 점이 튈 수 있기 때문입니다. 획은 `brief`에서 다시 계산되지만, 그림과 후보는 다시 계산되지 않습니다.

## 자세(pose)에서 전신 시드 만들기 (선택)

`pose`는 표준 스캔에 로컬 전신 자세 모델을 돌립니다. YOLOX HumanArt 사람 박스, RTMW 133개 키포인트(rtmlib, ONNX Runtime CPU), 검출된 컷마다 초기화되는 ByteTrack ID를 씁니다. ONNX 파일 두 개는 사용자가 직접 준비하며 크기와 SHA256으로 고정합니다. 아무것도 내려받거나 포함하지 않으며, 가중치 라이선스는 확인되지 않았습니다. 같은 환경에 `requirements-pose.txt`를 설치하세요. 패킷은 자세 계층 자체를 믿기 어려운 프레임도 표시합니다. 사람 박스가 겹치는 경우(잔상 복사본, 잔영)와 한 트랙 안에서 몸 골격이 튀는 경우입니다.

`seed`는 그 패킷의 한 프레임을 새 관찰 작업들로 바꿉니다. 12점 표면 계약에 맞도록 그룹마다 하나씩 만듭니다: `<job>-head`(코, 눈, 앞·위·옆 머리카락), `<job>-arms`, `<job>-legs`. 머리카락 점은 눈으로 세운 머리 좌표계(눈 사이 거리와, 코에서 멀어지는 위쪽 방향) 위의 기하 위치라서, 검출된 머리카락이 아니라 표면 점 후보입니다. 화면 밖 관절은 가장자리에 붙은 채로 돌아오기 때문에, 프레임 가장자리 근처 관절은 뺍니다. 시드마다 표시를 해 둔 원본 크롭이 고정된 시각 참조로 붙습니다. 추적하기 전에 꼭 보세요. `bind`는 추적이 끝난 작업 하나의 보고서 바인딩을 씁니다.

```sh
python -m image_lens --workspace W pose --shots shots.json --detector /models/yolox_humanart.onnx --pose-model /models/rtmw133.onnx --out pose.json
python -m image_lens --workspace W seed --pose pose.json --frame 215 --job s7 --out s7.obs.json --evidence-dir s7-crops
# s7-head, s7-arms, s7-legs 를 각각 prepare/track 한 다음:
python -m image_lens --workspace W bind --queries s7.obs.json --tracks s7-head.tracks.json --job s7-head --out s7-head.bindings.json
python -m image_lens --workspace W analyze --tracks s7-head.tracks.json --shots shots.json --report-bindings s7-head.bindings.json --start 215 --end 307 --format markdown --trajectory-dir s7-head-traj --pose pose.json --out s7-head.md
```

`--pose`를 주면 궤적 그림에 자세가 불안한 프레임이 표시되고(타임라인의 주황색), 그 근처의 변화 후보에 태그가 붙습니다. 경고는 묶여서, 후보나 자세 불안 구간과 겹치는 큰 획을 한 줄에 모두 적습니다. 점이 둘보다 많으면 주요 획에만 라벨을 달고, Markdown 앞부분에 점마다 한 줄 요약과 "함께 움직임" 줄(방향이 같고 시간이 겹치며 크기가 비슷한 여러 점의 주요 획)이 나옵니다. 점별 속도 구간, 단계, 길이 비교는 JSON에 남습니다. 눈동자는 모델링하지 않으며, 눈 점은 그려진 눈 영역을 따라갑니다.

분석 도움말에는 검토 대상 고르기, 명시적인 특징 바인딩, 원본 색 영역 보고서, 선택적인 전경 검사도 있습니다. 해석하기 전에 [한계](docs/limitations.md)를 읽어 주세요.

## 검증과 출처

```sh
python -B -m unittest discover -s tests -v
# 선택: 핵심 의존성 스모크 테스트 (새 출력 폴더로)
python -I -B tests/smoke_runtime.py --workspace workspace/NEW_SMOKE_DIRECTORY
```

[검증](docs/verification.md)은 테스트한 CLI·안전장치와, 실행하지 않은 추론 및 플랫폼 범위를 나눠 적습니다. [출처와 라이선스](docs/provenance.md)에는 원본 해시, 상위 저장소 고정값, 제한적 로딩 패치, 직접 작성한 코드의 라이선스 경계가 있습니다. 과거 실험 미디어, 픽스처, 기록, 결과, 버린 자동 분류기는 선택된 실행 경로가 아닙니다.
