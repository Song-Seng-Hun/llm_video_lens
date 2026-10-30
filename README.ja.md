<p align="center"><a href="README.md">한국어</a> · <a href="README.en.md">English</a> · <b>日本語</b></p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img src="assets/banner-light.svg" width="640" alt="ピクセルアートのバナー。虫めがねがカットをひとつ含むフィルムの 8 フレームを順に読むと、各フレームに姿勢のキーポイントと青から赤へ変わる追跡点が付き、Claude が BRIEF に 1 行ずつ書き、TRACK パネルにカットで途切れた点の経路が描かれます。">
  </picture>
</p>

# LLM Image Lens

LLM が動画を確かめられるようにする、元データに結び付いた（source-bound）動画検査 CLI です。ショット境界を見つけ、元フレーム数枚の小さなまとめ（contact）を要求し、控えめな 2D 表面の動きレポートを読むことができます。今回のリリースで限定的に改善したのは、簡潔なショット要約（brief）と一覧（catalog）です。元のフレーム／PTS 範囲、元データのハッシュ、欠けている区間、生の成果物の正確な位置（locator）を含みます。姿勢（pose）、動作、同一人物の判定、イージング（easing）の精度が上がったとは主張しません。

中心となる作業の流れは、選んである既存のエンジンをそのまま使います。診断用のフレーム比較、元データ全体基準（canonical full-source）のハードカットの文脈、表面点のタイミング、明示的な人物／特徴レポート、元の色の領域、そして任意の単色背景の前景検査です。既定の画像出力は **0 枚** です。明示的に要求した元フレームのまとめは合計 8 フレームまでで、比較には元画像 2 枚と差分（residual）を示すパネル 1 枚を使います。

> `docs/` 内の詳しい文書は英語です。

## インストールせずに実行

標準ライブラリだけを使うコマンドには Python 3.11 以上が必要です。デコード・描画のコマンドにはさらに `requirements.txt` が必要で、依存パッケージ・チェックポイント・メディアは含まれていません。抽出（extraction）を検証した環境は macOS arm64 の Python 3.12 です。Windows での主要動作とインストールした wheel の実行は未検証で、実験的なトラッキングには POSIX の親プロセスによる監視（watchdog）が必要です。

リポジトリ内から：

```sh
python -m image_lens --help
python -m image_lens --workspace /path/to/workspace scan --video input.mp4 --out shots.json
python -m image_lens --workspace /path/to/workspace brief --shots shots.json --out brief.json
python -m image_lens --workspace /path/to/workspace catalog --shots shots.json --offset 0 --limit 8 --format markdown --out brief.md
python -m image_lens --workspace /path/to/workspace compare --shots shots.json --frames 5,6 --evidence-dir comparison --out comparison.json
```

別の作業フォルダから実行するときは（隔離された Python を含む）、薄い直接ランチャーの絶対パスを使います。

```sh
python -I /path/to/llm_image_lens/image_lens_cli.py --workspace /path/to/workspace brief --shots shots.json --out brief.json
```

任意でインストールすると `image-lens` コマンドが使えます。追加パッケージ（extras）は `.[core]`、`.[tracking]`、`.[pose]` です。標準ライブラリだけでインストールした場合、必須の実行時依存はありません。バージョンを固定した PySceneDetect のソースはすでに同梱（vendored）されています。サポートする入口は公開 CLI で、各エンジンの main は内部アダプターです。

データ・出力の相対パスはすべて選んだワークスペースが基準で、既定は `<repository>/workspace` です。元の入力はそこへ明示的にコピーしておく必要があります。入力と出力はこの境界の外に出られず、シンボリックリンクを含むパスと上書きは拒否されます。境界の外から受け取れる入力は、任意のチェックポイントひとつだけです。ワークスペース、動画、モデル、生成画像と結果は Git の管理対象外です。

## 要約（brief）を読む

`brief` と `catalog` は同じ制限付きスキーマ `image-lens.brief/1` を出力します。既定ではショットを 8 行返し（最大 16 行）、`next_offset` と省略数でページ分けが明示されます。元の PTS 配列は生のスキャンの位置（locator）にそのまま残します。検証済みの要約を出す前に、元データのバイト列と現在の検出器ヘルパー・ソースの固定値（pin）が一致している必要があります。一部だけスキャンした元データは一部であると明示し、不完全なデコードパケットは拒否します。ハッシュの検査はバイト列と出所を結び付けるだけで、検出したカットが正しいことまでは保証しません。

任意の `--report report.json` は、サポートされたキャッシュ済みの動きレポートのうち、固定された元トラックとバインディングから正確な標準（canonical）スキャン基準で再計算できるものだけを加えます。存在しない、古い、未対応、カットをまたぐレポートは `not_admitted` と表示され、その区間は不明のまま残ります。レポートの参照は最大 8 件です。brief は LLM／API の呼び出し、ネットワーク、画像生成、生の座標のダンプ、モデルの読み込みを行いません。レポート内のラベルは信頼できない引用データとして扱います。

すべてのショットには、元データを新たに観察し直し、追跡し直す義務が残ります。1 フレームしかないショットは確認だけが可能です。カットやディゾルブを見逃すことがあり、カットが検出されなかったからといって連続性や同一性が証明されるわけではありません。クリップ全体をオフラインで推論した座標を切り出して、ショット単位の追跡と呼ばないでください。

## 新しいジョブと任意のトラッキング

[ジョブの契約](docs/jobs.md) を参照してください。`prepare` は新しいクエリパケットと標準の全体スキャンを受け取り、ショット単位のジョブをひとつ検証して、変更できない新しい固定版（freeze）を作ります。既定では検証のみを許可します。`validate` は元データのバイト列、クエリ、vendor・ヘルパーの固定値、デコード／PTS とカットを、**Torch を読み込まず**、チェックポイントなしで、トラックを作らずに検査します。主要なデコード依存パッケージが必要です。

```sh
python -m image_lens --workspace /path/to/workspace prepare --queries queries.json --shots shots.json --job shot0 --out job.freeze.json
python -m image_lens --workspace /path/to/workspace validate --queries queries.json --frozen job.freeze.json --job shot0 --out unused-tracks.json
```

実験的な推論には、`--allow-tracking` で別に用意した固定版、明示的なローカルのチェックポイント、固定された実行時間の予算が必要です。機会は 1 回だけで、モデルを読み込む直前に消費され、失敗しても自動では再試行しません。予算時間を過ぎると、CLI の親プロセスが自分の子プロセスグループを終了させて回収します。ダウンロードは発生しません。CoTracker のソースとチェックポイントには、原作者による非商用の利用条件があります。今回の抽出の検証では、ニューラルネットの推論は実行していません。

```sh
python -m image_lens --workspace /path/to/workspace prepare --queries queries.json --shots shots.json --job shot0 --allow-tracking --max-seconds 120 --out tracking.freeze.json
python -m image_lens --workspace /path/to/workspace track --queries queries.json --frozen tracking.freeze.json --job shot0 --max-seconds 120 --checkpoint /path/to/scaled_offline.pth --out tracks.json
```

観察した表面のトラックは、実際のエンジンで選んで分析できます。

```sh
python -m image_lens --workspace /path/to/workspace analyze --tracks tracks.json --shots shots.json --point surface0 --start 0 --end 30 --out timing.json
python -m image_lens analyze --help
```

LLM が読むレポートには `--report-bindings bindings.json --format markdown --trajectory-dir traj` を付けてください。すると Markdown の冒頭に大まかなストロークが並びます。中央値フィルターをかけた経路を角と停止で区切ったもので、最大単位の半分以上なら主要（major）ストロークです。`traj/trajectory.png` には、フレーム全体を基準にした経路（早い方が青、遅い方が赤）、ストローク・タイムラインの帯、切り抜いていないフレーム最大 4 枚が表示されます。同じ処理で外観変化の候補も一覧にします。連続フレームの灰色差分が跳ねる箇所で、`compare --frames a,b` で確かめる対象です。主要ストロークが候補と重なる場合は印を付けます。残像（echo）や描き直しで点が跳ぶことがあるためです。ストロークは `brief` で再計算されますが、画像と候補は再計算されません。

## コマンド 1 つで: look

`look` は動画 1 本を、ショットごとの短い動きの要約にして、新しいディレクトリ 1 つにまとめます。

```sh
python -m image_lens --workspace W look --video clip.mp4 --out look-clip
```

モデルのパスは `<workspace>/models.json`（[モデルのパス](#モデルのパス)を参照）から読むか、`--detector`、`--pose-model`、`--checkpoint` で指定します。正確なしきい値と、その根拠になった確認結果は [look ノート](docs/look-notes.md)（英語）にあります。

### 実行する内容

- `scan` と `pose` を 1 回ずつ実行し、ショットごとに姿勢をもとに `seed` / `prepare` / `track` / `bind` / `analyze` を実行します。既定では最初の 16 ショットで、`--max-shots` で 256 まで増やせます。
- 長いショット（160 フレーム超）は複数の区間に分けます。あるグループの点の半分以上が区間の終わりよりかなり前に止まったら、絵が切り替わったとみなし、直後のフレームから新しい区間を始めます。
- 1 フレームだけのショット、使える姿勢の追跡がないショット、短すぎる末尾は、理由を記録して飛ばします。
- 途中で失敗した手順は記録だけして先に進みます。追跡できなかった区間には「추적 결과 없음: 움직임 모름」（追跡結果なし: 動きは不明）と書きます。
- `--max-seconds` は手順の合間にしか確認しないため、1 手順ぶん時間を超えることがあります。
- 1920×1080 より大きい動画は、先に `source-<w>x<h>.mp4` に縮小して使います。その場合、レポートの px は縮小版が基準です。

### 出力ファイル

| ファイル | 内容 |
|---|---|
| `look.md` | 最初に読むファイルです。追跡区間ごとに 1 行です。 |
| `look-full.md` | グループごとの主な動き、一緒に動いた点、警告の数、完全なレポートへのリンク。`focus --look <dir> --shot shot-0007` で 1 ショットだけ表示できます。 |
| `look.json`、`steps.log` | 同じ内容のデータと、手順ごとの実行記録です。 |
| `<window>-onion.jpg`、`onion-sheet-N.jpg`、`frame-sheet-N.jpg`、`check-sheet-N.jpg` | 画像資料です（下記）。 |

### look.md の読み方

出力の文言は韓国語です。先頭の **먼저 확인할 곳**（先に確認する所）には、ツール自身が最も自信のない区間が最大 4 か所並びます。人物のボックスが重なった所、追跡や姿勢が途切れた所、姿勢がなく本文に出てこない所、短いカットが続く所などで、それぞれ `check-sheet-N.jpg` の密なフレームで確認できます。ここにない所も間違っていることがあります。2 人が 1 つのボックスにまとまった場面（Kaguya 18〜22 秒）はこの一覧に入らなかったので、フレームシートも合わせて見てください。

その下の各区間の行はショットの構図（전신＝全身、무릎 위＝膝上、상반신＝上半身、얼굴 클로즈업＝顔のアップ）で始まり、続いて次の内容が並びます。

- **전체**（全体）: 画面全体（カメラ）と体の中心がどう動いたかを一度だけ述べます。
- **몸 기준**（体基準）: その上で腕、脚、名前のない点がそれぞれ何をしたかを、体の中心を基準に述べます。向きが変わると「아래 → 위」（下 → 上）のように分け、「몸 쪽으로/바깥으로」は体の中心線に近づいたか離れたかです。左右は画面基準です。
- **속도**（速度）: 主な動きごとに、時間順の速度を最大 5 つ付けます（例:「속도 9→5→3→1 px/f」）。緩急はこの数値から判断してください。区間の最初のフレームですでに動いていた場合は、始まりは見えていないと書きます。
- **화면 전체**（画面全体）: カメラが 24 px 以上動いたときに出ます。時間順の速度、行って戻った動き（「오른쪽 80px까지 갔다 돌아옴」＝右へ 80px まで行って戻った）、ズーム（「줌 인 ×1.25」＝ズームイン）も書きます。カメラに付いて動いただけのものは省きます。半分以上のフレームを照合できなかったときは「카메라 모름」（カメラ不明）と書き、そのとき画面上の移動にはカメラの動きが混ざっているかもしれません。
- **높이별 흐름**（高さ別の流れ）: 人物の範囲を上下 5 段に分け、段ごとの移動を示します。名前の付いた点が要らないので、スカートの中で突き出した腰や小道具の動きも拾えます。
- **많이 움직인 곳**（よく動いた所、`move-a`…）: 形が最も大きく変わった所に自動で置いた、名前のない点です。

行の読み方が変わる表示もあります。

- 「전환」（切り替え）: 絵が切り替わった所です。2 フレームの姿勢の差だけを書いており、その間の動きは見ていません。
- 「f<N> 뒤로 점 k/n개 끊김」（f<N> 以降、点 k/n 個が途切れた）: 点を見失いました。それ以降は止まったのではなく、不明として読んでください。
- 「골격 신뢰 낮음」（骨格の信頼度が低い）: 姿勢の骨格がおかしいので、顔にだけ名前を付け、体基準の部分は飛ばしました。「(팔만)」（腕のみ）は上半身ショットなので腕だけを見たという意味です。
- 「자세 A」（姿勢 A）: 前に出た姿勢がもう一度出たという意味です。初回だけすべて書き、繰り返しでは違いだけを書きます。
- 姿勢イベント: 手が顔の近くにある（이마·눈/볼/입·턱＝額・目/頬/口・あご）、肩の上がり下がり、「몸 돌림 후보」（体を回した可能性）、「대부분 안 보임: 멈춤 아님, 모름」（ほぼ見えない: 静止ではなく不明）、「포즈 추적이 N프레임 끊김」（姿勢の追跡が N フレーム途切れた）。
- 「사람 N명」（N 人）: 複数の人が出るとき、人ごとの動き、転んだ可能性、触れた可能性、ボックスが重なった所を書きます。

### 画像資料

- `<window>-onion.jpg`: 最初のフレームをシアン、最後のフレームを赤で重ね、段ごとの移動を矢印で描きます。右側の灰色の枠には最初のフレームと最も違うフレームを出すので、行って戻る動きを静止と取り違えません。部位ごとの代表点の位置を、フレームごとに白い縁取りの点で打ちます（始点は大きな円、終点は大きな四角）。点の間隔が広ければ速く、狭ければ遅い動きです。
- `onion-sheet-N.jpg`: 1 回の実行のオニオン画像を、時間範囲とともにまとめたシートです。追跡するには短すぎる区間は「still」1 枚で、体を回した可能性が最も高い 6 区間は 5 フレームの帯で入れます。
- `frame-sheet-N.jpg`: 動画全体から約 1.25 秒ごとに 1 枚、最大 96 枚を均等に選んだフレーム全体です。
- `check-sheet-N.jpg`: 「먼저 확인할 곳」の区間ごとに 1 枚、密なフレームを時間順に並べます。
- `zoom --video <clip> --out <dir> --at 18-22`: 4 秒以内の範囲を最大 2 か所、密なフレームで見せます。読み手は自分が間違えた所をうまく選べないので、まずフレームシートを見てください。

### 1 つの区間を尋ねる: at

「18〜22 秒で何が起きた？」のように気になる区間があれば、要約全体を読み直さず、その区間だけを尋ねてください。

```bash
python -m image_lens --workspace W at --look W/look-clip --span 18-22 --out W/at-18-22
```

`at.md` には、その区間にかかる行、飛ばしたカット、切り替え、「먼저 확인할 곳」の表示と、開くべき画像の名前が集まります。`at-frames-1.jpg` はその区間の密なフレーム 12 枚です。区間は 4 秒以内です。質問は読み手が決め、ツールはその区間の根拠だけを集めます。

### 姿勢のシードがうまくいかないとき: grid と mark

小道具、変わった姿勢、姿勢モデルが名前を知らない部分は、エージェントが自分で点を置いて名前を付けられます。

```bash
python -m image_lens --workspace W grid --video V --frame 572 --box 100,250,1000,1000 --out W/grid-f572.jpg
python -m image_lens --workspace W mark --video V --shots W/look/shots.json --frame 572 --end 599 --point hammer-stripe=430,680 --point eye=322,738 --checkpoint C --out W/mark-s19
```

- `grid` は元のフレームの上に、座標を読み取れる目盛り（ズーム範囲内は 50 px、それ以外は 100 px 間隔）を描きます。
- `mark` は各点を 15 px 以内で最も追跡しやすい角に移し（`--no-snap` ならそのまま）、点に印を付けたクロップを証拠として保存し、追跡して `mark.md` とフレームごとの座標ファイル `mark.tracks.json` を書きます。
- 名前は ASCII で最大 8 個です。解剖学的に確かめた名前ではなく、エージェントが見て付けた名前です。

### モデルのパス

`look`、`pose`、`mark` でモデルのパスを省くと、`<workspace>/models.json` から読みます。

```json
{"detector": "/models/yolox_humanart.onnx", "pose_model": "/models/rtmw133.onnx", "checkpoint": "/models/scaled_offline.pth"}
```

コマンドラインで指定した値が優先します。ワークスペースは Git に入らないので、マシンごとに別々に置きます。

## 姿勢（pose）から全身のシードを作る（任意）

`pose` は標準スキャンに対してローカルの全身姿勢モデルを実行します。YOLOX HumanArt の人物ボックス、RTMW の 133 キーポイント（rtmlib、ONNX Runtime CPU）、検出したカットごとにリセットされる ByteTrack の ID を使います。2 つの ONNX ファイルはユーザーが用意し、サイズと SHA256 で固定します。何もダウンロード・同梱せず、重みのライセンスは確認されていません。同じ環境に `requirements-pose.txt` をインストールしてください。パケットは、姿勢の層そのものが信頼しにくいフレームにも印を付けます。人物ボックスが重なる場合（残像のコピー、残像）と、ひとつのトラック内で体の骨格が跳ぶ場合です。

`seed` はそのパケットの 1 フレームを、新しい観察ジョブに変えます。12 点の表面の契約に収まるよう、グループごとにひとつ作ります：`<job>-head`（鼻、目、前・上・横の髪）、`<job>-arms`、`<job>-legs`。髪の点は目から作った頭の座標系（目の間の距離と、鼻から離れる上方向）の上の幾何学的な位置なので、検出した髪ではなく表面点の候補です。画面外の関節は端に張り付いた状態で返ってくるため、フレームの端に近い関節は除きます。各シードには印を付けた元画像の切り抜きが固定の視覚参照として付きます。追跡する前に必ず見てください。`bind` は追跡を終えたジョブひとつのレポート用バインディングを書き出します。

```sh
python -m image_lens --workspace W pose --shots shots.json --detector /models/yolox_humanart.onnx --pose-model /models/rtmw133.onnx --out pose.json
python -m image_lens --workspace W seed --pose pose.json --frame 215 --job s7 --out s7.obs.json --evidence-dir s7-crops
# s7-head, s7-arms, s7-legs をそれぞれ prepare/track したあと：
python -m image_lens --workspace W bind --queries s7.obs.json --tracks s7-head.tracks.json --job s7-head --out s7-head.bindings.json
python -m image_lens --workspace W analyze --tracks s7-head.tracks.json --shots shots.json --report-bindings s7-head.bindings.json --start 215 --end 307 --format markdown --trajectory-dir s7-head-traj --pose pose.json --out s7-head.md
```

`--pose` を指定すると、軌跡の画像に姿勢が不安定なフレームが表示され（タイムラインのオレンジ色）、その近くの変化候補にタグが付きます。警告はまとめられ、候補や姿勢が不安定な区間と重なる大きなストロークを 1 行にすべて書きます。点が 3 つ以上あると主要ストロークだけにラベルを付け、Markdown の冒頭に点ごとの 1 行要約と「一緒に動いた」行（方向が同じで時間が重なり、大きさが近い複数の点の主要ストローク）が並びます。点ごとの速度区間、段階、長さの比較は JSON に残ります。瞳孔はモデル化しておらず、目の点は描かれた目の領域を追います。

分析のヘルプには、確認対象の選択、明示的な特徴バインディング、元の色の領域レポート、任意の前景検査もあります。解釈する前に [限界](docs/limitations.md) を読んでください。

## 検証と出所

```sh
python -B -m unittest discover -s tests -v
# 任意：主要依存パッケージのスモークテスト（新しい出力フォルダで）
python -I -B tests/smoke_runtime.py --workspace workspace/NEW_SMOKE_DIRECTORY
```

[検証](docs/verification.md) では、テスト済みの CLI・安全装置と、実行していない推論やプラットフォームの範囲を分けて記載しています。[出所とライセンス](docs/provenance.md) には、元のハッシュ、上流リポジトリの固定値、制限付き読み込みのパッチ、自作コードのライセンス境界があります。過去の実験メディア、フィクスチャ、記録、結果、採用しなかった自動分類器は、選ばれた実行経路ではありません。
