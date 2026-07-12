# LLMオンボーディングサマリー

この文書は、MoGe リポジトリに参加する新任 LLM エージェントが、プロジェクトの目的・制約・実行方法・作業境界を短時間で把握するための初期資料です。作業開始前にこの文書と [README.md](../README.md) を確認してください。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** MoGe（Accurate Monocular Geometry Estimation）。単一の RGB 画像から、3D 点群、深度、法線、カメラ視野角などの幾何情報を推定する PyTorch ベースの研究・推論ソフトウェア。
- **最終成果物:** MoGe-1 / MoGe-2 の学習済みモデルを利用した画像推論機能。画像ごとの depth、normal、point map、mask、FOV の可視化・保存、および GLB / PLY の 3D 出力。
- **ビジネス背景・価値:** オープンドメイン画像から 3D 構造を復元し、3D コンテンツ生成、画像理解、計測、可視化などの後段処理に利用できる。MoGe-2 は metric scale と sharp details を重視し、normal 対応モデルも提供する。
- **現時点の進捗サマリ:** リポジトリルートに uv 環境と `uv.lock` を整備済みで、CUDA GPU 上で `Ruicheng/moge-2-vits-normal` による normal 推論を確認済み。実行結果は `outputs/moge2-normal-noenv/` に保存されている。

## 2. クリティカルな要求・制約

> 以下は「壊してはいけない」品質・仕様ラインです。

- Python パッケージのインストール・実行は、リポジトリルートから `uv sync` / `uv run ...` を使う。システム Python や無関係な仮想環境へ直接インストールしない。
- `pyproject.toml` と `uv.lock` の依存関係を同期させる。OpenCV は EXR 出力互換性のため `opencv-python==4.10.0.84` を維持する。
- 推論時の既定モデルは MoGe-2 v2 の `Ruicheng/moge-2-vitl-normal`。軽量な動作確認には `Ruicheng/moge-2-vits-normal` を明示指定できる。
- `moge/scripts/infer.py` と `moge/scripts/cli.py` が OpenCV の EXR 保存を有効化する `OPENCV_IO_ENABLE_OPENEXR=1` を設定している。これを壊さない。
- GPU が利用可能な環境では `--device cuda` を優先する。CPU へ切り替える場合は、速度・メモリ・検証結果への影響を作業記録に明記する。
- モデル重みは Hugging Face Hub から取得される。認証警告と実際のダウンロード失敗を区別し、モデルや重みをリポジトリへ無断追加しない。
- 推論結果には少なくとも入力画像、depth 可視化、normal（normal 対応モデルの場合）、mask、FOV を含める。出力ディレクトリを Git 管理対象に含めるかは作業目的ごとに確認する。
- 変更前に既存の作業ツリーを確認し、他者の未コミット変更を破棄しない。破壊的な Git 操作はユーザーの明示許可なしに実行しない。

## 3. 参照すべき合意済み資料

新任エージェントは、作業内容に応じて次の一次資料を確認する。

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義書 | [README.md](../README.md) | プロジェクト概要、対応モデル、インストール、最小コード例、CLI オプションの基準。 |
| 要件定義書 | [pyproject.toml](../pyproject.toml) | パッケージ名、依存関係、Python バージョン、`moge` CLI エントリポイント。 |
| 依存ロック | [uv.lock](../uv.lock) | uv で再現する具体的な依存バージョンと Git 依存の固定情報。 |
| 実行手順 | [moge/scripts/infer.py](../moge/scripts/infer.py) | 画像入力、モデルロード、推論、maps / GLB / PLY 出力の実装。 |
| CLI 登録 | [moge/scripts/cli.py](../moge/scripts/cli.py) | `moge app`、`moge infer` などのコマンド登録。 |
| normal 仕様 | [normal.md](normal.md) | MoGe-2 normal 推定の背景、学習方法、normal 出力の説明。 |
| ONNX 仕様 | [onnx.md](onnx.md) | MoGe-2 の ONNX 変換、出力テンソル、後処理上の注意。 |
| 学習手順 | [train.md](train.md) | データ形式、設定、学習コードを変更する際の参照資料。 |
| 評価手順 | [eval.md](eval.md) | ベンチマークと baseline 評価の実行方法。 |
| 既知課題・作業記録 | [workdoc](../temp/workdoc_Jul12-2026_moge2_uv_setup.md) | 今回の uv 構築、OpenCV 互換性問題、MoGe-2 normal 実推論の証跡。 |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク

- `moge/` 以下の推論処理、出力形式、可視化、CLI の小規模な改善。
- 既存の `docs/` の補足、再現手順の整理、実行結果・既知問題の文書化。
- `uv run` を使った import 確認、CLI help、サンプル画像による smoke test、既存テストの実行。
- 変更に対応する最小限の回帰テストや検証スクリプトの追加。
- MoGe-2 の v2 / normal 対応コードの調査と、既存 API を壊さない範囲のリファクタリング。

### 任せないタスク

- 学習済み重みの再配布、Hugging Face モデルの削除・更新、外部サービスへの公開。
- GitHub への push、Issue / PR の作成・更新、外部ユーザーへの連絡。明示的な依頼がある場合のみ行う。
- ベンチマーク結果や精度を、実測なしに新しい数値として断定すること。
- `uv.lock` や OpenCV 固定を、互換性検証なしに更新すること。
- データセット、秘密鍵、アクセストークン、個人情報をリポジトリへ追加すること。
- ユーザーの未コミット変更を reset / checkout で破棄すること。

## 5. インタラクション方針

- **回答スタイル:** 日本語のビジネス敬語。結論を先に書き、必要に応じて見出し・箇条書き・実行コマンドを使う。
- **回答手順:** 前提と作業範囲 → 実施内容 → 検証結果 → 未解決事項・次のコマンドの順に整理する。
- **禁止事項・注意:** 未確認のモデル精度・実行成功を断定しない。コマンドの終了コード、エラー本文、生成物を根拠として報告する。代替手段を使った場合は理由を明記する。
- **秘匿情報の扱い:** `HF_TOKEN` などのトークン値、SSH 秘密鍵、環境変数の秘密部分を表示・記録・コミットしない。必要なら「設定済み / 未設定」の状態だけ報告する。
- **変更方針:** 実装変更前に対象ファイルと影響範囲を調査し、変更後は `git diff` と関連コマンドで確認する。ドキュメントだけの変更でもリンク先とコマンドの整合性を確認する。

## 6. 試行タスク（オンボーディング演習）

理解度確認のため、次の順で実施する。すべてリポジトリルートから実行する。

1. **環境診断:** `uv sync` を実行し、`uv run python -c 'import torch; print(torch.cuda.is_available())'` で uv と CUDA の状態を記録する。期待結果は環境が同期され、GPU 環境では `True` が出ること。
2. **CLI 確認:** `uv run moge infer --help` を実行し、`--pretrained`、`--device`、`--maps`、`--glb`、`--ply` の役割を説明する。期待結果は help が終了コード0で表示されること。
3. **normal smoke test:** `example_images/01_HouseIndoor.jpg` に対して `Ruicheng/moge-2-vits-normal` を `--device cuda --resize 512 --resolution_level 0 --maps` で実行し、出力ディレクトリ内の `normal.png` と `depth.exr` の存在を確認する。期待結果は終了コード0であること。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** 変更理由、対象ファイル、実行した検証コマンド、結果、未解決事項を同じ変更単位で記録する。コマンドはリポジトリルートから再実行できる形で書く。
- **TBDの扱い:** 未確認事項は `TBD` と明記し、確認方法・担当・確認期限または次のアクションを併記する。推測で埋めない。
- **レビュー/承認フロー:** 変更前後の `git diff` を確認し、関連テストまたは smoke test を実行する。外部公開・依存更新・モデル変更はユーザー確認を得てから確定する。
- **その他の運用ルール:** 依存変更後は `uv sync` と `uv.lock` の差分を確認する。推論成果物は検証用と配布用を区別し、サイズの大きい重みや秘密情報をコミットしない。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:** リポジトリルート、Python パッケージ `moge/`、CLI `moge/scripts/`、文書 `docs/`、サンプル `example_images/`。
- **代表的なコマンド:**
  ```bash
  cd <repository-root>
  uv sync
  uv run moge infer --help
  uv run moge infer -i example_images/01_HouseIndoor.jpg -o outputs/moge2-normal --pretrained Ruicheng/moge-2-vits-normal --device cuda --resize 512 --resolution_level 0 --maps
  uv run python -c 'import torch; print(torch.cuda.is_available())'
  ```
- **依存ライブラリ:** PyTorch、Torchvision、OpenCV、SciPy、Matplotlib、Trimesh、Pillow、Hugging Face Hub、Gradio、`utils3d`、`pipeline`。正確な解決結果は [uv.lock](../uv.lock) を参照する。
- **連絡先/責任者:** リポジトリ所有者・依頼者に確認する。リポジトリ内に別途指定された責任者情報がない場合、推測して補完しない。

> この文書は必要に応じて拡張・縮退する。記入済みの内容はバージョン管理し、実際のコード・コマンド・検証結果と乖離した場合は先に更新する。
