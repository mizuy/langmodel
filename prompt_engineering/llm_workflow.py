"""LLM DataFrame Workflow Library

Local LLMを使用してDataFrameの各行を分類し、プロンプト開発のワークフローをサポートするライブラリ。
すべてPolarsベースで実装。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from llm_classifier import LLMClassifier, file_logger

if TYPE_CHECKING:
    from collections.abc import Sequence


"""
Local LLMをもちいて、dataframeを処理して新たな列を計算するライブラリです。

db.procやdb.itemなどのpl.DataFrameを入力として、promptで、新たな列を定義します。
例として、db.procの複数の列(例: id_proc, mo_study_porpose, m_special_instruct, e_comment) を入力として
各行に対して、検査目的 (lm_indication) をLocal LLMで推測するとします。

1. db.procの一部(100-500行)に対して、人間が正解ラベルを作成する
2. Local LLMで推測した値と、正解ラベルの一致度を集計する
3. 不正解のものを人間が精査して、promptの改良を行う
4. 2-3を、正解率が一定以上になるまで繰り返す
5. 最後に、完成したpromptを用いて、db.proc全体に対してLocal LLMで検査目的を推測する

上記work-flowをサポートするライブラリを作ります。
"""


# ============================================================================
# EvaluationResult データクラス
# ============================================================================


@dataclass
class EvaluationResult:
    """評価結果を保持"""

    y_true: list[str]
    y_pred: list[str]
    categories: list[str]
    eval_df: pl.DataFrame | None = None  # 評価に使用したDataFrame（ID列など含む）

    def __post_init__(self) -> None:
        """初期化後に評価指標を計算"""
        # 正解率計算
        correct = [t == p for t, p in zip(self.y_true, self.y_pred, strict=True)]
        self.n_correct = sum(correct)
        self.n_samples = len(self.y_true)
        self.accuracy = self.n_correct / self.n_samples if self.n_samples > 0 else 0.0

    @property
    def incorrect_samples(self) -> pl.DataFrame:
        """不正解サンプルを取得"""
        if self.eval_df is None:
            # eval_dfがない場合は、y_trueとy_predから作成
            return pl.DataFrame(
                {
                    "y_true": self.y_true,
                    "y_pred": self.y_pred,
                }
            ).filter(pl.col("y_true") != pl.col("y_pred"))
        else:
            # eval_dfがある場合は、不正解サンプルを抽出
            return self.eval_df.filter(pl.col("y_true") != pl.col("y_pred"))

    @property
    def classification_report(self) -> pl.DataFrame:
        """分類レポートを生成"""
        report_dict = classification_report(
            self.y_true, self.y_pred, labels=self.categories, output_dict=True, zero_division=0
        )

        # report_dictを展開してDataFrameに変換
        rows = []
        for key, value in report_dict.items():
            if isinstance(value, dict):
                # カテゴリごとのメトリクス
                row = {"class": key}
                row.update(value)
                rows.append(row)
            else:
                # accuracyなどのスカラー値
                rows.append({"class": key, "precision": None, "recall": None, "f1-score": value, "support": None})

        return pl.DataFrame(rows)

    @property
    def confusion_matrix(self) -> pl.DataFrame:
        """混同行列を生成"""
        cm_array = confusion_matrix(self.y_true, self.y_pred, labels=self.categories)
        # 混同行列をDataFrameに変換（行がactual、列がpredicted）
        df = pl.DataFrame(cm_array, schema=self.categories)
        # actual列を追加
        return df.with_columns(pl.Series("actual", self.categories))

    def summary(self) -> None:
        """評価結果のサマリーを表示"""
        print(f"正解率: {self.accuracy:.1%} ({self.n_correct}/{self.n_samples})")
        print(f"\n分類レポート:")
        print(self.classification_report)

    def plot_confusion_matrix(self) -> None:
        """混同行列を可視化（matplotlib/seaborn使用）"""
        cm_df = self.confusion_matrix
        # actual列を除外して数値配列に変換
        cm_array = cm_df.select(self.categories).to_numpy()
        labels = self.categories

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("予測", fontsize=12)
        plt.ylabel("実際", fontsize=12)
        plt.title("混同行列", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_parallel_categories(self):
        """並列カテゴリ図を可視化（plotly express使用）"""
        # y_trueとy_predのDataFrameを作成
        data = pl.DataFrame(
            {
                "actual": self.y_true,
                "predicted": self.y_pred,
            }
        )

        # 実際に存在するカテゴリのみを取得
        actual_categories = set(self.y_true) | set(self.y_pred)
        # self.categoriesから存在するカテゴリのみを抽出してソート
        existing_categories = [cat for cat in self.categories if cat in actual_categories]

        # plotly用にpandasに変換
        df_pd = data.to_pandas()

        # 並列カテゴリ図を作成
        fig = px.parallel_categories(
            df_pd,
            dimensions=["actual", "predicted"],
            labels={"actual": "実際", "predicted": "予測"},
            title="予測と実際のカテゴリ比較",
        )

        # カテゴリの順序を指定（存在するカテゴリのみ）
        fig.update_traces(
            dimensions=[
                {"categoryorder": "array", "categoryarray": existing_categories} for _ in ["actual", "predicted"]
            ]
        )

        # Jupyter notebookで表示するためにfigを返す
        return fig

    def export_errors(self, path: Path) -> None:
        """不正解サンプルをExcelに出力（プロンプト改良用）"""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.incorrect_samples.write_excel(path)


# ============================================================================
# Run クラス
# ============================================================================


@dataclass
class Run:
    """1回の推論試行を表す"""

    run_id: str
    run_dir: Path
    prompt: str
    timestamp: datetime
    id_column: str
    target_column: str
    categories: list[str]

    @classmethod
    def create(
        cls,
        run_dir: Path,
        id_column: str,
        target_column: str,
        categories: Sequence[str],
        classifier: LLMClassifier,
        df: pl.DataFrame,
        training_ids: list[str],
        batch_size: int = 10,
    ) -> Run:
        """新しいrunを作成し、推論実行とデータ保存を行う

        Args:
            run_dir: runディレクトリのパス
            id_column: ID列名
            target_column: ターゲット列名（例: "indication"）
            categories: 分類カテゴリのリスト
            classifier: LLM分類器
            df: 入力DataFrame（id_columnとそれ以外の必要な列を含む）
            training_ids: Training SetのIDリスト
            batch_size: バッチサイズ

        Returns:
            Runオブジェクト
        """
        run_dir.mkdir(parents=True, exist_ok=True)

        # プロンプトを保存
        prompt_path = run_dir / "prompt.txt"
        prompt_path.write_text(classifier.prompt, encoding="utf-8")

        # Training Setに対して推論（run専用のファイルロガーを使用）
        logfile = run_dir / "log.txt"
        subset_df = df.filter(pl.col(id_column).is_in(training_ids))
        with file_logger(logfile) as run_logger:
            results = classifier.classify(subset_df, batch_size, logger=run_logger)

        # 元のDataFrameに推論結果を追加
        # subset_dfとresultsを結合（id_columnで）
        merged_df = subset_df.join(results, on=id_column, how="left")

        # 結果を保存（parquet形式）
        results_path = run_dir / "results.parquet"
        merged_df.write_parquet(results_path)

        # 結果をCSV形式でも保存
        results_csv_path = run_dir / "results.csv"
        merged_df.write_csv(results_csv_path)

        # timestampを保存
        timestamp = datetime.now()
        timestamp_path = run_dir / "timestamp.txt"
        timestamp_path.write_text(timestamp.isoformat(), encoding="utf-8")

        # run_config.jsonに設定を保存
        config = {
            "id_column": id_column,
            "target_column": target_column,
            "categories": list(categories),
        }
        config_path = run_dir / "run_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Runオブジェクトを作成
        run_id = run_dir.name
        return cls(
            run_id=run_id,
            run_dir=run_dir,
            prompt=classifier.prompt,
            timestamp=timestamp,
            id_column=id_column,
            target_column=target_column,
            categories=list(categories),
        )

    @classmethod
    def from_dir(cls, run_dir: Path) -> Run:
        """ディレクトリからRunを読み込み"""
        run_id = run_dir.name

        # timestamp.txtから読み込み
        timestamp_path = run_dir / "timestamp.txt"
        if timestamp_path.exists():
            timestamp_str = timestamp_path.read_text(encoding="utf-8").strip()
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = datetime.now()

        # prompt.txtから読み込み
        prompt_path = run_dir / "prompt.txt"
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
        else:
            prompt = ""

        # run_config.jsonから設定を読み込み
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            with config_path.open() as f:
                config = json.load(f)
            id_column = config.get("id_column", "")
            target_column = config.get("target_column", "")
            categories = config.get("categories", [])
        else:
            # 後方互換性のため、デフォルト値を設定
            id_column = ""
            target_column = ""
            categories = []

        return cls(
            run_id=run_id,
            run_dir=run_dir,
            prompt=prompt,
            timestamp=timestamp,
            id_column=id_column,
            target_column=target_column,
            categories=categories,
        )

    def load_results(self) -> pl.DataFrame:
        """推論結果を読み込み"""
        results_path = self.run_dir / "results.parquet"
        if results_path.exists():
            return pl.read_parquet(results_path)
        results_csv_path = self.run_dir / "results.csv"
        if results_csv_path.exists():
            return pl.read_csv(results_csv_path)
        raise ValueError(f"Results file not found: {results_path} or {results_csv_path}")

    def evaluate(self, truth: pl.DataFrame) -> EvaluationResult:
        """truth DataFrameを使用してEvaluationResultを計算

        Args:
            truth: 正解ラベルDataFrame [id_column, target_column, ...]

        Returns:
            EvaluationResult
        """
        # 結果を読み込み
        predictions = self.load_results()

        # 予測列名と正解列名を決定
        pred_col = f"lm_{self.target_column}"
        truth_col = self.target_column

        # 列の存在確認
        if pred_col not in predictions.columns:
            raise ValueError(f"予測列 '{pred_col}' が見つかりません")
        if truth_col not in truth.columns:
            raise ValueError(f"正解列 '{truth_col}' が見つかりません")

        # 結合
        merged = predictions.join(truth, on=self.id_column, how="inner")

        # 評価対象のみ抽出（両方に値がある行）
        eval_df = merged.filter(pl.col(pred_col).is_not_null() & pl.col(truth_col).is_not_null())

        if len(eval_df) == 0:
            raise ValueError("No valid samples for evaluation")

        # y_true, y_predを取得
        y_true = eval_df[truth_col].to_list()
        y_pred = eval_df[pred_col].to_list()

        # 評価用DataFrameにy_true, y_pred列を追加（incorrect_samples計算用）
        eval_df_with_labels = eval_df.with_columns(
            [
                pl.Series("y_true", y_true),
                pl.Series("y_pred", y_pred),
            ]
        )

        return EvaluationResult(
            y_true=y_true,
            y_pred=y_pred,
            categories=self.categories,
            eval_df=eval_df_with_labels,
        )


# ============================================================================
# Experiment クラス
# ============================================================================


@dataclass
class Experiment:
    """プロンプト開発実験全体を管理"""

    base_dir: Path
    training_ids: list[str]
    id_column: str
    target_column: str
    categories: list[str]
    truth: pl.DataFrame
    runs: list[Run] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        base_dir: Path,
        training_ids: list[str],
        id_column: str,
        target_column: str,
        categories: Sequence[str],
        truth: pl.DataFrame,
    ) -> Experiment:
        """新規実験を作成

        Args:
            base_dir: 実験ディレクトリ
            training_ids: Training SetのIDリスト
            id_column: ID列名
            target_column: ターゲット列名（例: "indication"）
            categories: 分類カテゴリのリスト
            truth: 正解ラベルDataFrame [id_column, target_column, ...]
        """
        base_dir.mkdir(parents=True, exist_ok=True)

        # ID列とtarget_column列の存在確認
        if id_column not in truth.columns:
            raise ValueError(f"ID column '{id_column}' not found in truth")
        if target_column not in truth.columns:
            raise ValueError(f"Target column '{target_column}' not found in truth")

        return cls(
            base_dir=base_dir,
            training_ids=training_ids,
            id_column=id_column,
            target_column=target_column,
            categories=list(categories),
            truth=truth,
            runs=[],
        )

    @classmethod
    def load(
        cls,
        base_dir: Path,
        truth: pl.DataFrame,
        id_column: str,
        target_column: str,
        categories: Sequence[str],
    ) -> Experiment:
        """既存の実験を読み込み

        Args:
            base_dir: 実験ディレクトリ
            truth: 正解ラベルDataFrame [id_column, target_column, ...]
            id_column: ID列名
            target_column: ターゲット列名（例: "indication"）
            categories: 分類カテゴリのリスト
        """
        # ID列とtarget_column列の存在確認
        if id_column not in truth.columns:
            raise ValueError(f"ID column '{id_column}' not found in truth")
        if target_column not in truth.columns:
            raise ValueError(f"Target column '{target_column}' not found in truth")

        # Training SetはtruthのID列から構築
        training_ids = truth[id_column].drop_nulls().unique().to_list()

        # Runs読み込み
        runs = []
        history_path = base_dir / "history.json"
        if history_path.exists():
            with history_path.open() as f:
                history = json.load(f)
            for run_id in history.get("runs", []):
                run_dir = base_dir / run_id
                if run_dir.exists():
                    runs.append(Run.from_dir(run_dir))
        else:
            # ディレクトリから直接検索
            for run_dir in sorted(base_dir.glob("run_*")):
                if run_dir.is_dir():
                    runs.append(Run.from_dir(run_dir))

        return cls(
            base_dir=base_dir,
            training_ids=training_ids,
            id_column=id_column,
            target_column=target_column,
            categories=list(categories),
            truth=truth,
            runs=runs,
        )

    def _next_run_num_for_name(self, name: str) -> int:
        """base_dir内の run_{name}_* を調べ、最大の番号+1を返す。存在しなければ1。"""
        pattern = re.compile(rf"^run_{re.escape(name)}_(\d+)$")
        max_num = 0
        if self.base_dir.exists():
            for p in self.base_dir.iterdir():
                if not p.is_dir():
                    continue
                m = pattern.match(p.name)
                if m:
                    max_num = max(max_num, int(m.group(1)))
        return max_num + 1

    def run(
        self,
        classifier: LLMClassifier,
        df: pl.DataFrame,
        *,
        name: str = "default",
        batch_size: int = 10,
    ) -> Run:
        """新しいrunを実行し、prompt/resultsを保存（評価は行わない）

        Args:
            classifier: LLM分類器
            df: 入力DataFrame（id_columnとそれ以外の必要な列を含む）
            name: runの名前。run_dirは run_{name}_{num:03} となる。nameごとに番号は独立。
            batch_size: バッチサイズ

        Returns:
            Runオブジェクト
        """
        # 次のrun_idを決定（既存の run_{name}_* の最大番号+1。len(self.runs)は使わない）
        next_run_num = self._next_run_num_for_name(name)
        run_id = f"run_{name}_{next_run_num:03d}"
        run_dir = self.base_dir / run_id

        # Run.create()を呼び出してrunを作成
        run = Run.create(
            run_dir=run_dir,
            id_column=self.id_column,
            target_column=self.target_column,
            categories=self.categories,
            classifier=classifier,
            df=df,
            training_ids=self.training_ids,
            batch_size=batch_size,
        )

        # runsリストに追加
        self.runs.append(run)

        # history.jsonを更新
        self._save_history()

        return run

    def _save_history(self) -> None:
        """history.jsonを保存"""
        history = {
            "runs": [run.run_id for run in self.runs],
        }
        history_path = self.base_dir / "history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def history(self) -> pl.DataFrame:
        """全runの履歴をDataFrameで取得（評価結果は含まない）"""
        if not self.runs:
            return pl.DataFrame(
                {
                    "run_id": [],
                    "timestamp": [],
                },
            )

        return pl.DataFrame(
            {
                "run_id": [r.run_id for r in self.runs],
                "timestamp": [r.timestamp for r in self.runs],
            },
        )

    def plot_accuracy(self) -> None:
        """正解率の推移をグラフ表示（全runsを評価してから表示）"""
        if not self.runs:
            print("No runs to plot")
            return

        # 全runsを評価
        eval_results = []
        for run in self.runs:
            try:
                eval_result = run.evaluate(self.truth)
                eval_results.append(eval_result.accuracy)
            except Exception:
                eval_results.append(0.0)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(eval_results) + 1), eval_results, marker="o")
        plt.xlabel("Run")
        plt.ylabel("Accuracy")
        plt.title("正解率の推移")
        plt.grid(True)
        plt.xticks(range(1, len(self.runs) + 1), [r.run_id for r in self.runs])
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def best_run(self) -> Run:
        """最も正解率の高いrunを取得（全runsを評価してから判定）"""
        if not self.runs:
            raise ValueError("No runs available")

        # 全runsを評価してaccuracyを取得
        run_accuracies = []
        for run in self.runs:
            try:
                eval_result = run.evaluate(self.truth)
                run_accuracies.append((run, eval_result.accuracy))
            except Exception:
                run_accuracies.append((run, 0.0))

        return max(run_accuracies, key=lambda x: x[1])[0]

    def last_run(self) -> Run:
        """最後のrunを読み込み"""
        runs = self.runs
        if not runs:
            raise ValueError("No runs available")
        return runs[-1]
