"""LLM分類用クラスとロギングユーティリティ."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING
import logging

import polars as pl
from loguru import logger as _default_logger
from openai import APIConnectionError, OpenAI
from requests.exceptions import ConnectionError as RequestsConnectionError
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence
    from loguru import Logger


@contextmanager
def file_logger(logfile: Path, level: str = "DEBUG"):
    """ファイル専用ロガーを提供するコンテキストマネージャ.

    - 標準出力には一切出さず、指定ファイルのみに出力します。
    - 標準の logging.Logger を返すので、loguru とは独立したロギングが可能です。
    """
    py_logger = logging.getLogger(f"llm_classifier.file.{logfile}")
    py_logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    handler = logging.FileHandler(logfile, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    py_logger.addHandler(handler)

    try:
        yield py_logger
    finally:
        py_logger.removeHandler(handler)
        handler.close()


class LLMClassifier:
    """Local LLMを使用した分類器."""

    def __init__(
        self,
        prompt: str,
        categories: Sequence[str],
        id_column: str,
        target_column: str,
        *,
        multiple_choice_prompt: bool = False,
        api_base: str = "http://localhost:1234/v1",
        model: str = "local-model",
    ):
        """分類タスクの設定を保持.

        Args:
            prompt: LLMに送るプロンプトテンプレート
            categories: 分類カテゴリのリスト
            id_column: ID列名
            target_column: ターゲット列名（例: "indication"）
            multiple_choice_prompt: 複数選択プロンプトの場合True（|区切りの処理を行う）
            api_base: LM Studio APIのベースURL
            model: モデル名
        """
        self.prompt = prompt
        self.categories = list(categories)
        self.id_column = id_column
        self.target_column = target_column
        self.multiple_choice_prompt = multiple_choice_prompt
        self.api_base = api_base
        self.model = model

        # ローカル接続向けにNO_PROXYを設定
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"

        self.client = OpenAI(base_url=api_base, api_key="lm_studio")

    def classify(
        self,
        df: pl.DataFrame,
        batch_size: int = 10,
        logger: "Logger | None" = None,
    ) -> pl.DataFrame:
        """DataFrameの各行を分類.

        Args:
            df: 入力DataFrame（id_columnとそれ以外の必要な列を含む）
            batch_size: バッチサイズ
            logger: ロガー（loguruのLogger、Noneの場合はデフォルトのloggerを使用）

        Returns:
            [id_column, response] のDataFrame
            - response: LLMが報告したすべての検査目的（パイプ区切り文字列）
        """
        log = logger or _default_logger

        log.info(f"Classification started at {datetime.now()}")
        results = self._process_batches(df, batch_size, log)

        # post_process: responseを処理してlm_{target_column}を作成
        return self._post_process(results)

    def classify_with_cache(
        self,
        df: pl.DataFrame,
        cache_dir: Path,
        *,
        groupby: str,
        batch_size: int = 10,
        logger: "Logger | None" = None,
    ) -> pl.DataFrame:
        """大きなDataFrameをgroupbyしつつキャッシュしながら分類する汎用メソッド.

        各グループごとに ``{group_value}.parquet`` というファイル名で ``cache_dir`` に保存します。
        既にキャッシュが存在するグループはLLMを呼ばずに読み込みのみを行います。

        Args:
            df: 入力DataFrame（id_columnとそれ以外の必要な列を含む）
            cache_dir: キャッシュを保存するディレクトリ
            groupby: グループ化する列名（例: "ym"）
            batch_size: classify時のバッチサイズ
            logger: ロガー（None の場合はデフォルトロガー）

        Returns:
            全グループ分の結果を縦結合したDataFrame
        """
        log = logger or _default_logger

        cache_dir.mkdir(parents=True, exist_ok=True)

        if groupby not in df.columns:
            raise ValueError(f"groupby列 '{groupby}' が見つかりません")

        all_results: list[pl.DataFrame] = []

        # Polars の group_by をそのまま利用してグループごとに処理
        with tqdm(total=len(df), desc=f"Processing groups by '{groupby}'") as pbar:
            for (group_key,), group_df in df.group_by(groupby, maintain_order=True):
                # group_key は単一キーならスカラー、複数キーならタプル
                group_name_str = str(group_key)

                cache_file = cache_dir / f"{group_name_str}.parquet"

                # キャッシュがあれば読み込み
                if cache_file.exists():
                    log.info(f"Loading cached results for {groupby}={group_name_str}")
                    cached = pl.read_parquet(cache_file)
                    all_results.append(cached)
                    pbar.update(len(group_df))
                    continue

                log.info(f"Running classify for {groupby}={group_name_str} ({len(group_df)} rows)")
                group_result = self.classify(group_df, batch_size=batch_size, logger=log)

                group_result.write_parquet(cache_file)
                all_results.append(group_result)
                pbar.update(len(group_df))

        return pl.concat(all_results) if all_results else pl.DataFrame()

    def _get_response(self, batch_df: pl.DataFrame, logger: "Logger") -> str:
        """バッチデータからLLMの応答を取得."""
        batch_json = json.dumps(batch_df.to_dicts(), ensure_ascii=False)
        expected_ids = batch_df[self.id_column].to_list()

        logger.debug("-" * 40)
        logger.debug(f"Batch: {expected_ids}")

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": self.prompt,
                    },
                    {"role": "user", "content": batch_json},
                ],
                temperature=0.0,
            )
            logger.debug(f"Response:\n{response.output_text}")

        except (APIConnectionError, RequestsConnectionError) as e:
            # Connection Errorの場合はリトライせずに例外を再発生
            logger.error(f"CONNECTION ERROR: {e}")
            raise

        return response.output_text

    def _process_single_batch(
        self,
        batch_df: pl.DataFrame,
        logger: "Logger",
    ) -> pl.DataFrame | None:
        """1バッチを処理して結果を返す."""
        expected_ids = batch_df[self.id_column].to_list()

        response_text = self._get_response(batch_df, logger)

        # LLMの応答をパース
        results: list[dict[str, str]] = []

        for line in [l.strip() for l in response_text.strip().split("\n") if l.strip()]:
            # CSV形式: id,検査目的1|検査目的2|... をパース
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                result_id = parts[0]
                # パイプ区切りの検査目的を文字列として保存
                responses = [p.strip() for p in parts[1].split("|") if p.strip()]
                if not responses or not all(item in self.categories for item in responses):
                    return None

                results.append({self.id_column: result_id, "response": "|".join(responses)})

        if not results:
            logger.warning("No results parsed from response for batch %s", expected_ids)
            return None

        # idの順序チェック（listベース）
        result_ids = [r[self.id_column] for r in results]
        if result_ids != expected_ids:
            logger.error("id mismatch in batch (order check)")
            logger.error("Expected: %s", expected_ids)
            logger.error("Got: %s", result_ids)
            # ミスマッチの場合はNoneを返してリトライを促す
            return None

        return pl.DataFrame(results)

    def _process_batches(
        self,
        df: pl.DataFrame,
        batch_size: int,
        logger: "Logger",
    ) -> pl.DataFrame:
        """バッチ処理を実行し、失敗時はbatch_size=1でリトライ.

        Args:
            df: 入力DataFrame
            batch_size: バッチサイズ
            logger: ロガー（loguruのLogger）

        Returns:
            [id_column, response] のDataFrame
        """
        all_results: list[pl.DataFrame] = []

        # 行インデックスのリストを作成してbatchedで分割
        row_indices = list(range(len(df)))
        n_batches = (len(df) + batch_size - 1) // batch_size

        with tqdm(total=len(df), desc="Processing batches") as pbar:
            for batch_num, batch_indices in enumerate(batched(row_indices, batch_size), start=1):
                # バッチの開始位置と長さを取得
                start_idx = batch_indices[0]
                batch_length = len(batch_indices)
                batch_df = df.slice(start_idx, batch_length)

                logger.info(f"Processing batch {batch_num}/{n_batches} ({len(batch_df)} rows)")

                batch_results = self._process_single_batch(batch_df, logger)

                if batch_results is not None:
                    all_results.append(batch_results)
                    pbar.update(len(batch_results))
                else:
                    logger.warning(
                        "Batch %s failed, retrying with batch_size=1 for %d rows",
                        batch_num,
                        len(batch_df),
                    )

                    # 各レコードを個別に処理
                    for idx in batch_indices:
                        single_row_df = df.slice(idx, 1)
                        single_result = self._process_single_batch(single_row_df, logger)
                        if single_result is not None:
                            all_results.append(single_result)
                            pbar.update(1)
                        else:
                            # 個別処理でも失敗した場合は空の結果を追加
                            logger.error(
                                "Failed to process single row: %s",
                                single_row_df[self.id_column].to_list()[0],
                            )
                            all_results.append(
                                pl.DataFrame(
                                    {
                                        self.id_column: single_row_df[self.id_column].to_list(),
                                        "response": [None],
                                    },
                                )
                            )
                            pbar.update(1)

        return (
            pl.concat(all_results)
            if all_results
            else pl.DataFrame(schema={self.id_column: pl.Utf8, "response": pl.Utf8})
        )

    def _post_process(self, results: pl.DataFrame) -> pl.DataFrame:
        """response列を処理してlm_{target_column}列を作成.

        Args:
            results: [id_column, response] のDataFrame

        Returns:
            [id_column, response, lm_raw_{target_column}, lm_{target_column}] のDataFrame
        """
        if len(results) == 0:
            return results

        # multiple_choice_promptがTrueの場合のみ|の処理を行う
        if self.multiple_choice_prompt:
            results = results.with_columns(
                pl.col("response").alias(f"lm_raw_{self.target_column}"),
                pl.col("response")
                .str.split("|")
                .cast(pl.List(pl.Enum(self.categories)))
                .list.min()
                .alias(f"lm_{self.target_column}"),
            )
        else:
            results = results.with_columns(
                pl.col("response").alias(f"lm_{self.target_column}"),
            )

        return results
