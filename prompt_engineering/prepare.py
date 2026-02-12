# ruff: noqa
import polars as pl
import endolab as el
from pathlib import Path
import subprocess

@el.snapshot_cache(Path("data/cs_proc.parquet"))
def prepare_data():
    import nccdb

    db = nccdb.get_db()
    return (
        db.cs.proc.pp.with_year()
        .filter(~pl.col("rectal_scopy").fill_null(False))
        .filter(~(pl.col("is_hcs") == True))
        .select(
            [
                "id_proc",
                "date",
                "year",
                "nendo",
                "ym",
                "j_検査目的",
                "mo_study_purpose",
                "m_special_instruct",
                "hx_surgery",
                "hx_er",
                "hx_crt",
                "hx_ibd",
                # "is_hcs",
                "e_comment",
            ]
        )
        .with_columns(
            pl.col("date").cast(pl.String),
            pl.col("hx_surgery").cast(pl.Int64).fill_nan(0),
            pl.col("hx_er").cast(pl.Int64).fill_nan(0),
            pl.col("hx_crt").cast(pl.Int64).fill_nan(0),
            pl.col("hx_ibd").cast(pl.Int64).fill_nan(0),
            pl.col("m_special_instruct")
            .replace("★★ここを右クリック★★  ★★「基準日選択」を行ってください★★", "")
            .cast(pl.String),
            # pl.col("is_hcs").cast(pl.Int64).fill_nan(0),
        )
    )


def generate_new_validate_df(n, df: pl.DataFrame, exclude_me: pl.DataFrame | None = None):
    if exclude_me is not None:
        df = df.filter(~pl.col("id_proc").is_in(exclude_me["id_proc"]))
    return df.sample(n).with_columns(pl.lit(None).alias("indication"))

def show_exp(experiment, which_run: str = "best"):
    """
    which_run: "best" (デフォルト) で最良のrunを、"latest" で最新のrunの詳細を表示
    """
    # 履歴を確認
    print(experiment.history())

    # 正解率の推移をグラフ表示
    experiment.plot_accuracy()

    # 各runのaccuracyをtableで表示
    rows = []
    for run in experiment.runs:
        eval_result = run.evaluate(experiment.truth)
        rows.append({"run_id": run.run_id, "accuracy": f"{eval_result.accuracy:.1%}"})
    if rows:
        acc_df = pl.DataFrame(rows)
        with pl.Config(set_tbl_width_chars=100, set_tbl_rows=100):
            display(acc_df)

    # 表示するrunを選択（best or latest）
    if len(experiment.runs) == 0:
        return
    if which_run == "latest":
        target_run = experiment.runs[-1]
        label = "最新"
    else:
        target_run = experiment.best_run()
        label = "最良"

    eval_result = target_run.evaluate(experiment.truth)
    print(f"\n{label}のrun: {target_run.run_id} (accuracy={eval_result.accuracy:.1%})")
    print(f"正解率: {eval_result.accuracy:.1%} ({eval_result.n_correct}/{eval_result.n_samples})")

    with pl.Config(set_tbl_width_chars=100, set_tbl_rows=100):
        display(eval_result.classification_report)

    # 混同行列を可視化
    eval_result.plot_confusion_matrix()

    # 不正解サンプルを確認
    print(f"\n不正解サンプル数: {len(eval_result.incorrect_samples)}")
    eval_result.eval_df.drop("y_true", "y_pred").with_columns(
        (pl.col("indication") == pl.col("lm_indication")).alias("correct")
    ).write_excel("eval_result.xlsx")
    eval_result.incorrect_samples.write_excel("errors.xlsx")
    print("→ errors.xlsxを確認してプロンプトを改良してください")
# generate_new_validate_df(500, df, exclude_me=trainig_df).write_csv(path_validate)

def get_getway_ipaddress():
    # WSL2からWindowsのIPを取得する
    return subprocess.check_output(
        "ip route show | grep -i default | awk '{ print $3}'",
        shell=True,
        text=True
    ).strip()