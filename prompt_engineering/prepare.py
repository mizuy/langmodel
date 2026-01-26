# ruff: noqa
import polars as pl
import endolab as el
from pathlib import Path


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


# generate_new_validate_df(500, df, exclude_me=trainig_df).write_csv(path_validate)
