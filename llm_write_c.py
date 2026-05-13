#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI


def load_env_if_exists(env_path: str = ".env") -> None:
    """Load .env variables into process env if available."""
    path = Path(env_path)
    if not path.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=path, override=False)
        return
    except Exception:
        pass

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v).strip()


def normalize_binary(v: Any) -> int:
    s = _to_text(v)
    if not s:
        return 0

    try:
        num = float(s)
        return 1 if num >= 0.5 else 0
    except Exception:
        pass

    low = s.lower()
    if low in {"true", "yes", "y"}:
        return 1
    if low in {"false", "no", "n"}:
        return 0
    return 0


def infer_stage_index(text: str) -> int | None:
    t = _to_text(text)
    if not t:
        return None

    # Stage keywords are checked from later-stage to earlier-stage for disambiguation.
    s4_keywords = ["施工", "安装", "调试", "联调", "设备到货", "到货", "设备制作", "实施"]
    s3_keywords = ["深化设计", "深化", "技术冻结", "冻结", "需求深度调研", "接口细化", "接口"]
    s2_keywords = ["方案比选", "比选", "招标", "招采", "采购", "容量配置", "技术选型"]
    s1_keywords = ["规范", "初步设计", "初设", "准备工作", "前置准备", "前期准备", "需求澄清"]

    if any(k in t for k in s4_keywords):
        return 3
    if any(k in t for k in s3_keywords):
        return 2
    if any(k in t for k in s2_keywords):
        return 1
    if any(k in t for k in s1_keywords):
        return 0
    return None


def normalize_stage_one_hot(s1: Any, s2: Any, s3: Any, s4: Any, ref_text: str) -> tuple[int, int, int, int]:
    vals = [normalize_binary(s1), normalize_binary(s2), normalize_binary(s3), normalize_binary(s4)]
    total = sum(vals)
    inferred = infer_stage_index(ref_text)

    if total == 1:
        return vals[0], vals[1], vals[2], vals[3]

    if total > 1:
        if inferred is not None and vals[inferred] == 1:
            keep = inferred
        else:
            keep = next(i for i, v in enumerate(vals) if v == 1)
        out = [0, 0, 0, 0]
        out[keep] = 1
        return out[0], out[1], out[2], out[3]

    # total == 0
    keep = inferred if inferred is not None else 2  # Default to S3 when no clear signal.
    out = [0, 0, 0, 0]
    out[keep] = 1
    return out[0], out[1], out[2], out[3]


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def build_messages(system_val: str, description: str) -> list[dict[str, str]]:
    system_prompt = (
        "你是机场建设任务结构化分析助手。"
        "请根据机场的建设文本 description 输出结构化字段。"
        "必须只输出一个 JSON 对象，不能输出额外文字。"
        "JSON 字段必须严格为: "
        "space, task_text, target_s1, target_s2, target_s3, target_s4。"
        "target_s1~target_s4 只能为 0/1，且四个字段中只能有一个字段为 1。"
    )

    user_prompt = f"""
请分析建设任务描述并输出字段：
- system(输入参考): {system_val}
- description(建设文本): {description}

输出要求：
1) space: 空间层级（格式为"建筑/片区-功能区-子区域/对象"，例如：指廊-公共区域-飞行区站坪除冰机位，不可照抄成建筑/片区-功能区-子区域/对象）。
2) task_text: 基于 description 输出任务文本，内容应聚焦：
   - 所属系统/专业
   - 工程行为类型
   - 前置条件
   - 对未来运行的影响
   - 技术冻结相关性
   - 协同阶段承载特征
3) target_s1~target_s4: 0/1 且只允许一个为 1，按任务最适宜介入阶段判断：
   - S1：xxx的规范初步设计 / 准备工作
   - S2：xxx的方案比选 / 招标
   - S3：深化设计 / 需求深度调研
   - S4：施工 / 安装调试（包含设备制作到货）

仅输出 JSON。
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def should_process_row(row: pd.Series) -> bool:
    # Keep the original rule: first 4 columns all present.
    for idx in [0, 1, 2, 3]:
        if _to_text(row.iloc[idx] if len(row) > idx else "") == "":
            return False
    return True


async def call_deepseek_with_retry(
    client: AsyncOpenAI,
    model: str,
    system_val: str,
    description: str,
    max_retries: int,
    timeout: int,
) -> dict[str, Any]:
    fallback_s1, fallback_s2, fallback_s3, fallback_s4 = normalize_stage_one_hot(0, 0, 0, 0, description)
    fallback = {
        "space": "UNKNOWN",
        "task_text": "UNKNOWN",
        "target_s1": fallback_s1,
        "target_s2": fallback_s2,
        "target_s3": fallback_s3,
        "target_s4": fallback_s4,
    }

    for attempt in range(max_retries + 1):
        try:
            messages = build_messages(system_val, description)
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                ),
                timeout=timeout,
            )

            content = (resp.choices[0].message.content or "").strip()
            obj = extract_json(content)

            space = _to_text(obj.get("space", "")) or "UNKNOWN"
            task_text = _to_text(obj.get("task_text", "")) or "UNKNOWN"
            ref_text = f"{description} {task_text}"
            s1, s2, s3, s4 = normalize_stage_one_hot(
                obj.get("target_s1", 0),
                obj.get("target_s2", 0),
                obj.get("target_s3", 0),
                obj.get("target_s4", 0),
                ref_text,
            )

            return {
                "space": space,
                "task_text": task_text,
                "target_s1": s1,
                "target_s2": s2,
                "target_s3": s3,
                "target_s4": s4,
            }
        except Exception as e:
            if attempt >= max_retries:
                print(f"[WARN] row failed after retries: {e}")
                return fallback
            await asyncio.sleep(2 ** attempt)

    return fallback


async def process_one_row(
    idx: int,
    row: pd.Series,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    max_retries: int,
    timeout: int,
) -> tuple[int, dict[str, Any]]:
    # Fixed input: col1=system(reference), col2=description
    system_val = _to_text(row.iloc[0] if len(row) > 0 else "")
    description = _to_text(row.iloc[1] if len(row) > 1 else "")

    async with sem:
        result = await call_deepseek_with_retry(
            client=client,
            model=model,
            system_val=system_val,
            description=description,
            max_retries=max_retries,
            timeout=timeout,
        )
    return idx, result


async def process_dataframe(
    df: pd.DataFrame,
    client: AsyncOpenAI,
    model: str,
    batch_size: int,
    concurrency: int,
    max_retries: int,
    timeout: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    n = len(df)

    def col_or_empty(col_idx: int) -> list[Any]:
        if df.shape[1] > col_idx:
            return df.iloc[:, col_idx].tolist()
        return [""] * n

    current_space = [_to_text(v) for v in col_or_empty(4)]
    current_task_text = [_to_text(v) for v in col_or_empty(5)]
    current_s1 = col_or_empty(6)
    current_s2 = col_or_empty(7)
    current_s3 = col_or_empty(8)
    current_s4 = col_or_empty(9)

    results: list[dict[str, Any]] = []
    for i in range(n):
        s1, s2, s3, s4 = normalize_stage_one_hot(
            current_s1[i],
            current_s2[i],
            current_s3[i],
            current_s4[i],
            _to_text(df.iloc[i, 1] if df.shape[1] > 1 else ""),
        )
        results.append(
            {
                "space": current_space[i],
                "task_text": current_task_text[i],
                "target_s1": s1,
                "target_s2": s2,
                "target_s3": s3,
                "target_s4": s4,
            }
        )

    eligible_indices = [i for i, (_, row) in enumerate(df.iterrows()) if should_process_row(row)]
    eligible_set = set(eligible_indices)

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(eligible_indices)

    for start in range(0, total, batch_size):
        batch_idx = eligible_indices[start : start + batch_size]
        tasks = [
            process_one_row(
                idx=i,
                row=df.iloc[i],
                client=client,
                sem=sem,
                model=model,
                max_retries=max_retries,
                timeout=timeout,
            )
            for i in batch_idx
        ]

        out = await asyncio.gather(*tasks)
        for i, value in out:
            results[i] = value

        completed = min(start + batch_size, total)
        print(f"Progress: {completed}/{total} (eligible rows)")

    return results, eligible_set


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure output columns exist through col10.
    while df.shape[1] < 10:
        df[f"__extra_{df.shape[1] + 1}"] = ""

    cols = list(df.columns)
    cols[4] = "space"
    cols[5] = "task_text"
    cols[6] = "target_s1"
    cols[7] = "target_s2"
    cols[8] = "target_s3"
    cols[9] = "target_s4"
    df.columns = cols
    return df


async def run(args: argparse.Namespace) -> None:
    load_env_if_exists(args.env_file)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", args.base_url)
    if not api_key:
        raise ValueError(f"缺少环境变量 DEEPSEEK_API_KEY。已尝试加载: {Path(args.env_file).resolve()}")

    df = pd.read_excel(args.input_xlsx)
    df = prepare_dataframe(df)

    # Keep output columns as object to avoid dtype warnings during writes.
    df = df.astype(
        {
            df.columns[4]: "object",
            df.columns[5]: "object",
            df.columns[6]: "object",
            df.columns[7]: "object",
            df.columns[8]: "object",
            df.columns[9]: "object",
        }
    )

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    results, eligible_set = await process_dataframe(
        df=df,
        client=client,
        model=args.model,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    df.iloc[:, 4] = [r["space"] for r in results]
    df.iloc[:, 5] = [r["task_text"] for r in results]
    df.iloc[:, 6] = [r["target_s1"] for r in results]
    df.iloc[:, 7] = [r["target_s2"] for r in results]
    df.iloc[:, 8] = [r["target_s3"] for r in results]
    df.iloc[:, 9] = [r["target_s4"] for r in results]

    # Keep original behavior: only force-fill LLM rows; non-eligible rows remain as-is.
    for i in range(len(df)):
        if i in eligible_set:
            space_val = _to_text(df.iat[i, 4])
            task_val = _to_text(df.iat[i, 5])
            df.iat[i, 4] = space_val or "UNKNOWN"
            df.iat[i, 5] = task_val or "UNKNOWN"

            ref_text = f"{_to_text(df.iat[i, 1] if df.shape[1] > 1 else '')} {task_val}"
            s1, s2, s3, s4 = normalize_stage_one_hot(df.iat[i, 6], df.iat[i, 7], df.iat[i, 8], df.iat[i, 9], ref_text)
            df.iat[i, 6] = s1
            df.iat[i, 7] = s2
            df.iat[i, 8] = s3
            df.iat[i, 9] = s4

    output_path = Path(args.output_csv)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Done. CSV saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="异步调用 DeepSeek 批量回填 space/task_text/target_s1-s4 并导出 CSV"
    )
    parser.add_argument("--input-xlsx", default=r"D:\construction-operation\construction\task.xlsx", help="输入 Excel 路径")
    parser.add_argument("--output-csv", default="output.csv", help="输出 CSV 路径")
    parser.add_argument("--env-file", default=".env", help=".env 文件路径，默认当前目录 .env")
    parser.add_argument("--model", default="deepseek-v4-flash", help="模型名，默认 deepseek-v4-flash")
    parser.add_argument("--batch-size", type=int, default=10, help="每轮处理条数，默认 10")
    parser.add_argument("--concurrency", type=int, default=10, help="并发上限，默认 10")
    parser.add_argument("--max-retries", type=int, default=4, help="单条最大重试次数，默认 4")
    parser.add_argument("--timeout", type=int, default=60, help="单次请求超时秒数，默认 60")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="DeepSeek base url（可被环境变量覆盖）")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
