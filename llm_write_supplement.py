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

    # Preferred: python-dotenv
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=path, override=False)
        return
    except Exception:
        pass

    # Fallback: minimal parser
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


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # fenced json block
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # first object fallback
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def build_messages(code: str, description: str, stakeholder: str, current_system: str) -> list[dict[str, str]]:
    system_prompt = (
        "你是机场运营需求结构化分析助手。"
        "请根据输入表格前4列信息输出运营需求文本。"
        "current_system 仅供参考，可能为空或有误，你可以纠正。"
        "必须只输出一个 JSON 对象，不能输出任何额外文字。"
        "JSON 字段必须严格为: need_text。"
    )

    user_prompt = f"""
请分析这条机场运营需求：
- 编码(code): {code}
- 描述(description): {description}
- stakeholder: {stakeholder}
- 当前 system(可为空或有误，仅供参考): {current_system}

输出要求：
1) 只生成 need_text
2) need_text 应基于前4列信息补充成一段清晰、完整、可用于后续匹配建模的运营需求文本。
3) 内容应聚焦：
   - 想解决的问题
   - 依赖的系统与空间
   - 所需前置条件
   - 对建设侧的潜在约束
   - 最适宜介入的阶段
4) 不要编造具体编号、数量、日期、责任人等输入中没有的信息。

仅输出 JSON 对象，格式为 {{"need_text": "..."}}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def call_deepseek_with_retry(
    client: AsyncOpenAI,
    model: str,
    code: str,
    description: str,
    stakeholder: str,
    current_system: str,
    max_retries: int,
    timeout: int,
) -> dict[str, Any]:
    fallback = {"need_text": "UNKNOWN"}

    for attempt in range(max_retries + 1):
        try:
            messages = build_messages(code, description, stakeholder, current_system)
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

            need_text = _to_text(obj.get("need_text", "")) or "UNKNOWN"
            return {"need_text": need_text}
        except Exception as e:
            if attempt >= max_retries:
                print(f"[WARN] row failed after retries: {e}")
                return fallback

            msg = str(e).lower()
            maybe_rate_limit = ("429" in msg) or ("rate limit" in msg) or ("too many requests" in msg)
            backoff = (2 ** attempt) + (0.5 if maybe_rate_limit else 0.0)
            await asyncio.sleep(backoff)

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
    # Fixed schema: col1=code, col2=description, col3=stakeholder, col4=system
    code = _to_text(row.iloc[0] if len(row) > 0 else "")
    description = _to_text(row.iloc[1] if len(row) > 1 else "")
    stakeholder = _to_text(row.iloc[2] if len(row) > 2 else "")
    current_system = _to_text(row.iloc[3] if len(row) > 3 else "")

    async with sem:
        result = await call_deepseek_with_retry(
            client=client,
            model=model,
            code=code,
            description=description,
            stakeholder=stakeholder,
            current_system=current_system,
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
) -> list[dict[str, Any]]:
    n = len(df)
    results: list[dict[str, Any]] = [{"need_text": "UNKNOWN"} for _ in range(n)]

    sem = asyncio.Semaphore(concurrency)
    completed = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df.iloc[start:end]

        tasks = [
            process_one_row(
                idx=i,
                row=row,
                client=client,
                sem=sem,
                model=model,
                max_retries=max_retries,
                timeout=timeout,
            )
            for i, (_, row) in enumerate(batch.iterrows(), start=start)
        ]

        out = await asyncio.gather(*tasks)
        for i, result in out:
            results[i] = result

        completed = end
        print(f"Progress: {completed}/{n}")

    return results


def find_need_text_col(df: pd.DataFrame) -> int | None:
    for i, col in enumerate(df.columns):
        if str(col).strip().lower() == "need_text":
            return i
    return None


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 4:
        raise ValueError("Input Excel must contain at least 4 columns: code, description, stakeholder, system.")
    if find_need_text_col(df) is None:
        df["need_text"] = ""
    return df


async def run(args: argparse.Namespace) -> None:
    load_env_if_exists(args.env_file)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", args.base_url)
    if not api_key:
        raise ValueError(
            f"缺少环境变量 DEEPSEEK_API_KEY。已尝试加载: {Path(args.env_file).resolve()}"
        )

    df = pd.read_excel(args.input_xlsx)
    df = prepare_dataframe(df)
    need_text_idx = find_need_text_col(df)
    if need_text_idx is None:
        raise RuntimeError("need_text column was not created.")
    # Avoid dtype warnings when writing strings into existing columns.
    df.iloc[:, need_text_idx] = df.iloc[:, need_text_idx].astype("object")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    results = await process_dataframe(
        df=df,
        client=client,
        model=args.model,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    # Write back only need_text; all other columns are preserved.
    cleaned_need_text = pd.Series([r["need_text"] for r in results], index=df.index)
    cleaned_need_text = cleaned_need_text.fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")
    df.iloc[:, need_text_idx] = cleaned_need_text

    output_path = Path(args.output_csv)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Done. CSV saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="异步调用 DeepSeek，根据前4列批量回填 need_text 并导出 CSV"
    )
    parser.add_argument("--input-xlsx", default="input.xlsx", help="输入 Excel 路径")
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
