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

MOSCOW_ALLOWED = {"Must", "Should", "Could", "Wont"}


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


def normalize_moscow(v: Any) -> str:
    s = _to_text(v)
    if not s:
        return "UNKNOWN"

    low = s.lower().replace("-", "").replace("_", "").replace(" ", "")
    mapping = {
        "must": "Must",
        "musthave": "Must",
        "should": "Should",
        "shouldhave": "Should",
        "could": "Could",
        "couldhave": "Could",
        "wont": "Wont",
        "wonthave": "Wont",
        "wontfix": "Wont",
        "wonot": "Wont",
        "won't": "Wont",
        "wonth": "Wont",
    }
    if s in MOSCOW_ALLOWED:
        return s
    return mapping.get(low, "UNKNOWN")


def normalize_stage_score(v: Any) -> float:
    s = _to_text(v)
    if not s:
        return 0.0

    try:
        num = float(s)
        if num < 0:
            return 0.0
        if num > 1:
            return 1.0
        return num
    except Exception:
        pass

    low = s.lower()
    if low in {"true", "yes", "y"}:
        return 1.0
    if low in {"false", "no", "n"}:
        return 0.0
    return 0.0


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


def build_messages(description: str, stakeholder: str, current_system: str) -> list[dict[str, str]]:
    system_prompt = (
        "你是机场运营需求结构化分析助手。"
        "请根据 description、stakeholder 和 current_system 输出结构化结果。"
        "current_system 仅供参考，可能为空或有误，你可以纠正。"
        "必须只输出一个 JSON 对象，不能输出任何额外文字。"
        "JSON 字段必须严格为: "
        "system, space, jobs_to_be_done, moscow, need_text, target_s1, target_s2, target_s3, target_s4。"
        "其中 moscow 只能是 Must/Should/Could/Wont。"
        "space 为空间层级，格式为 建筑或片区-功能区-子区域或对象。"
        "target_s1 到 target_s4 取值范围为[0,1]取小数，表示该运营需求自身适宜介入阶段的程度，不针对某个具体建设任务。"
    )

    user_prompt = f"""
请分析这条机场运营需求：
- 描述(description): {description}
- stakeholder: {stakeholder}
- 当前 system(可为空或有误，仅供参考): {current_system}

输出要求：
1) system: 一句简短的系统或模块归属，某个需求可能对应1个或多个系统，若有多个请用逗号分隔
2) space: 空间层级（建筑/片区-功能区-子区域/对象），例如：指廊-公共区域-飞行区站坪除冰机位，某个需求可能对应1个或多个空间，若有多个请用逗号分隔
3) jobs_to_be_done: 该需求想解决的核心问题（短语）
4) moscow: 只能是 Must/Should/Could/Wont
5) need_text: 基于 description 输出运营需求文本，内容应聚焦：
   - 想解决的问题
   - 依赖的系统与空间
   - 所需前置条件
   - 对建设侧的潜在约束
   - 最适宜介入的阶段
6) target_s1/target_s2/target_s3/target_s4: 取值范围为[0,1]取小数，表示该运营需求自身适宜介入阶段的程度，共同构成目标阶段向量。
   - S1：需求澄清与前置准备阶段（准备、规范研究、初步设计）
   - S2：方案配置与技术比选阶段（方案比选、容量配置、招标准备）
   - S3：深化设计与技术冻结阶段（深化设计、接口细化、需求深度落实）
   - S4：施工实施与调试验证阶段（施工、安装调试、设备到货、联调验证）

仅输出 JSON 对象。
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def call_deepseek_with_retry(
    client: AsyncOpenAI,
    model: str,
    description: str,
    stakeholder: str,
    current_system: str,
    max_retries: int,
    timeout: int,
) -> dict[str, Any]:
    fallback = {
        "system": "UNKNOWN",
        "space": "UNKNOWN",
        "jobs_to_be_done": "UNKNOWN",
        "moscow": "UNKNOWN",
        "need_text": "UNKNOWN",
        "target_s1": 0.0,
        "target_s2": 0.0,
        "target_s3": 0.0,
        "target_s4": 0.0,
    }

    for attempt in range(max_retries + 1):
        try:
            messages = build_messages(description, stakeholder, current_system)
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

            fixed_system = _to_text(obj.get("system", "")) or "UNKNOWN"
            space = _to_text(obj.get("space", "")) or "UNKNOWN"
            jobs = _to_text(obj.get("jobs_to_be_done", "")) or "UNKNOWN"
            moscow = normalize_moscow(obj.get("moscow", ""))
            need_text = _to_text(obj.get("need_text", "")) or "UNKNOWN"
            target_s1 = normalize_stage_score(obj.get("target_s1", 0))
            target_s2 = normalize_stage_score(obj.get("target_s2", 0))
            target_s3 = normalize_stage_score(obj.get("target_s3", 0))
            target_s4 = normalize_stage_score(obj.get("target_s4", 0))

            return {
                "system": fixed_system,
                "space": space,
                "jobs_to_be_done": jobs,
                "moscow": moscow,
                "need_text": need_text,
                "target_s1": target_s1,
                "target_s2": target_s2,
                "target_s3": target_s3,
                "target_s4": target_s4,
            }
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
    description = _to_text(row.iloc[1] if len(row) > 1 else "")
    stakeholder = _to_text(row.iloc[2] if len(row) > 2 else "")
    current_system = _to_text(row.iloc[3] if len(row) > 3 else "")

    async with sem:
        result = await call_deepseek_with_retry(
            client=client,
            model=model,
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
    results: list[dict[str, Any]] = [
        {
            "system": "UNKNOWN",
            "space": "UNKNOWN",
            "jobs_to_be_done": "UNKNOWN",
            "moscow": "UNKNOWN",
            "need_text": "UNKNOWN",
            "target_s1": 0.0,
            "target_s2": 0.0,
            "target_s3": 0.0,
            "target_s4": 0.0,
        }
        for _ in range(n)
    ]

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


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    while df.shape[1] < 12:
        df[f"__extra_{df.shape[1] + 1}"] = ""

    cols = list(df.columns)
    # Keep col3 stakeholder untouched.
    cols[3] = "system"
    cols[4] = "space"
    cols[5] = "jobs to be done"
    cols[6] = "moscow"
    cols[7] = "need_text"
    cols[8] = "target_s1"
    cols[9] = "target_s2"
    cols[10] = "target_s3"
    cols[11] = "target_s4"
    df.columns = cols
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
    # Avoid dtype warnings when writing strings into numeric columns.
    df = df.astype(
        {
            df.columns[3]: "object",
            df.columns[4]: "object",
            df.columns[5]: "object",
            df.columns[6]: "object",
            df.columns[7]: "object",
            df.columns[8]: "object",
            df.columns[9]: "object",
            df.columns[10]: "object",
            df.columns[11]: "object",
        }
    )

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

    # Write back generated fields.
    df.iloc[:, 3] = [r["system"] for r in results]
    df.iloc[:, 4] = [r["space"] for r in results]
    df.iloc[:, 5] = [r["jobs_to_be_done"] for r in results]
    df.iloc[:, 6] = [normalize_moscow(r["moscow"]) for r in results]
    df.iloc[:, 7] = [r["need_text"] for r in results]
    df.iloc[:, 8] = [normalize_stage_score(r["target_s1"]) for r in results]
    df.iloc[:, 9] = [normalize_stage_score(r["target_s2"]) for r in results]
    df.iloc[:, 10] = [normalize_stage_score(r["target_s3"]) for r in results]
    df.iloc[:, 11] = [normalize_stage_score(r["target_s4"]) for r in results]

    # Clean text columns by fixed positions to avoid duplicate-name ambiguity.
    for idx in [2, 3, 4, 5, 6, 7]:
        cleaned = (
            df.iloc[:, idx]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .replace("", "UNKNOWN")
        )
        df.iloc[:, idx] = cleaned

    for idx in [8, 9, 10, 11]:
        df.iloc[:, idx] = df.iloc[:, idx].map(normalize_stage_score).astype("float64")

    output_path = Path(args.output_csv)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Done. CSV saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="异步调用 DeepSeek 批量回填 system/space/jobs/moscow/need_text/target_s1-s4 并导出 CSV"
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
