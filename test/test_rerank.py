"""
测试 rerank 模型的独立脚本，逻辑与 dify_rerank 一致。
用法：
    python test_rerank.py
    python test_rerank.py --query "自定义查询" --model models/BAAI/bge-reranker-v2-m3
"""

import argparse
import os
from time import perf_counter

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

RERANK_MODEL = os.getenv("RERANK_MODEL", "models/BAAI/bge-reranker-v2-m3")

TEST_QUERY = "什么是机器学习？"
TEST_DOCUMENTS = [
    "机器学习是人工智能的一个分支，旨在让计算机从数据中学习。",
    "北京今天天气晴朗，温度 25 摄氏度。",
    "深度学习使用多层神经网络来进行模式识别。",
    "Python 是一门广泛使用的编程语言，常用于数据科学。",
    "自然语言处理是机器学习的一个重要应用领域。",
    "足球比赛中，两队各 11 名球员，目标是将球踢入对方球门。",
    "Transformer 架构极大地推动了 NLP 领域的发展。",
    "2024 年夏季奥运会在巴黎举行。",
    "监督学习和无监督学习是机器学习的两种主要范式。",
    "意大利面是一种来自意大利的传统美食。",
]


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    model: CrossEncoder | None = None,
):
    """与 dify_rerank 核心逻辑一致"""
    if model is None:
        raise RuntimeError("rerank model not loaded")

    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs, show_progress_bar=False)

    indexed = list(enumerate(scores.tolist()))
    indexed.sort(key=lambda x: x[1], reverse=True)

    top_n = top_n or len(indexed)
    results = [
        {
            "index": orig_idx,
            "document": documents[orig_idx],
            "relevance_score": float(score),
        }
        for orig_idx, score in indexed[:top_n]
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description="测试 rerank 模型")
    parser.add_argument(
        "--model",
        default=RERANK_MODEL,
        help=f"模型路径 (默认: {RERANK_MODEL})",
    )
    parser.add_argument(
        "--query",
        default=TEST_QUERY,
        help=f"查询文本 (默认: {TEST_QUERY!r})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="返回前 N 个结果 (0=全部, 默认: 0)",
    )
    args = parser.parse_args()

    # 加载模型
    t0 = perf_counter()
    print(f"Loading model: {args.model}")
    model = CrossEncoder(args.model)
    print(f"Loaded in {perf_counter() - t0:.1f}s\n")

    # 执行 rerank
    t0 = perf_counter()
    results = rerank(
        query=args.query,
        documents=TEST_DOCUMENTS,
        top_n=args.top_n or None,
        model=model,
    )
    elapsed = perf_counter() - t0

    # 打印结果
    print(f"Query: {args.query!r}\n")
    print(f"{'Rank':<6}{'Score':<10}{'Index':<8}{'Document'}")
    print("-" * 80)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<6}{r['relevance_score']:<10.4f}{r['index']:<8}{r['document']}")

    print(f"\nTop {len(results)} / {len(TEST_DOCUMENTS)} results, {elapsed:.3f}s")


if __name__ == "__main__":
    main()
