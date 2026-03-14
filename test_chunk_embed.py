"""
直接运行测试 chunk_and_embed 逻辑，使用真实模型。
运行方式: uv run python test_chunk_embed.py
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chunker import chunk_text

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# ── 加载模型 ──────────────────────────────────────────────────────────────────
print(f"加载模型: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
DIM = model.get_sentence_embedding_dimension()
print(f"向量维度: {DIM}\n")


def chunk_and_embed(text: str, max_chars: int = 500):
    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        return []
    vectors = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    return [{"text": c, "char_count": len(c), "embedding": v.tolist()} for c, v in zip(chunks, vectors)]


def run(label: str, text: str, max_chars: int = 100):
    print(f"{'─'*60}")
    print(f"[{label}]  max_chars={max_chars}")
    print(f"输入长度: {len(text)} 字")
    results = chunk_and_embed(text, max_chars=max_chars)
    for i, r in enumerate(results):
        vec_preview = r["embedding"][:4]
        print(f"  chunk {i+1}: {r['char_count']} 字 | 向量前4维{[round(x,4) for x in vec_preview]} | {r['text'][:40]!r}...")
    # 验证：每个 chunk 不超过 max_chars
    violations = [r for r in results if r["char_count"] > max_chars]
    if violations:
        print(f"  ⚠ 超限 chunk 数: {len(violations)}")
    else:
        print(f"  ✓ 所有 chunk 均 ≤ {max_chars} 字，共 {len(results)} 个")
    print()
    return results


# ── 测试用例 ──────────────────────────────────────────────────────────────────

# 1. 短文本，整体在 max_chars 之内，应只有 1 个 chunk
run("短文本，单一chunk",
    "今天天气很好，适合出门散步。",
    max_chars=100)

# 2. 按段落分割：两段各自独立，不应合并
run("多段落，段落边界优先",
    "第一段内容，描述了一些事情。\n\n第二段内容，描述了另一些事情。",
    max_chars=30)

# 3. 长段落按句子切分
run("长段落，句子级切分",
    "自然语言处理是人工智能的重要分支。它研究人类语言与计算机之间的交互。"
    "文本向量化是其中的核心技术之一。通过将文字转换为向量，机器可以理解语义。",
    max_chars=40)

# 4. 超长单句，按子句（逗号）切分兜底
run("超长单句，子句级兜底",
    "这是一个非常非常非常非常非常长的句子，它没有句号，只有逗号，"
    "所以需要在逗号处切分，才能保证每个chunk不超过max_chars的限制。",
    max_chars=30)

# 5. 完全无标点，硬截断兜底
run("无标点文本，硬截断",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ",
    max_chars=20)

# 6. 空文本
print("─"*60)
print("[空文本]")
result = chunk_and_embed("   ", max_chars=100)
assert result == [], f"期望空列表，得到: {result}"
print("  ✓ 空文本返回空列表\n")

# 7. 真实长文（模拟 RAG 场景）
long_text = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

机器学习是人工智能的核心子领域之一。它使计算机能够从数据中自动学习和改进，而无需显式编程。
深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的工作方式。

自然语言处理（NLP）是AI的另一个重要分支，专注于使计算机能够理解、解释和生成人类语言。
文本向量化技术将文本转换为数值向量，使得机器能够进行语义计算和相似度比较。
BERT、GPT等预训练模型的出现极大地推动了NLP领域的发展，使得语义理解能力大幅提升。

向量数据库用于高效存储和检索高维向量，是构建现代RAG系统的基础设施之一。
Milvus、Pinecone、Weaviate等是当前主流的向量数据库产品。
"""

run("真实长文（RAG场景）", long_text.strip(), max_chars=80)

print("全部测试完成。")
