"""Semantic search skill bằng Qdrant.

Ý tưởng
-------
Thay vì nhồi mô tả của TẤT CẢ skill vào system prompt (cách `agent.py` đang làm —
16 skill của `anthropic_skills` tốn ~9.6k ký tự, gửi lại ở MỌI vòng ReAct), ta:

  1. **Build index (một lần):** quét một thư mục gốc chứa nhiều folder skill, với mỗi
     skill lưu vào Qdrant: **đường dẫn tuyệt đối**, **description**, và **vector
     embedding của description**.
  2. **Mỗi câu hỏi:** embed câu hỏi -> similarity search -> lấy **top-K skill** giống
     nhất -> chỉ nạp K skill đó vào prompt cho agent tự chọn.

Điểm cộng: catalog gửi cho LLM không còn phình theo số lượng skill — thêm 100 skill
mới vào thư viện cũng không làm prompt dài thêm.

Thành phần
----------
* `discover_skills(root)` — tìm mọi thư mục có `SKILL.md` (đệ quy, nên gom được cả
  `skills/Telecom_Skills/<skill>` lẫn `skills/anthropic_skills/<skill>`).
* `Embedder` — 2 lựa chọn:
    - `fastembed` (mặc định): chạy LOCAL, không cần API key.
    - `openai`: dùng endpoint OpenAI-compatible (VD Qwen3-Embedding trên gateway nội bộ).
* `SkillIndex` — bọc Qdrant: `ensure_collection` / `upsert_skills` / `add_skill` /
  `remove_skill` / `search` / `list_all`.
* `open_index(...)` — factory đọc cấu hình từ tham số hoặc biến môi trường.

Qdrant chạy ở đâu
-----------------
* Có `QDRANT_URL` (VD `http://localhost:6333`) -> nối tới **server**.
* Không có -> dùng **Qdrant nhúng, lưu on-disk** tại `SKILL_INDEX_PATH`
  (mặc định `<AgenticSkills>/.qdrant_skills`). Không cần dựng service nào.
  Lưu ý: bản nhúng KHOÁ thư mục — một tiến trình dùng tại một thời điểm. Muốn vừa
  chạy `nat serve` vừa chạy CLI index thì hãy dùng server.

Biến môi trường
---------------
| Env | Ý nghĩa | Mặc định |
|---|---|---|
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant server | (không có -> dùng bản nhúng) |
| `SKILL_INDEX_PATH` | thư mục Qdrant nhúng | `<AgenticSkills>/.qdrant_skills` |
| `SKILL_INDEX_COLLECTION` | tên collection | `agentic_skills` |
| `SKILL_INDEX_EMBEDDER` | `fastembed` \| `openai` | `fastembed` |
| `SKILL_INDEX_EMBED_MODEL` | tên model embedding | tuỳ provider |
| `LLM_BASE_URL` / `MODEL_API_KEY` | dùng khi embedder = `openai` | — |
"""

import logging
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import QdrantClient, models

from skills_ref.errors import ParseError, ValidationError
from skills_ref.parser import read_properties

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_COLLECTION = "agentic_skills"
DEFAULT_INDEX_PATH = PROJECT_ROOT / ".qdrant_skills"
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OPENAI_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Namespace cố định -> cùng một đường dẫn skill luôn cho cùng một point id.
# Nhờ vậy upsert lại là IDEMPOTENT (cập nhật, không nhân bản).
_ID_NAMESPACE = uuid.UUID("6f1c2f7e-9a1e-4f9d-8b3c-0f2a1b7c5d40")


# --------------------------------------------------------------------------- #
# Bản ghi
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SkillRecord:
    """Một skill trong index."""

    name: str
    path: str          # ĐƯỜNG DẪN TUYỆT ĐỐI tới folder skill
    description: str
    group: str = ""    # thư mục cha (VD "Telecom_Skills") — tiện để lọc

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(_ID_NAMESPACE, self.path))


@dataclass(frozen=True)
class SkillHit(SkillRecord):
    """Một skill trả về từ similarity search."""

    score: float = 0.0


def discover_skills(root: Path | str) -> list[SkillRecord]:
    """Tìm ĐỆ QUY mọi thư mục có `SKILL.md` dưới `root`.

    Nhờ đệ quy, `root=skills/` gom được cả `skills/Telecom_Skills/<skill>` lẫn
    `skills/anthropic_skills/<skill>`. Skill hỏng frontmatter thì bỏ qua kèm cảnh
    báo, không làm hỏng cả lần build.
    """
    root = Path(root).resolve()
    records: list[SkillRecord] = []

    for skill_md in sorted(root.rglob("SKILL.md")):
        skill_dir = skill_md.parent
        try:
            props = read_properties(skill_dir)
        except (ParseError, ValidationError) as exc:
            logger.warning("[skill_index] bỏ qua '%s': %s", skill_dir.name, exc)
            continue
        if not props:
            continue
        parent = skill_dir.parent
        records.append(
            SkillRecord(
                name=props.name,
                path=str(skill_dir),
                description=props.description,
                group=parent.name if parent != root else "",
            ))

    return records


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    """Tối thiểu cần: biết số chiều, embed nhiều văn bản, embed một truy vấn."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class FastEmbedEmbedder:
    """Embedding chạy LOCAL bằng `fastembed` — không cần API key, không cần mạng
    sau lần tải model đầu tiên."""

    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.model_name = model_name
        self.dim = len(next(iter(self._model.embed(["probe"]))))
        logger.info("[skill_index] embedder=fastembed model=%s dim=%d", model_name, self.dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()


class OpenAIEmbedder:
    """Embedding qua endpoint OpenAI-compatible (VD Qwen3-Embedding trên vLLM)."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_EMBED_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        from langchain_openai import OpenAIEmbeddings

        base_url = base_url or os.getenv("EMBED_BASE_URL") or os.getenv("LLM_BASE_URL")
        api_key = (api_key or os.getenv("EMBED_API_KEY") or os.getenv("MODEL_API_KEY")
                   or os.getenv("OPENAI_API_KEY") or "EMPTY")

        # `check_embedding_ctx_length=False`: endpoint không phải OpenAI thật, không
        # dùng tokenizer tiktoken của họ để cắt chunk.
        self._model = OpenAIEmbeddings(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            check_embedding_ctx_length=False,
        )
        self.model_name = model_name
        self.dim = len(self._model.embed_query("probe"))
        logger.info("[skill_index] embedder=openai model=%s base_url=%s dim=%d", model_name, base_url, self.dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


def make_embedder(provider: str | None = None, model_name: str | None = None) -> Embedder:
    """Tạo embedder theo `provider` (mặc định lấy từ `SKILL_INDEX_EMBEDDER`)."""
    provider = (provider or os.getenv("SKILL_INDEX_EMBEDDER") or "fastembed").lower()
    model_name = model_name or os.getenv("SKILL_INDEX_EMBED_MODEL")

    if provider == "fastembed":
        return FastEmbedEmbedder(model_name or DEFAULT_FASTEMBED_MODEL)
    if provider == "openai":
        return OpenAIEmbedder(model_name or DEFAULT_OPENAI_EMBED_MODEL)
    raise ValueError(f"SKILL_INDEX_EMBEDDER không hợp lệ: {provider!r}. Chỉ nhận 'fastembed' hoặc 'openai'.")


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
def _embed_text(record: SkillRecord) -> str:
    """Văn bản đem đi embed. Ghép TÊN + DESCRIPTION: tên skill thường chứa từ khoá
    có ích ("dien-giai-cuoc", "pptx") mà description có thể diễn đạt khác đi."""
    return f"{record.name}\n{record.description}"


class SkillIndex:
    """Bọc một collection Qdrant chứa các skill."""

    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    # ---- schema ---------------------------------------------------------- #

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
            )
            logger.info("[skill_index] tạo collection '%s' (dim=%d, cosine)", self.collection, self.embedder.dim)

    # ---- ghi ------------------------------------------------------------- #

    def upsert_skills(self, records: list[SkillRecord]) -> int:
        """Thêm/cập nhật nhiều skill. Idempotent theo đường dẫn tuyệt đối."""
        if not records:
            return 0
        self.ensure_collection()

        vectors = self.embedder.embed_documents([_embed_text(r) for r in records])
        points = [
            models.PointStruct(id=record.point_id, vector=vector, payload=asdict(record))
            for record, vector in zip(records, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("[skill_index] đã upsert %d skill vào '%s'", len(points), self.collection)
        return len(points)

    def add_skill(self, skill_dir: Path | str) -> SkillRecord:
        """Thêm/cập nhật MỘT skill từ thư mục của nó (đọc `SKILL.md`)."""
        skill_dir = Path(skill_dir).resolve()
        if not (skill_dir / "SKILL.md").is_file():
            raise FileNotFoundError(f"Không thấy SKILL.md trong: {skill_dir}")

        props = read_properties(skill_dir)  # ném ParseError/ValidationError nếu hỏng
        if not props:
            raise ValueError(f"SKILL.md không có frontmatter hợp lệ: {skill_dir}")

        record = SkillRecord(
            name=props.name,
            path=str(skill_dir),
            description=props.description,
            group=skill_dir.parent.name,
        )
        self.upsert_skills([record])
        return record

    def remove_skill(self, skill_dir: Path | str) -> bool:
        skill_dir = Path(skill_dir).resolve()
        point_id = str(uuid.uuid5(_ID_NAMESPACE, str(skill_dir)))
        if not self.client.collection_exists(self.collection):
            return False
        self.client.delete(collection_name=self.collection,
                           points_selector=models.PointIdsList(points=[point_id]))
        logger.info("[skill_index] đã xoá skill khỏi index: %s", skill_dir)
        return True

    # ---- đọc ------------------------------------------------------------- #

    def search(self, query: str, top_k: int = 3, score_threshold: float | None = None) -> list[SkillHit]:
        """Top-K skill giống câu hỏi nhất (cosine similarity trên description)."""
        if not query or top_k <= 0 or not self.client.collection_exists(self.collection):
            return []

        response = self.client.query_points(
            collection_name=self.collection,
            query=self.embedder.embed_query(query),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        hits: list[SkillHit] = []
        for point in response.points:
            payload: dict[str, Any] = point.payload or {}
            hits.append(
                SkillHit(
                    name=payload.get("name", ""),
                    path=payload.get("path", ""),
                    description=payload.get("description", ""),
                    group=payload.get("group", ""),
                    score=float(point.score),
                ))
        return hits

    def list_all(self) -> list[SkillRecord]:
        if not self.client.collection_exists(self.collection):
            return []
        points, _ = self.client.scroll(collection_name=self.collection, limit=10_000, with_payload=True)
        return [
            SkillRecord(
                name=(p.payload or {}).get("name", ""),
                path=(p.payload or {}).get("path", ""),
                description=(p.payload or {}).get("description", ""),
                group=(p.payload or {}).get("group", ""),
            ) for p in points
        ]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(collection_name=self.collection).count

    def close(self) -> None:
        self.client.close()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_client(qdrant_url: str | None = None, qdrant_path: Path | str | None = None) -> QdrantClient:
    """Nối tới Qdrant server nếu có URL; nếu không, dùng bản NHÚNG lưu on-disk."""
    url = qdrant_url or os.getenv("QDRANT_URL")
    if url:
        logger.info("[skill_index] Qdrant server: %s", url)
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))

    path = Path(qdrant_path or os.getenv("SKILL_INDEX_PATH") or DEFAULT_INDEX_PATH)
    path.mkdir(parents=True, exist_ok=True)
    logger.info("[skill_index] Qdrant nhúng (on-disk): %s", path)
    return QdrantClient(path=str(path))


def open_index(
    *,
    collection: str | None = None,
    qdrant_url: str | None = None,
    qdrant_path: Path | str | None = None,
    embedder: Embedder | None = None,
    embedder_provider: str | None = None,
    embed_model: str | None = None,
) -> SkillIndex:
    """Mở (không tự build) index skill."""
    return SkillIndex(
        client=make_client(qdrant_url, qdrant_path),
        collection=collection or os.getenv("SKILL_INDEX_COLLECTION") or DEFAULT_COLLECTION,
        embedder=embedder or make_embedder(embedder_provider, embed_model),
    )


def build_index(
    skills_root: Path | str,
    *,
    recreate: bool = False,
    index: SkillIndex | None = None,
    **open_kwargs: Any,
) -> tuple[SkillIndex, list[SkillRecord]]:
    """Quét `skills_root` rồi nạp toàn bộ skill vào Qdrant. Trả về (index, records)."""
    index = index or open_index(**open_kwargs)
    records = discover_skills(skills_root)
    index.ensure_collection(recreate=recreate)
    index.upsert_skills(records)
    return index, records
