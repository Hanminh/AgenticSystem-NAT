#!/usr/bin/env python
"""CLI quản lý index skill trên Qdrant (semantic search).

Chạy từ thư mục gốc `AgenticSkills/` để `agenticskills` và `skills_ref` import được.

    # 1) Build index từ một thư mục gốc chứa nhiều folder skill (quét đệ quy)
    python scripts/skill_index_cli.py build --skills-root skills

    # Build lại từ đầu (xoá collection cũ) — dùng khi đổi model embedding
    python scripts/skill_index_cli.py build --skills-root skills --recreate

    # 2) THÊM MỘT SKILL MỚI vào index (không cần build lại toàn bộ)
    python scripts/skill_index_cli.py add skills/Telecom_Skills/dien-giai-cuoc-skill-optimize

    # 3) Thử tìm kiếm xem top-K trả về gì
    python scripts/skill_index_cli.py search "cước tháng này của tôi bao nhiêu?" --top-k 3

    # 4) Xem / xoá
    python scripts/skill_index_cli.py list
    python scripts/skill_index_cli.py remove skills/Telecom_Skills/dien-giai-cuoc-skill

Qdrant chạy ở đâu: có `QDRANT_URL` thì nối server, không thì dùng bản NHÚNG on-disk
tại `SKILL_INDEX_PATH` (mặc định `<AgenticSkills>/.qdrant_skills`).
LƯU Ý: bản nhúng KHOÁ thư mục — không chạy CLI này trong lúc `nat serve` đang mở
cùng index. Muốn dùng song song thì dựng Qdrant server và set `QDRANT_URL`.
"""

import argparse
import logging
import sys
from pathlib import Path

# Cho phép chạy trực tiếp từ gốc project mà không cần cài package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from agenticskills.skill_index import (  # noqa: E402
    build_index,
    discover_skills,
    open_index,
)

load_dotenv()


def _index(args: argparse.Namespace):
    return open_index(
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        embedder_provider=args.embedder,
        embed_model=args.embed_model,
    )


def cmd_build(args: argparse.Namespace) -> int:
    root = Path(args.skills_root).resolve()
    if not root.is_dir():
        print(f"❌ Không thấy thư mục skill: {root}")
        return 1

    found = discover_skills(root)
    if not found:
        print(f"❌ Không tìm thấy SKILL.md nào dưới {root}")
        return 1

    index, records = build_index(root, recreate=args.recreate, index=_index(args))
    print(f"✅ Đã index {len(records)} skill vào collection '{index.collection}'"
          f" (embedder: {index.embedder.model_name}, dim={index.embedder.dim})")
    for record in records:
        print(f"   - {record.group + '/' if record.group else ''}{record.name}\n     {record.path}")
    index.close()
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    index = _index(args)
    try:
        record = index.add_skill(args.skill_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}")
        return 1
    print(f"✅ Đã thêm/cập nhật skill '{record.name}' trong '{index.collection}'")
    print(f"   path       : {record.path}")
    print(f"   description: {record.description[:160]}{'...' if len(record.description) > 160 else ''}")
    print(f"   tổng số skill trong index: {index.count()}")
    index.close()
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    index = _index(args)
    index.remove_skill(args.skill_dir)
    print(f"✅ Đã xoá khỏi index: {Path(args.skill_dir).resolve()}")
    print(f"   tổng số skill còn lại: {index.count()}")
    index.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index = _index(args)
    hits = index.search(args.query, top_k=args.top_k, score_threshold=args.score_threshold)
    if not hits:
        print("(không có skill nào khớp)")
        index.close()
        return 0

    print(f"Top {len(hits)} skill cho: {args.query!r}\n")
    for rank, hit in enumerate(hits, 1):
        print(f"{rank}. [{hit.score:.4f}] {hit.name}")
        print(f"   {hit.path}")
        print(f"   {hit.description[:200]}{'...' if len(hit.description) > 200 else ''}\n")
    index.close()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    index = _index(args)
    records = index.list_all()
    print(f"Collection '{index.collection}': {len(records)} skill")
    for record in sorted(records, key=lambda r: (r.group, r.name)):
        print(f"  - {record.group + '/' if record.group else ''}{record.name}  ->  {record.path}")
    index.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quản lý index skill trên Qdrant (semantic search).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="In log chi tiết")

    # Cấu hình Qdrant/embedder dùng chung cho mọi subcommand.
    parser.add_argument("--collection", help="Tên collection (mặc định: env SKILL_INDEX_COLLECTION / 'agentic_skills')")
    parser.add_argument("--qdrant-url", help="URL Qdrant server (mặc định: env QDRANT_URL; không có -> bản nhúng)")
    parser.add_argument("--qdrant-path", help="Thư mục Qdrant nhúng (mặc định: env SKILL_INDEX_PATH)")
    parser.add_argument("--embedder", choices=["fastembed", "openai"],
                        help="Provider embedding (mặc định: env SKILL_INDEX_EMBEDDER / 'fastembed')")
    parser.add_argument("--embed-model", help="Tên model embedding")

    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Quét thư mục gốc và index TOÀN BỘ skill")
    p_build.add_argument("--skills-root", default="skills", help="Thư mục gốc chứa các folder skill (mặc định: skills)")
    p_build.add_argument("--recreate", action="store_true", help="Xoá collection cũ rồi tạo lại")
    p_build.set_defaults(func=cmd_build)

    p_add = sub.add_parser("add", help="Thêm/cập nhật MỘT skill mới vào index")
    p_add.add_argument("skill_dir", help="Đường dẫn tới folder skill (chứa SKILL.md)")
    p_add.set_defaults(func=cmd_add)

    p_remove = sub.add_parser("remove", help="Xoá một skill khỏi index")
    p_remove.add_argument("skill_dir", help="Đường dẫn tới folder skill")
    p_remove.set_defaults(func=cmd_remove)

    p_search = sub.add_parser("search", help="Thử similarity search")
    p_search.add_argument("query", help="Câu hỏi của người dùng")
    p_search.add_argument("--top-k", type=int, default=3)
    p_search.add_argument("--score-threshold", type=float, default=None)
    p_search.set_defaults(func=cmd_search)

    p_list = sub.add_parser("list", help="Liệt kê skill đang có trong index")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
