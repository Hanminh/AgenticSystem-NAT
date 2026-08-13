# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tool_passthrough — tool `send_package_details`: chạy script `package_details.py` và gửi kết
quả (list JSON) THẲNG tới client, KHÔNG nhét vào ngữ cảnh của agent.

Ý tưởng (đối xứng với `todo_event_stream`)
-------------------------------------------
Bình thường output của một tool quay về ngữ cảnh dưới dạng `ToolMessage` -> LLM đọc rồi VIẾT LẠI
thành câu trả lời. Với `package_details` ta KHÔNG muốn vậy: JSON gói cần đến client NGUYÊN VẸN để
UI tự render (thẻ gói), còn agent vẫn trả lời bằng chữ như hiện tại.

Cách làm:
  1. Tool chạy `package_details.py <codes> --isdn <isdn>` -> stdout là MỘT list JSON.
  2. Đẩy JSON đó qua kênh **custom** của LangGraph: `get_stream_writer()({"script_output": {...}})`.
     `stream_passthrough.PassthroughEventStreamGraph` đọc kênh custom này và biến nó thành MỘT NAT
     intermediate step tên `package_details` -> bridge nhận và phát ra item type RIÊNG cho client.
  3. Tool RETURN một câu ACK NGẮN (không phải JSON) -> ngữ cảnh agent SẠCH, agent viết tư vấn từ
     dữ liệu đã lấy ở bước `parallel_api_calling` trước đó.

Chỉ dùng ở HAI trường hợp (điều khiển bằng prompt trong SKILL.md, không phải ở code):
  * sau khi agent ĐỀ XUẤT gói -> gửi chi tiết CÁC gói vừa đề xuất, hoặc
  * người dùng yêu cầu xem MỘT gói cụ thể.
KHÔNG gọi tuỳ tiện ngoài hai ca này.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Tên "script" gắn vào payload custom + NAT step -> client key theo tên này để biết là gói cước.
SCRIPT_OUTPUT_NAME = "package_details"


def _emit_to_client(payload_text: str, *, name: str, meta: dict[str, Any] | None = None) -> bool:
    """Đẩy `payload_text` qua kênh stream custom của LangGraph (nếu đang chạy trong graph).

    Trả True nếu ghi được (đang trong một `graph.astream`), False nếu không (vd chạy standalone /
    `/generate` không stream) -> caller degrade êm, chỉ trả ack.
    """
    try:
        from langgraph.config import get_stream_writer  # lazy: chỉ có ý nghĩa khi chạy trong graph
    except Exception:  # pragma: no cover - langgraph quá cũ
        return False
    try:
        writer = get_stream_writer()
    except Exception:  # pragma: no cover - ngoài ngữ cảnh graph
        return False
    if writer is None:
        return False
    try:
        writer({"script_output": {"name": name, "text": payload_text, "meta": meta or {}}})
        return True
    except Exception:  # pragma: no cover
        logger.exception("[tool_passthrough] không ghi được stream_writer cho '%s'", name)
        return False


def _run_package_details(script_path: Path, codes: list[str], isdn: str, timeout: int) -> tuple[str, str | None]:
    """Chạy `package_details.py` với danh sách mã. Trả (stdout_json, error_message)."""
    cmd = [sys.executable, str(script_path), *codes]
    if isdn:
        cmd += ["--isdn", isdn]
    cmd += ["--timeout", str(timeout)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        return "", f"timeout sau {timeout + 15}s khi chạy package_details.py"
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 and not out:
        return "", (proc.stderr or "").strip() or f"exit code {proc.returncode}"
    return out, None


def make_package_details_tool(script_path: Path | str, *, default_timeout: int = 20):
    """Tạo tool `send_package_details` gắn với ĐƯỜNG DẪN tuyệt đối của `package_details.py`.

    Builder truyền sẵn `script_path` (resolve từ skills_dir) nên model KHÔNG cần biết path — chỉ
    truyền danh sách mã gói + (tuỳ chọn) số thuê bao.
    """
    script = Path(script_path)

    @tool("send_package_details")
    def send_package_details(package_codes: str, isdn: str = "") -> str:
        """Gửi CHI TIẾT các gói cước tới MÀN HÌNH người dùng để hiển thị dạng thẻ.

        CHỈ gọi ở HAI trường hợp:
          1. Ngay sau khi bạn ĐỀ XUẤT gói -> truyền các mã gói bạn vừa đề xuất.
          2. Người dùng yêu cầu xem MỘT gói cụ thể -> truyền mã gói đó.
        KHÔNG gọi ở các trường hợp khác.

        Args:
            package_codes: các mã gói cách nhau bởi dấu phẩy, vd "DRE20,SD90,30N".
            isdn: số thuê bao (tuỳ chọn) — truyền nếu đã biết trong hội thoại.

        Returns:
            Một câu xác nhận NGẮN. Kết quả chi tiết đã được gửi thẳng tới màn hình người dùng,
            BẠN KHÔNG cần dán lại JSON vào câu trả lời.
        """
        codes = [c.strip() for c in str(package_codes).replace(" ", ",").split(",") if c.strip()]
        if not codes:
            return "Không có mã gói nào để gửi. Hãy truyền `package_codes` (vd 'DRE20,SD90')."

        out, err = _run_package_details(script, codes, str(isdn or "").strip(), default_timeout)
        if err:
            logger.warning("[tool_passthrough] package_details lỗi: %s", err)
            return (f"Không lấy được chi tiết gói ({err}). Cứ tư vấn cho người dùng bằng dữ liệu "
                    "đã có, không cần chi tiết bổ sung.")

        # Xác thực JSON nhẹ (không chặn nếu script đổi định dạng — vẫn gửi text thô).
        n = None
        try:
            parsed = json.loads(out)
            n = len(parsed) if isinstance(parsed, list) else None
        except Exception:  # noqa: BLE001
            pass

        sent = _emit_to_client(out, name=SCRIPT_OUTPUT_NAME, meta={"codes": codes, "isdn": isdn or None})
        if not sent:
            logger.info("[tool_passthrough] ngoài ngữ cảnh stream (vd /generate) -> chỉ trả ack, "
                        "không gửi được payload tới client.")

        count = f"{n} gói" if n is not None else "chi tiết gói"
        return (f"✅ Đã gửi {count} tới màn hình người dùng để hiển thị riêng. "
                "Tiếp tục viết câu tư vấn bằng lời, KHÔNG dán lại JSON.")

    return send_package_details
