# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tool_passthrough — tool `send_package_details`: chạy script `package_details.py` và gửi kết
quả (list JSON) THẲNG tới client, KHÔNG nhét vào ngữ cảnh của agent.

Ý tưởng (đối xứng với `todo_event_stream`)
-------------------------------------------
Bình thường output của một tool quay về ngữ cảnh dưới dạng `ToolMessage` -> LLM đọc rồi VIẾT LẠI
thành câu trả lời. Với `package_details` ta KHÔNG muốn vậy: JSON gói cần đến client NGUYÊN VẸN để
UI tự render (thẻ gói), còn agent vẫn trả lời bằng chữ như hiện tại.

Cách làm (ĐẨY STEP TRỰC TIẾP — chắc ăn):
  1. Tool chạy `package_details.py <codes> --isdn <isdn>` -> stdout là MỘT list JSON.
  2. Tool ĐẨY một NAT intermediate step tên **`package_details_payload`** (KHÁC tên tool
     `send_package_details`) mang JSON đó -> front-end phát `intermediate_data:` -> bridge phát ra
     item type RIÊNG cho client. Đẩy trực tiếp trong Context của workflow (Context CHẮC CHẮN có mặt
     khi tool chạy — chính NAT profiler cũng ghi tool step trong context này).
  3. Kèm fallback: ghi kênh **custom** của LangGraph (`get_stream_writer`) để `stream_passthrough`
     đẩy step giúp nếu vì lý do nào đó bước (2) không chạy. Bridge dedup theo NỘI DUNG nên không đôi.
  4. Tool RETURN một câu ACK NGẮN (không phải JSON) -> ngữ cảnh agent SẠCH.

⚠️  Tên STEP (`package_details_payload`) phải KHÁC và KHÔNG là chuỗi con của tên TOOL
(`send_package_details`): nếu trùng/substring, bridge sẽ tóm nhầm step "lời gọi tool" (Input = args
`{'package_codes':...}`) thay vì step JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Tên STEP mang JSON (đi tới bridge). KHÁC tên tool `send_package_details` (tránh khớp nhầm substring).
SCRIPT_STEP_NAME = "package_details_payload"
# `name` client-facing gắn vào item (để client biết là gói cước).
SCRIPT_OUTPUT_NAME = "package_details"


def _push_script_step(name: str, payload_text: str) -> bool:
    """Đẩy cặp TOOL_START + TOOL_END mang `payload_text` vào Context workflow hiện tại.

    Front-end NAT phát `intermediate_data:` với name = "Tool: <name>" và payload chứa
    `**Input:**`/`**Output:**` = `payload_text`. Trả True nếu đẩy được (đang trong Context), False
    nếu không (vd chạy standalone / `/generate`) -> caller degrade êm.
    """
    try:
        from nat.builder.context import Context  # lazy: chỉ có ý nghĩa khi chạy trong NAT
        from nat.data_models.intermediate_step import (
            IntermediateStepPayload,
            IntermediateStepType,
            StreamEventData,
        )
    except Exception:  # pragma: no cover
        return False
    try:
        mgr = Context.get().intermediate_step_manager
    except Exception:  # pragma: no cover - không có Context
        return False

    uid = f"script-{uuid.uuid4().hex[:12]}"
    try:
        mgr.push_intermediate_step(IntermediateStepPayload(
            event_type=IntermediateStepType.TOOL_START, name=name, UUID=uid,
            data=StreamEventData(input=payload_text)))
        mgr.push_intermediate_step(IntermediateStepPayload(
            event_type=IntermediateStepType.TOOL_END, name=name, UUID=uid,
            data=StreamEventData(input=payload_text, output=payload_text)))
        return True
    except Exception:  # pragma: no cover
        logger.exception("[tool_passthrough] không đẩy được intermediate step '%s'", name)
        return False


def _emit_to_client_custom(payload_text: str, *, name: str, meta: dict[str, Any] | None = None) -> bool:
    """(Fallback) Ghi `payload_text` qua kênh stream custom của LangGraph -> stream_passthrough đẩy step.

    Chỉ có tác dụng khi đang chạy trong một `graph.astream` có bật stream_mode 'custom'.
    """
    try:
        from langgraph.config import get_stream_writer  # lazy
    except Exception:  # pragma: no cover
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
    async def send_package_details(package_codes: str, isdn: str = "") -> str:
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
        # ⚠️  Tool PHẢI là async: hàm sync bị LangGraph chạy trong THREAD EXECUTOR -> contextvar của
        #     NAT (Context.get()) KHÔNG truyền vào thread -> _push_script_step thất bại âm thầm.
        #     async -> chạy thẳng trong async context của workflow (như wrapper) -> push hoạt động.
        codes = [c.strip() for c in str(package_codes).replace(" ", ",").split(",") if c.strip()]
        if not codes:
            return "Không có mã gói nào để gửi. Hãy truyền `package_codes` (vd 'DRE20,SD90')."

        # subprocess blocking -> đẩy sang thread; nhưng phần PUSH ở dưới chạy trong async context.
        out, err = await asyncio.to_thread(
            _run_package_details, script, codes, str(isdn or "").strip(), default_timeout)
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

        meta = {"codes": codes, "isdn": isdn or None}
        sent_step = _push_script_step(SCRIPT_STEP_NAME, out)                   # PRIMARY: đẩy step trực tiếp
        sent_custom = _emit_to_client_custom(out, name=SCRIPT_STEP_NAME, meta=meta)  # fallback qua wrapper
        logger.info("[tool_passthrough] send_package_details: %s gói (%d ký tự JSON) | "
                    "push_step=%s | custom=%s | step_name=%s",
                    n, len(out), sent_step, sent_custom, SCRIPT_STEP_NAME)
        if not (sent_step or sent_custom):
            logger.warning("[tool_passthrough] KHÔNG gửi được payload tới client (ngoài ngữ cảnh "
                           "stream/Context) -> item ui_package_list sẽ KHÔNG xuất hiện.")
            return ("Đã lấy được chi tiết gói nhưng CHƯA gửi được lên màn hình. Cứ tư vấn bằng lời.")

        count = f"{n} gói" if n is not None else "chi tiết gói"
        return (f"✅ Đã gửi {count} tới màn hình người dùng để hiển thị riêng. "
                "Tiếp tục viết câu tư vấn bằng lời, KHÔNG dán lại JSON.")

    return send_package_details
