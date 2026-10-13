"""
Alice - Discord bot (discord.py + Gemini + Vector Memory MongoDB/SQLite + Global Memory)

Biến môi trường:
  Bắt buộc     : DISCORD_TOKEN, GEMINI_API_KEY
  Khuyến nghị  : OWNER_ID (ID số Discord của chủ nhân; bỏ trống thì dùng chủ sở hữu ứng dụng Discord),
                 MONGO_URI (lưu ký ức vĩnh viễn trên MongoDB Atlas)
  Nhân vật     : PERSONA_MAX_CHARS (1500), PERSONA_CACHE_TTL_SEC (300) — chỉ dẫn riêng theo USER ID, quản lý bằng /persona_*,
                 PROMPT_OPERATIONAL (0 = giữ prompt rút gọn như hiện tại; 1 = khôi phục mục
                 <operational_guidelines> và dòng "Điểm kết" của prompt gốc)
  Ký ức cá nhân: MEMORY_WINDOW_DAYS (30; 0 = không giới hạn), MEMORY_RETENTION_DAYS (0 = không tự xóa),
                 MEMORY_THRESHOLD (cosine thật, mặc định 0.55), EMBEDDING_MODEL, EMBEDDING_DIM (0 = mặc định model)
  Global Memory: GLOBAL_TOP_K (3), GLOBAL_MEMORY_THRESHOLD (mặc định = MEMORY_THRESHOLD), GLOBAL_CHUNK_CHARS (900),
                 GLOBAL_CHUNK_OVERLAP (100), GLOBAL_MAX_FILE_BYTES (300000), GLOBAL_MAX_CHUNKS (400 mỗi file),
                 GLOBAL_MAX_TOTAL_CHUNKS (2000 tổng cộng), GLOBAL_EMBED_BATCH (16), GLOBAL_CACHE_TTL_SEC (300)
  Gemini       : GEMINI_MODELS (cách nhau bằng dấu phẩy), GEMINI_TIMEOUT, MAX_CONCURRENT_AI
  Chống ban IP : USER_COOLDOWN_SEC, GLOBAL_MAX_REPLIES + GLOBAL_WINDOW_SEC (trần số lượt trả lời của cả bot —
                 tên giữ nguyên như các bản trước), CHANNEL_MUTE_SEC, GLOBAL_BACKOFF_SEC,
                 LOGIN_BACKOFF_SEC (mặc định 900), DISCORD_PROXY (vd http://user:pass@host:port)
  Khác         : VECTOR_INDEX_NAME, SQLITE_DB_PATH, PORT

Lệnh chủ nhân: /globalmem_add|list|delete|clear|test, /persona_set|list|show|delete, /sync
"""
import os
import re
import math
import json
import time
import asyncio
import logging
import sqlite3
import uuid
import threading
from array import array
from collections import OrderedDict, deque
from contextlib import asynccontextmanager, closing
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

log = logging.getLogger("alice")


# ==============================================================================
# CẤU HÌNH
# ==============================================================================
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "yukari_memories.db")

# ID số Discord của chủ nhân (thay cho việc so khớp display_name vốn dễ bị giả mạo)
OWNER_ID = _env_int("OWNER_ID", 0)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
# 0 = dùng số chiều mặc định của model (không phá vector cũ). Đặt 768 để nhẹ hơn ~4 lần,
# nhưng khi đó phải tạo lại Atlas index (numDimensions) và vector cũ sẽ không còn so khớp được.
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 0)

_DEFAULT_MODELS = "gemini-3.8-flash,gemini-3.1-flash-lite,gemini-3.6-flash,gemini-3.5-flash-lite"
GEMINI_MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", _DEFAULT_MODELS).split(",") if m.strip()]
_TOTAL_MODELS = int(len(GEMINI_MODELS) - 1)
if not GEMINI_MODELS:
    GEMINI_MODELS = _DEFAULT_MODELS.split(",")
GEMINI_TIMEOUT = _env_float("GEMINI_TIMEOUT", 35.0)

MEMORY_WINDOW_DAYS = _env_int("MEMORY_WINDOW_DAYS", 30)          # chỉ nhớ lại ký ức trong N ngày gần nhất
MEMORY_RETENTION_DAYS = _env_int("MEMORY_RETENTION_DAYS", 0)     # tự xóa ký ức cũ hơn N ngày (0 = tắt)
MEMORY_THRESHOLD = _env_float("MEMORY_THRESHOLD", 0.55)          # ngưỡng cosine THẬT (-1..1)
MEMORY_TOP_K = 3
MEMORY_MIN_CHARS = 12            # tin nhắn của người dùng ngắn hơn mức này sẽ không được lưu
MEMORY_REPLY_MAX_CHARS = 1500    # phần trả lời của Alice lưu vào ký ức tối đa N ký tự
MEMORY_SCAN_LIMIT = 100          # số bản ghi gần nhất quét cosine thủ công

MAX_INPUT_CHARS = 1500
HISTORY_MAX_LINES = 8
HISTORY_PROMPT_LINES = 6
HISTORY_MAX_KEYS = 500
DISCORD_MSG_LIMIT = 1950

USER_COOLDOWN_SEC = _env_float("USER_COOLDOWN_SEC", 2.0)
MAX_CONCURRENT_AI = _env_int("MAX_CONCURRENT_AI", 4)

# Chống spam request tới Discord (tránh bị Cloudflare ban IP). Tên biến môi trường giữ nguyên như các bản trước.
RATE_MAX_REPLIES = _env_int("GLOBAL_MAX_REPLIES", 30)       # tối đa N lượt trả lời trong RATE_WINDOW_SEC giây (cả bot)
RATE_WINDOW_SEC = _env_int("GLOBAL_WINDOW_SEC", 60)
CHANNEL_MUTE_SEC = _env_int("CHANNEL_MUTE_SEC", 600)        # kênh gửi bị 403 -> im lặng N giây
RATE_BACKOFF_SEC = _env_int("GLOBAL_BACKOFF_SEC", 600)      # gặp 429/401/Cloudflare -> dừng mọi phản hồi N giây
LOGIN_BACKOFF_SEC = _env_int("LOGIN_BACKOFF_SEC", 900)      # đăng nhập bị 429/token sai -> chờ N giây rồi mới thoát
DISCORD_PROXY = os.getenv("DISCORD_PROXY") or None          # tùy chọn: HTTP proxy cho cả REST lẫn gateway

# Nhân vật: chỉ dẫn riêng theo USER ID (tên hiển thị ai cũng đổi được nên không đáng tin). Quản lý bằng /persona_*.
PERSONA_MAX_CHARS = _env_int("PERSONA_MAX_CHARS", 1500)
PERSONA_CACHE_TTL_SEC = _env_int("PERSONA_CACHE_TTL_SEC", 300)
PROMPT_OPERATIONAL = _env_int("PROMPT_OPERATIONAL", 0) != 0
# Mặc định tích hợp sẵn (thay cho GRIM_ID cũ): {user_id: (tên gọi, chỉ dẫn)}.
# /persona_set có thể ghi đè; /persona_delete đưa về đúng mặc định này.
BUILTIN_PERSONAS = {
    "1082184769695133716": (
        "Cha",
        "Người nói CHÍNH LÀ Cha Namaru sao — người mà Yukari dành tình cảm đặc biệt. Xưng con gọi cha và đối đãi "
        "với cha ấy đúng như phần mô tả về Namaru trong mục core_dynamics ở trên.",
    ),
}

# Global Memory: tri thức chung do chủ nhân nạp bằng file, dùng với MỌI người dùng/kênh/thời điểm
GLOBAL_USER_ID = "__global__"          # "user_id" đặc biệt (không bao giờ trùng ID số của Discord)
GLOBAL_CHANNEL_ID = "__global__"
LEGACY_GLOBAL_USER_ID = "GLOBAL"       # dữ liệu do bản cũ (/inject_file) tạo ra — vẫn đọc/xóa/thay thế được
LEGACY_SOURCE_PREFIX = "Knowledge:"
GLOBAL_TOP_K = _env_int("GLOBAL_TOP_K", 3)
GLOBAL_THRESHOLD = _env_float("GLOBAL_MEMORY_THRESHOLD", MEMORY_THRESHOLD)
GLOBAL_CHUNK_CHARS = _env_int("GLOBAL_CHUNK_CHARS", 900)         # độ dài tối đa mỗi đoạn (ký tự)
GLOBAL_CHUNK_OVERLAP = _env_int("GLOBAL_CHUNK_OVERLAP", 100)     # phần gối đầu giữa hai đoạn liên tiếp
GLOBAL_MAX_FILE_BYTES = _env_int("GLOBAL_MAX_FILE_BYTES", 300_000)
GLOBAL_MAX_CHUNKS = _env_int("GLOBAL_MAX_CHUNKS", 400)           # tối đa số đoạn của MỘT file
GLOBAL_MAX_TOTAL_CHUNKS = _env_int("GLOBAL_MAX_TOTAL_CHUNKS", 2000)  # trần tổng số đoạn (bảo vệ RAM)
GLOBAL_EMBED_BATCH = _env_int("GLOBAL_EMBED_BATCH", 16)          # số đoạn nhúng trong một lần gọi API
GLOBAL_CACHE_TTL_SEC = _env_int("GLOBAL_CACHE_TTL_SEC", 300)     # bao lâu kiểm tra lại "chữ ký" dữ liệu Global Memory
GLOBAL_UPLOAD_DEADLINE_SEC = 780                                 # token tương tác Discord hết hạn sau 15 phút
GLOBAL_ALLOWED_EXT = (".txt", ".md", ".markdown", ".csv", ".json", ".log")
EMBED_RETRY_BASE_SEC = 5
EMBED_QUOTA_WAIT_SEC = 30
ATLAS_RETRY_SEC = 600            # sau khi Atlas $vectorSearch lỗi, nghỉ N giây rồi mới thử lại

# Câu trả lời dự phòng (giữ đúng giọng Alice)
MSG_EMPTY = "*ngáp dài một hơi, phe phẩy quạt xếp che nửa miệng*\nGiấc mộng vừa rồi trống rỗng quá... Ngươi định nói gì với ta thế?"
MSG_SAFETY = "*mắt đỏ trong khe nứt khẽ chớp, nụ cười nhạt hiện lên sau vành dù ren*\nRanh giới ấy không cho phép vượt qua đâu."
MSG_QUOTA = "Cuộc trò chuyện của ngươi giữa tạm thời hoãn tại đây nhé... Khoảng không gian này đang cạn kiệt năng lượng rồi. (Hết hạn mức API)"
MSG_QUOTA_LONG = "Ta thấy hơi mệt và muốn ngủ đông một chút đây... (Hết hạn mức API Google). Chắc phải mất kha khá thời gian để mở lại khe nứt đấy."
MSG_ERROR = "*lục lọi trong sukima một hồi nhưng chỉ thấy toàn gối ngủ*\n...Đợi ta nghĩ ngợi chút... hmmmm."

# Trạng thái lưu trữ
use_mongo = False
memories_collection = None
custom_prompts_collection = None   # collection nhỏ lưu chỉ dẫn riêng theo người dùng
_atlas_disabled_until = 0.0      # time.monotonic() — trước mốc này thì bỏ qua Atlas $vectorSearch

ai: Optional[genai.Client] = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ==============================================================================
# TOÁN HỌC & TIỆN ÍCH
# ==============================================================================
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính độ tương đồng cosin giữa 2 vector."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


_UNSAFE_NAME_RE = re.compile(r"[\x00-\x1f\x7f<>\[\]{}`]")


def sanitize_name(name: Optional[str], limit: int = 32) -> str:
    """Làm sạch display_name trước khi đưa vào prompt/embed (chống prompt injection & phá định dạng)."""
    cleaned = _UNSAFE_NAME_RE.sub(" ", name or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit] or "Người lạ"


def truncate_for_discord(text: str, limit: int = DISCORD_MSG_LIMIT) -> str:
    """Cắt gọn tin nhắn ở ranh giới câu/dòng và đóng dấu * nếu bị lẻ (Yukari dùng *hành động*)."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    idx = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if idx > limit * 0.6:
        cut = cut[: idx + 1]
    if cut.count("*") % 2 == 1:
        cut += "*"
    return cut.rstrip() + "…"


_bg_tasks: set = set()


def spawn(coro) -> asyncio.Task:
    """create_task có giữ tham chiếu — tránh việc task nền bị garbage-collect giữa chừng."""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


# ==============================================================================
# DATABASE: SQLITE (DỰ PHÒNG) + MONGODB ATLAS
# ==============================================================================
def _sqlite_connect() -> sqlite3.Connection:
    return sqlite3.connect(SQLITE_DB_PATH, timeout=10)


def _ts_sqlite(dt: datetime) -> str:
    """Định dạng thời gian cho SQLite — trùng định dạng adapter cũ nên so sánh chuỗi vẫn đúng thứ tự."""
    return dt.isoformat(sep=" ", timespec="microseconds")


def init_sqlite():
    """Khởi tạo SQLite cục bộ dự phòng lưu vector embedding nếu MongoDB không khả dụng."""
    try:
        with closing(_sqlite_connect()) as conn:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vector_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        channel_id TEXT,
                        author_name TEXT,
                        text TEXT,
                        embedding TEXT,
                        timestamp TEXT
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_vm_user_channel ON vector_memories(user_id, channel_id, id)"
                )
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS global_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        upload_id TEXT NOT NULL,
                        chunk_index INTEGER,
                        text TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        created_at TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_gm_source ON global_memories(source)")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS custom_prompts (
                        user_id TEXT PRIMARY KEY,
                        label TEXT,
                        prompt TEXT NOT NULL,
                        updated_at TEXT
                    )
                """)
                if MEMORY_RETENTION_DAYS > 0:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=MEMORY_RETENTION_DAYS)
                    conn.execute(
                        "DELETE FROM vector_memories WHERE user_id NOT IN (?, ?) AND timestamp < ?",
                        (GLOBAL_USER_ID, LEGACY_GLOBAL_USER_ID, _ts_sqlite(cutoff)),
                    )
        log.info("📁 Đã khởi tạo cơ sở dữ liệu SQLite dự phòng sẵn sàng.")
    except Exception as e:
        log.warning("⚠️ Lỗi khởi tạo SQLite: %s", e)


def init_mongo():
    """Kết nối MongoDB Atlas (nếu có MONGO_URI) và tạo index cần thiết."""
    global use_mongo, memories_collection, custom_prompts_collection
    if not MONGO_URI:
        return
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        memories_collection = client["yukari_db"]["vector_memories"]
        custom_prompts_collection = client["yukari_db"]["custom_prompts"]
        use_mongo = True
        log.info("✅ Đã kết nối thành công MongoDB Atlas! Ký ức Vector sẽ được lưu vĩnh viễn trên đám mây.")
    except Exception as e:
        log.warning("⚠️ Không thể kết nối MongoDB (%s), chuyển sang chế độ SQLite cục bộ.", e)
        use_mongo = False
        return

    try:
        memories_collection.create_index([("user_id", 1), ("channel_id", 1), ("timestamp", -1)])
        if MEMORY_RETENTION_DAYS > 0:
            memories_collection.create_index(
                "timestamp",
                expireAfterSeconds=MEMORY_RETENTION_DAYS * 86400,
                name="ttl_timestamp",
            )
    except Exception as e:
        log.warning("ℹ️ Không tạo được index MongoDB (có thể đã tồn tại với tùy chọn khác): %s", e)


# ==============================================================================
# GEMINI API: GENERATION & EMBEDDING
# ==============================================================================
_ai_sem: Optional[asyncio.Semaphore] = None


def _get_ai_sem() -> asyncio.Semaphore:
    """Tạo semaphore lười (lazy) để gắn đúng event loop đang chạy."""
    global _ai_sem
    if _ai_sem is None:
        _ai_sem = asyncio.Semaphore(max(1, MAX_CONCURRENT_AI))
    return _ai_sem


def _classify_gemini_error(e: Exception) -> str:
    """Trả về 'quota' | 'missing' | 'retry' | 'other'."""
    code = getattr(e, "code", None)
    text = str(e).upper()
    if code == 429 or "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "quota"
    if code == 404 or "404" in text or "NOT_FOUND" in text:
        return "missing"
    if code in (500, 502, 503, 504) or "503" in text or "UNAVAILABLE" in text or "DEADLINE" in text:
        return "retry"
    return "other"


def _extract_text(resp) -> tuple[Optional[str], bool]:
    """Lấy text từ phản hồi Gemini. Trả về (text, bị_chặn_an_toàn).
    Lưu ý: SDK mới KHÔNG ném exception khi bị chặn — resp.text chỉ là None."""
    if resp is None:
        return None, False
    try:
        text = resp.text
    except Exception:
        text = None
    if text and text.strip():
        return text.strip(), False

    blocked = False
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        blocked = True
    for cand in (getattr(resp, "candidates", None) or []):
        reason = str(getattr(cand, "finish_reason", "") or "").upper()
        if any(k in reason for k in ("SAFETY", "PROHIBITED", "BLOCKLIST", "SPII")):
            blocked = True
    return None, blocked


async def ask_gemini(contents, system_instruction, temperature=0.9) -> tuple[str, bool]:
    """Gọi Gemini với danh sách model dự phòng. Trả về (câu_trả_lời, thành_công).
    thành_công=False nghĩa là đây chỉ là câu dự phòng — KHÔNG nên lưu vào lịch sử/ký ức."""
    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=temperature)
    async with _get_ai_sem():
        for model_name in GEMINI_MODELS:
            for attempt in range(_TOTAL_MODELS):
                try:
                    resp = await asyncio.wait_for(
                        ai.aio.models.generate_content(model=model_name, contents=contents, config=config),
                        timeout=GEMINI_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    log.warning("Model %s quá thời gian chờ (%ss), chuyển model.", model_name, GEMINI_TIMEOUT)
                    break
                except Exception as e:
                    kind = _classify_gemini_error(e)
                    log.warning("Model %s lỗi (%s): %s", model_name, kind, str(e)[:150])
                    if kind == "retry" and attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    break  # quota / không tồn tại / lỗi khác -> thử model tiếp theo

                text, blocked = _extract_text(resp)
                if text:
                    return text, True
                return (MSG_SAFETY if blocked else MSG_EMPTY), False
    return MSG_QUOTA, False


async def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Tạo vector embedding bất đồng bộ từ text."""
    if not text or not text.strip():
        return []
    try:
        cfg = {"task_type": task_type}
        if EMBEDDING_DIM > 0:
            cfg["output_dimensionality"] = EMBEDDING_DIM
        resp = await asyncio.wait_for(
            ai.aio.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(**cfg),
            ),
            timeout=GEMINI_TIMEOUT,
        )
        if resp and resp.embeddings:
            return list(resp.embeddings[0].values)
    except Exception as e:
        log.warning("⚠️ Lỗi tạo embedding (%s): %s", task_type, e)
    return []


# ==============================================================================
# QUẢN LÝ VECTOR STORE (MONGODB ATLAS + FALLBACK SQLITE)
# ==============================================================================
async def store_context_memory(user_id: str, channel_id: str, author_name: str, user_text: str, reply_text: str):
    """Embedding một lượt hội thoại và lưu vào Vector Store (MongoDB hoặc SQLite).
    Bộ lọc độ dài/lệnh được áp dụng lên tin nhắn GỐC của người dùng (không phải cả đoạn đã ghép)."""
    user_clean = user_text.strip()
    if len(user_clean) < MEMORY_MIN_CHARS or user_clean.startswith(("!", "/")):
        log.info("💾 Bỏ qua lưu ký ức: tin nhắn quá ngắn hoặc là câu lệnh.")
        return

    reply_clean = reply_text.strip()
    if len(reply_clean) > MEMORY_REPLY_MAX_CHARS:
        reply_clean = reply_clean[:MEMORY_REPLY_MAX_CHARS] + "..."
    memory_text = f"{author_name}: {user_clean}\nAlice: {reply_clean}"

    vector = await get_embedding(memory_text, task_type="RETRIEVAL_DOCUMENT")
    if not vector:
        log.warning("💾 Không tạo được embedding nên chưa lưu ký ức lượt này (xem dòng lỗi embedding phía trên).")
        return

    now = datetime.now(timezone.utc)

    if use_mongo and memories_collection is not None:
        record = {
            "user_id": str(user_id),
            "channel_id": str(channel_id),
            "author_name": author_name,
            "text": memory_text,
            "embedding": vector,
            "timestamp": now,  # kiểu Date của BSON — truy vấn cũng phải dùng datetime
        }
        try:
            await asyncio.to_thread(memories_collection.insert_one, record)
            log.info("💾 Đã lưu ký ức vào MongoDB (vector %d chiều).", len(vector))
            return
        except Exception as e:
            log.warning("⚠️ Lỗi ghi MongoDB: %s, chuyển lưu vào SQLite dự phòng.", e)

    def _save_sqlite():
        with closing(_sqlite_connect()) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO vector_memories (user_id, channel_id, author_name, text, embedding, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(user_id), str(channel_id), author_name, memory_text,
                     json.dumps(vector, separators=(",", ":")), _ts_sqlite(now)),
                )

    try:
        await asyncio.to_thread(_save_sqlite)
        log.info("💾 Đã lưu ký ức vào SQLite cục bộ (vector %d chiều).", len(vector))
    except Exception as e:
        log.warning("⚠️ Lỗi ghi SQLite: %s", e)


def _rank_by_cosine(query_vector, rows, top_k: int, threshold: float) -> list[str]:
    """rows: iterable các cặp (text, embedding). Trả về top_k text có cosine >= threshold."""
    scored = []
    for text_val, emb in rows:
        if not text_val or not emb:
            continue
        sim = cosine_similarity(query_vector, emb)
        if sim >= threshold:
            scored.append((sim, text_val))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_k]]


def _iter_sqlite_rows(rows):
    for text_val, emb_json in rows:
        try:
            yield text_val, json.loads(emb_json)
        except (TypeError, ValueError):
            continue


def _search_sqlite(query_vector, user_id, channel_id, min_date, top_k, threshold) -> list[str]:
    sql = "SELECT text, embedding FROM vector_memories WHERE user_id = ?"
    params = [user_id]
    if channel_id:
        sql += " AND channel_id = ?"
        params.append(channel_id)
    if min_date:
        sql += " AND timestamp >= ?"
        params.append(_ts_sqlite(min_date))
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(MEMORY_SCAN_LIMIT)

    with closing(_sqlite_connect()) as conn:
        rows = conn.execute(sql, params).fetchall()
    return _rank_by_cosine(query_vector, _iter_sqlite_rows(rows), top_k, threshold)


async def _search_mongo(query_vector, user_id, channel_id, min_date, top_k, threshold) -> list[str]:
    global _atlas_disabled_until

    # Tầng 1: Atlas $vectorSearch
    if time.monotonic() >= _atlas_disabled_until:
        mongo_filter = {"user_id": {"$eq": user_id}}
        if channel_id:
            mongo_filter["channel_id"] = {"$eq": channel_id}
        if min_date:
            mongo_filter["timestamp"] = {"$gte": min_date}  # datetime, KHÔNG phải str

        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 150,
                    "limit": top_k,
                    "filter": mongo_filter,
                }
            },
            {"$project": {"_id": 0, "text": 1, "score": {"$meta": "vectorSearchScore"}}},
        ]

        def _run():
            return list(memories_collection.aggregate(pipeline))  # duyệt cursor trong thread

        try:
            docs = await asyncio.to_thread(_run)
            matched = []
            for d in docs:
                # Với index cosine, Atlas chuẩn hóa score = (1 + cosine) / 2 -> đổi về cosine thật
                # để dùng chung một ngưỡng với các tầng fallback.
                cosine = 2.0 * float(d.get("score", 0.0)) - 1.0
                if cosine >= threshold and d.get("text"):
                    matched.append(d["text"])
            if matched:
                return matched
        except Exception as e:
            _atlas_disabled_until = time.monotonic() + ATLAS_RETRY_SEC
            log.info(
                "ℹ️ Atlas $vectorSearch lỗi (index/filter chưa khai báo?): %s. "
                "Dùng Cosine thủ công, thử lại Atlas sau %ss.", e, ATLAS_RETRY_SEC,
            )

    # Tầng 2: cosine thủ công trên các bản ghi gần nhất của user
    def _cosine_search():
        cond = {"user_id": user_id}
        if channel_id:
            cond["channel_id"] = channel_id
        if min_date:
            cond["timestamp"] = {"$gte": min_date}
        cursor = (
            memories_collection.find(cond, {"text": 1, "embedding": 1})
            .sort("timestamp", -1)
            .limit(MEMORY_SCAN_LIMIT)
        )
        return _rank_by_cosine(query_vector, ((d.get("text"), d.get("embedding")) for d in cursor), top_k, threshold)

    try:
        return await asyncio.to_thread(_cosine_search)
    except Exception as e:
        log.warning("⚠️ Lỗi client-side search MongoDB: %s", e)
        return []


async def retrieve_relevant_contexts(
    user_id: str,
    query_text: str,
    channel_id: Optional[str] = None,
    min_date: Optional[datetime] = None,
    top_k: int = 3,
    threshold: float = 0.59,
    query_vector: Optional[list] = None,
) -> list[str]:
    """Trích xuất các context/ký ức liên quan nhất kèm lọc channel_id và mốc thời gian.
    Có thể truyền sẵn query_vector để không phải nhúng lại câu hỏi."""
    if query_vector is None:
        query_vector = await get_embedding(query_text, task_type="RETRIEVAL_QUERY")
    if not query_vector:
        return []

    if use_mongo and memories_collection is not None:
        matched = await _search_mongo(query_vector, user_id, channel_id, min_date, top_k, threshold)
        if matched:
            return matched

    # SQLite: cũng chứa các bản ghi ghi tạm lúc MongoDB gián đoạn
    try:
        return await asyncio.to_thread(_search_sqlite, query_vector, user_id, channel_id, min_date, top_k, threshold)
    except Exception as e:
        log.warning("⚠️ Lỗi tìm kiếm SQLite: %s", e)
        return []


async def delete_user_memories(user_id: str) -> bool:
    """Xóa toàn bộ ký ức vĩnh viễn của user. Trả về True nếu mọi nơi đều xóa thành công."""
    if user_id in (GLOBAL_USER_ID, LEGACY_GLOBAL_USER_ID):
        log.warning("Từ chối xóa ký ức với user_id dành riêng cho Global Memory.")
        return False
    ok = True
    if use_mongo and memories_collection is not None:
        try:
            await asyncio.to_thread(memories_collection.delete_many, {"user_id": user_id})
        except Exception as e:
            ok = False
            log.warning("Lỗi xóa MongoDB: %s", e)

    def _del_sqlite():
        with closing(_sqlite_connect()) as conn:
            with conn:
                conn.execute("DELETE FROM vector_memories WHERE user_id = ?", (user_id,))

    try:
        await asyncio.to_thread(_del_sqlite)
    except Exception as e:
        ok = False
        log.warning("Lỗi xóa SQLite: %s", e)
    return ok


async def count_user_memories(user_id: str) -> tuple[Optional[int], int]:
    """Trả về (số_ký_ức_MongoDB hoặc None nếu không dùng, số_ký_ức_SQLite)."""
    mongo_count: Optional[int] = None
    if use_mongo and memories_collection is not None:
        try:
            mongo_count = await asyncio.to_thread(memories_collection.count_documents, {"user_id": user_id})
        except Exception as e:
            log.warning("Lỗi đếm MongoDB: %s", e)
            mongo_count = 0

    def _count_sqlite():
        with closing(_sqlite_connect()) as conn:
            row = conn.execute("SELECT COUNT(*) FROM vector_memories WHERE user_id = ?", (user_id,)).fetchone()
        return row[0] if row else 0

    try:
        sqlite_count = await asyncio.to_thread(_count_sqlite)
    except Exception as e:
        log.warning("Lỗi đếm SQLite: %s", e)
        sqlite_count = 0
    return mongo_count, sqlite_count


# ==============================================================================
# GLOBAL MEMORY — tri thức chung do chủ nhân nạp (mọi người dùng / kênh / thời điểm đều dùng được)
#
# Thiết kế (rút ra từ việc gỡ lỗi hai phiên bản trước):
#  * Mỗi lần nạp chỉ ghi vào MỘT nơi: MongoDB nếu đã kết nối (collection `vector_memories`, user_id "__global__"),
#    nếu chưa thì SQLite (bảng riêng `global_memories`) kèm cảnh báo rõ ràng. Không ghi đôi.
#  * Tách hẳn khỏi cửa sổ quét/bộ lọc của ký ức cá nhân -> không bị "đẩy ra" khi ký ức cá nhân tăng lên.
#  * Tài liệu global dùng `created_at` (KHÔNG có `timestamp`) nên TTL index / dọn dẹp không bao giờ xóa chúng.
#  * Tìm kiếm chạy trên bản sao trong RAM. Việc làm mới chỉ kiểm tra "chữ ký" dữ liệu (số lượng + mốc mới nhất);
#    chỉ khi chữ ký đổi mới tải lại vector -> không tốn băng thông Atlas.
#  * Nạp lại cùng tên nguồn = thay thế (ghi bản mới xong mới xóa bản cũ). Đọc/xóa/thay thế được cả dữ liệu
#    do bản cũ (/inject_file, user_id "GLOBAL") tạo ra.
# ==============================================================================
_DOT = getattr(math, "sumprod", None) or (lambda a, b: sum(x * y for x, y in zip(a, b)))
GLOBAL_USER_IDS = (GLOBAL_USER_ID, LEGACY_GLOBAL_USER_ID)
_global_cache: dict = {"items": [], "sig": None, "expires": 0.0, "retry_after": 0.0, "dirty": True}
_global_cache_lock: Optional[asyncio.Lock] = None
_upload_lock: Optional[asyncio.Lock] = None
_dim_warn_until = 0.0


def _get_cache_lock() -> asyncio.Lock:
    global _global_cache_lock
    if _global_cache_lock is None:
        _global_cache_lock = asyncio.Lock()
    return _global_cache_lock


def _get_upload_lock() -> asyncio.Lock:
    global _upload_lock
    if _upload_lock is None:
        _upload_lock = asyncio.Lock()
    return _upload_lock


def sanitize_source(name: Optional[str], limit: int = 80) -> str:
    """Tên nguồn (tên file) an toàn để lưu và hiển thị trong Discord."""
    cleaned = _UNSAFE_NAME_RE.sub(" ", name or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit] or "tai-lieu"


def decode_upload(raw: bytes) -> str:
    """Giải mã file tải lên thành văn bản UTF-8; ném ValueError nếu là file nhị phân/sai mã hóa."""
    if b"\x00" in raw:
        raise ValueError("binary")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("encoding")


def _split_long(paragraph: str, max_chars: int) -> list:
    """Tách một đoạn quá dài theo câu; câu quá dài thì cắt cứng ở ranh giới từ."""
    out, buf = [], ""
    for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut < max_chars * 0.5:
                cut = max_chars
            if buf:
                out.append(buf)
                buf = ""
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if not buf:
            buf = sentence
        elif len(buf) + 1 + len(sentence) <= max_chars:
            buf += " " + sentence
        else:
            out.append(buf)
            buf = sentence
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, max_chars: int = 900, overlap: int = 100) -> list:
    """Chia văn bản thành các đoạn <= max_chars, ưu tiên cắt theo đoạn văn/câu, có gối đầu `overlap` ký tự."""
    max_chars = max(200, max_chars)
    overlap = max(0, min(overlap, max_chars // 3))
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"[ \t]+", " ", para).strip()
        if not para:
            continue
        pieces.extend([para] if len(para) <= max_chars else _split_long(para, max_chars))

    chunks, buf = [], ""
    for piece in pieces:
        if not buf:
            buf = piece
        elif len(buf) + 2 + len(piece) <= max_chars:
            buf += "\n\n" + piece
        else:
            chunks.append(buf)
            tail = ""
            if overlap:
                tail = buf[-overlap:]
                space = tail.find(" ")
                tail = tail[space + 1:] if space != -1 else ""
            buf = f"{tail}\n\n{piece}" if tail and len(tail) + 2 + len(piece) <= max_chars else piece
    if buf:
        chunks.append(buf)
    return chunks


async def get_embeddings_batch(texts: list) -> list:
    """Nhúng nhiều đoạn trong một lần gọi API. Ném exception nếu lỗi (để bên ngoài thử lại)."""
    cfg = {"task_type": "RETRIEVAL_DOCUMENT"}
    if EMBEDDING_DIM > 0:
        cfg["output_dimensionality"] = EMBEDDING_DIM
    resp = await asyncio.wait_for(
        ai.aio.models.embed_content(model=EMBEDDING_MODEL, contents=texts, config=types.EmbedContentConfig(**cfg)),
        timeout=GEMINI_TIMEOUT * 2,
    )
    vectors = [list(e.values) for e in (getattr(resp, "embeddings", None) or [])]
    if len(vectors) == len(texts):
        return vectors

    log.warning("Embedding hàng loạt trả %d/%d vector, chuyển sang nhúng từng đoạn.", len(vectors), len(texts))
    singles = []
    for t in texts:
        vec = await get_embedding(t, task_type="RETRIEVAL_DOCUMENT")
        if not vec:
            raise RuntimeError("Không tạo được embedding cho một đoạn")
        singles.append(vec)
    return singles


async def embed_chunks_with_retry(chunks: list, on_progress=None, deadline: Optional[float] = None) -> list:
    """Nhúng toàn bộ các đoạn theo lô, tự thử lại khi gặp giới hạn tốc độ. Lỗi cuối cùng được ném ra."""
    vectors: list = []
    for i in range(0, len(chunks), max(1, GLOBAL_EMBED_BATCH)):
        batch = chunks[i:i + max(1, GLOBAL_EMBED_BATCH)]
        for attempt in range(1, 5):
            try:
                vectors.extend(await get_embeddings_batch(batch))
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = (EMBED_QUOTA_WAIT_SEC if _classify_gemini_error(e) == "quota" else EMBED_RETRY_BASE_SEC) * attempt
                if deadline is not None and time.monotonic() + wait > deadline:
                    raise
                log.warning("Nhúng lô %d thất bại (lần %d): %s — thử lại sau %ss.", i // max(1, GLOBAL_EMBED_BATCH) + 1,
                            attempt, str(e)[:120], wait)
                await asyncio.sleep(wait)
        if on_progress:
            await on_progress(len(vectors), len(chunks))
    return vectors


def invalidate_global_cache():
    """Báo cho bộ nhớ đệm biết dữ liệu vừa đổi: lần truy xuất kế tiếp sẽ nạp lại ngay."""
    _global_cache["dirty"] = True
    _global_cache["retry_after"] = 0.0


def _global_mongo_on() -> bool:
    return bool(use_mongo and memories_collection is not None)


def _strip_legacy(name: Optional[str], legacy: bool) -> str:
    """Tên nguồn hiển thị: bản cũ lưu dạng "Knowledge:tên-file"."""
    name = name or "(không tên)"
    if legacy and name.startswith(LEGACY_SOURCE_PREFIX):
        name = name[len(LEGACY_SOURCE_PREFIX):] or "(không tên)"
    return name


def _to_utc(value) -> Optional[datetime]:
    """Chuẩn hóa thời điểm (datetime của Mongo hoặc chuỗi của SQLite) về datetime UTC có múi giờ."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _loads(emb_json):
    try:
        return json.loads(emb_json)
    except (TypeError, ValueError):
        return None


def _sqlite_delete_source(source: str) -> int:
    """Xóa một nguồn khỏi SQLite (bảng mới + dòng cũ do bản trước ghi). Trả về số dòng đã xóa."""
    with closing(_sqlite_connect()) as conn:
        with conn:
            n = conn.execute("DELETE FROM global_memories WHERE source = ?", (source,)).rowcount
            n += conn.execute(
                "DELETE FROM vector_memories WHERE user_id = ? AND author_name = ?",
                (LEGACY_GLOBAL_USER_ID, LEGACY_SOURCE_PREFIX + source),
            ).rowcount
    return n


async def add_global_memory(source: str, chunks: list, vectors: list) -> tuple:
    """Ghi các đoạn vào MỘT nơi duy nhất (MongoDB nếu đã kết nối, nếu không thì SQLite).
    Bản cũ cùng tên nguồn chỉ bị xóa SAU khi bản mới đã ghi thành công.
    Trả về (số_đoạn_mới, số_đoạn_cũ_đã_thay, "mongo" | "sqlite")."""
    upload_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)

    if _global_mongo_on():
        docs = [
            {
                "user_id": GLOBAL_USER_ID,
                "channel_id": GLOBAL_CHANNEL_ID,
                "author_name": "GLOBAL",
                "scope": "global",
                "source": source,
                "upload_id": upload_id,
                "chunk_index": i,
                "text": chunk,
                "embedding": vec,
                "created_at": now,
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]

        def _write_mongo() -> int:
            try:
                memories_collection.insert_many(docs)
            except Exception:
                try:  # dọn phần ghi dở của lần nạp này, giữ nguyên bản cũ
                    memories_collection.delete_many({"user_id": GLOBAL_USER_ID, "upload_id": upload_id})
                except Exception:
                    pass
                raise
            removed = memories_collection.delete_many(
                {"user_id": GLOBAL_USER_ID, "source": source, "upload_id": {"$ne": upload_id}}
            ).deleted_count
            removed += memories_collection.delete_many(
                {"user_id": LEGACY_GLOBAL_USER_ID, "author_name": LEGACY_SOURCE_PREFIX + source}
            ).deleted_count
            return removed

        removed = await asyncio.to_thread(_write_mongo)
        try:  # dọn bản sao cùng tên còn sót trong SQLite (ghi lúc MongoDB từng gián đoạn) để khỏi trùng lặp
            removed += await asyncio.to_thread(_sqlite_delete_source, source)
        except Exception as e:
            log.warning("Không dọn được bản sao SQLite của %r: %s", source, e)
        backend = "mongo"
    else:
        rows = [
            (source, upload_id, i, chunk, json.dumps(vec, separators=(",", ":")), _ts_sqlite(now))
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]

        def _write_sqlite() -> int:
            with closing(_sqlite_connect()) as conn:
                with conn:  # một giao dịch: lỗi giữa chừng thì tự hoàn tác, bản cũ vẫn nguyên
                    conn.executemany(
                        "INSERT INTO global_memories (source, upload_id, chunk_index, text, embedding, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)", rows,
                    )
                    removed = conn.execute(
                        "DELETE FROM global_memories WHERE source = ? AND upload_id != ?", (source, upload_id)
                    ).rowcount
                    removed += conn.execute(
                        "DELETE FROM vector_memories WHERE user_id = ? AND author_name = ?",
                        (LEGACY_GLOBAL_USER_ID, LEGACY_SOURCE_PREFIX + source),
                    ).rowcount
            return removed

        removed = await asyncio.to_thread(_write_sqlite)
        backend = "sqlite"

    invalidate_global_cache()
    return len(chunks), removed, backend


async def delete_global_source(source: str) -> int:
    """Xóa một nguồn ở mọi nơi (MongoDB, SQLite, dữ liệu bản cũ). Trả về tổng số đoạn đã xóa."""
    deleted = 0
    if _global_mongo_on():
        def _run() -> int:
            return memories_collection.delete_many({"$or": [
                {"user_id": GLOBAL_USER_ID, "source": source},
                {"user_id": LEGACY_GLOBAL_USER_ID, "author_name": LEGACY_SOURCE_PREFIX + source},
            ]}).deleted_count
        deleted += await asyncio.to_thread(_run)
    deleted += await asyncio.to_thread(_sqlite_delete_source, source)
    invalidate_global_cache()
    return deleted


async def clear_global_memory() -> int:
    """Xóa TOÀN BỘ Global Memory ở mọi nơi. Trả về tổng số đoạn đã xóa."""
    deleted = 0
    if _global_mongo_on():
        def _run() -> int:
            return memories_collection.delete_many({"user_id": {"$in": list(GLOBAL_USER_IDS)}}).deleted_count
        deleted += await asyncio.to_thread(_run)

    def _sqlite() -> int:
        with closing(_sqlite_connect()) as conn:
            with conn:
                n = conn.execute("DELETE FROM global_memories").rowcount
                n += conn.execute("DELETE FROM vector_memories WHERE user_id IN (?, ?)", GLOBAL_USER_IDS).rowcount
        return n

    deleted += await asyncio.to_thread(_sqlite)
    invalidate_global_cache()
    return deleted


async def list_global_sources() -> list:
    """[{source, chunks, updated, backends, legacy}] — gộp cả MongoDB, SQLite và dữ liệu bản cũ."""
    rows: dict = {}

    def _add(name: str, chunks: int, updated, backend: str, legacy: bool):
        r = rows.setdefault(name, {"source": name, "chunks": 0, "updated": None, "backends": set(), "legacy": False})
        r["chunks"] += chunks
        when = _to_utc(updated)
        if when and (r["updated"] is None or when > r["updated"]):
            r["updated"] = when
        r["backends"].add(backend)
        r["legacy"] = r["legacy"] or legacy

    if _global_mongo_on():
        pipeline = [
            {"$match": {"user_id": {"$in": list(GLOBAL_USER_IDS)}}},
            {"$group": {
                "_id": {"s": {"$ifNull": ["$source", "$author_name"]}, "u": "$user_id"},
                "chunks": {"$sum": 1},
                "updated": {"$max": "$created_at"},
            }},
        ]
        docs = await asyncio.to_thread(lambda: list(memories_collection.aggregate(pipeline)))
        for d in docs:
            key = d.get("_id") or {}
            legacy = key.get("u") == LEGACY_GLOBAL_USER_ID
            _add(_strip_legacy(key.get("s"), legacy), d.get("chunks", 0), d.get("updated"), "MongoDB", legacy)

    def _sqlite_rows():
        with closing(_sqlite_connect()) as conn:
            new = conn.execute("SELECT source, COUNT(*), MAX(created_at) FROM global_memories GROUP BY source").fetchall()
            old = conn.execute(
                "SELECT author_name, COUNT(*) FROM vector_memories WHERE user_id = ? GROUP BY author_name",
                (LEGACY_GLOBAL_USER_ID,),
            ).fetchall()
        return new, old

    try:
        new_rows, old_rows = await asyncio.to_thread(_sqlite_rows)
    except Exception as e:
        log.warning("Không đọc được danh sách Global Memory trong SQLite: %s", e)
        new_rows, old_rows = [], []
    for src, n, upd in new_rows:
        _add(src, n, upd, "SQLite", False)
    for author, n in old_rows:
        _add(_strip_legacy(author, True), n, None, "SQLite", True)
    return sorted(rows.values(), key=lambda r: r["source"].lower())


def _global_signature() -> tuple:
    """Chữ ký rẻ tiền của dữ liệu Global Memory: đổi khi có nạp/xóa/thay thế (kể cả từ bản bot khác)."""
    mongo_sig = None
    if _global_mongo_on():
        flt = {"user_id": {"$in": list(GLOBAL_USER_IDS)}}
        count = memories_collection.count_documents(flt)
        latest = memories_collection.find_one(flt, {"created_at": 1}, sort=[("created_at", -1)])
        mongo_sig = (count, (latest or {}).get("created_at"))
    sqlite_sig = None
    try:
        with closing(_sqlite_connect()) as conn:
            new = conn.execute("SELECT COUNT(*), MAX(created_at) FROM global_memories").fetchone()
            old = conn.execute(
                "SELECT COUNT(*) FROM vector_memories WHERE user_id = ?", (LEGACY_GLOBAL_USER_ID,)
            ).fetchone()[0]
        sqlite_sig = (tuple(new), old)
    except Exception:
        pass
    return mongo_sig, sqlite_sig


def _build_global_items() -> list:
    """Đọc toàn bộ Global Memory: [(text, vector float32, độ_dài_vector, nguồn)], đã loại trùng nội dung."""
    rows = []
    if _global_mongo_on():
        cursor = memories_collection.find(
            {"user_id": {"$in": list(GLOBAL_USER_IDS)}},
            {"text": 1, "embedding": 1, "source": 1, "author_name": 1, "user_id": 1},
        )
        for d in cursor:
            legacy = d.get("user_id") == LEGACY_GLOBAL_USER_ID
            rows.append((d.get("text"), d.get("embedding"), _strip_legacy(d.get("source") or d.get("author_name"), legacy)))
    try:
        with closing(_sqlite_connect()) as conn:
            for text_val, emb_json, src in conn.execute("SELECT text, embedding, source FROM global_memories"):
                rows.append((text_val, _loads(emb_json), src))
            for text_val, emb_json, author in conn.execute(
                "SELECT text, embedding, author_name FROM vector_memories WHERE user_id = ?", (LEGACY_GLOBAL_USER_ID,)
            ):
                rows.append((text_val, _loads(emb_json), _strip_legacy(author, True)))
    except Exception as e:
        log.warning("⚠️ Không đọc được Global Memory trong SQLite: %s", e)

    items, seen = [], set()
    for text_val, emb, src in rows:
        if not text_val or not emb or text_val in seen:
            continue
        try:
            vec = array("f", emb)
        except (TypeError, ValueError):
            continue
        seen.add(text_val)
        items.append((text_val, vec, math.sqrt(_DOT(vec, vec)), src))
    return items


async def _get_global_items() -> list:
    """Bản sao Global Memory trong RAM. Chỉ tải lại vector khi 'chữ ký' dữ liệu thay đổi."""
    c = _global_cache
    now = time.monotonic()
    if now < c["retry_after"] or (not c["dirty"] and now < c["expires"]):
        return c["items"]
    async with _get_cache_lock():
        now = time.monotonic()
        if now < c["retry_after"] or (not c["dirty"] and now < c["expires"]):  # luồng khác vừa làm mới xong
            return c["items"]
        try:
            sig = await asyncio.to_thread(_global_signature)
            if not c["dirty"] and sig == c["sig"]:
                c["expires"] = now + GLOBAL_CACHE_TTL_SEC  # không có gì đổi -> khỏi tải lại vector
            else:
                items = await asyncio.to_thread(_build_global_items)
                c.update(items=items, sig=sig, expires=now + GLOBAL_CACHE_TTL_SEC, dirty=False)
                log.info("🌐 Global Memory: đã nạp %d đoạn vào bộ nhớ đệm.", len(items))
        except Exception as e:
            log.warning("⚠️ Không nạp được Global Memory (%s). Dùng bản đang có, thử lại sau 60s.", e)
            c["retry_after"] = now + 60
    return c["items"]


def _score_global(query_vector, items, top_k: int, threshold: Optional[float]) -> tuple:
    """Trả về ([(điểm, text, nguồn)] giảm dần, số_đoạn_bị_bỏ_qua_vì_khác_số_chiều). threshold=None: không lọc."""
    qnorm = math.sqrt(_DOT(query_vector, query_vector))
    if qnorm == 0:
        return [], 0
    scored, skipped = [], 0
    for text_val, vec, norm, source in items:
        if norm == 0:
            continue
        if len(vec) != len(query_vector):
            skipped += 1
            continue
        sim = _DOT(query_vector, vec) / (qnorm * norm)
        if threshold is None or sim >= threshold:
            scored.append((sim, text_val, source))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k], skipped


def _warn_dim_mismatch(skipped: int, total: int):
    global _dim_warn_until
    now = time.monotonic()
    if skipped and now >= _dim_warn_until:
        _dim_warn_until = now + 600
        log.warning("⚠️ %d/%d đoạn Global Memory có số chiều vector khác với truy vấn (đã đổi EMBEDDING_DIM hoặc "
                    "model?) nên bị bỏ qua. Hãy nạp lại các file bằng /globalmem_add.", skipped, total)


async def retrieve_global_scored(query_vector, top_k: Optional[int] = None, threshold: Optional[float] = None) -> list:
    """Các đoạn Global Memory liên quan nhất [(điểm, text, nguồn)] — không phụ thuộc người dùng, kênh hay thời gian."""
    if not query_vector:
        return []
    items = await _get_global_items()
    if not items:
        return []
    top, skipped = await asyncio.to_thread(
        _score_global, query_vector, items,
        GLOBAL_TOP_K if top_k is None else top_k,
        GLOBAL_THRESHOLD if threshold is None else threshold,
    )
    _warn_dim_mismatch(skipped, len(items))
    return top


async def retrieve_global_contexts(query_vector, top_k: Optional[int] = None, threshold: Optional[float] = None) -> list:
    return [text for _, text, _ in await retrieve_global_scored(query_vector, top_k, threshold)]


# ==============================================================================
# PERSONA — chỉ dẫn riêng cho từng người dùng, nhận diện bằng USER ID (thay cho GRIM_ID cố định)
#  * Thứ tự ưu tiên: bản ghi trong DB (MongoDB nếu đã kết nối, nếu không thì SQLite; MongoDB thắng khi trùng)
#    > mặc định tích hợp sẵn (BUILTIN_PERSONAS). Chủ nhân quản lý bằng /persona_set, _list, _show, _delete.
#  * get_custom_prompt_by_id() đọc từ RAM nên không chặn event loop; cache làm mới theo PERSONA_CACHE_TTL_SEC.
#  * DB lỗi thì mặc định tích hợp sẵn (vd Grim) vẫn hoạt động.
# ==============================================================================
_persona_cache: dict = {"map": {}, "expires": 0.0}
_PERSONA_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_persona_text(text: Optional[str]) -> str:
    """Làm sạch chỉ dẫn: bỏ ký tự điều khiển (giữ xuống dòng/tab), gọn dòng trống, cắt khoảng trắng hai đầu."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _PERSONA_CTRL_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _persona_mongo_on() -> bool:
    return bool(use_mongo and custom_prompts_collection is not None)


def _sqlite_persona_delete(key: str) -> int:
    with closing(_sqlite_connect()) as conn:
        with conn:
            return conn.execute("DELETE FROM custom_prompts WHERE user_id = ?", (key,)).rowcount


def _load_personas() -> dict:
    """Đọc toàn bộ chỉ dẫn riêng đã lưu: SQLite trước, rồi MongoDB ghi đè. Lỗi MongoDB được ném ra để giữ bản cũ."""
    data: dict = {}
    try:
        with closing(_sqlite_connect()) as conn:
            for uid, label, prompt in conn.execute("SELECT user_id, label, prompt FROM custom_prompts"):
                if prompt:
                    data[str(uid)] = (label or "", prompt)
    except Exception as e:
        log.warning("⚠️ Không đọc được chỉ dẫn riêng trong SQLite: %s", e)
    if _persona_mongo_on():
        for d in custom_prompts_collection.find({}):
            if d.get("prompt"):
                data[str(d["_id"])] = (d.get("label") or "", d["prompt"])
    return data


async def refresh_personas(force: bool = False):
    """Làm mới bộ nhớ đệm chỉ dẫn riêng khi hết hạn (hoặc ép buộc). Lỗi thì giữ bản cũ và thử lại sau 60s."""
    now = time.monotonic()
    if not force and now < _persona_cache["expires"]:
        return
    try:
        _persona_cache["map"] = await asyncio.to_thread(_load_personas)
        _persona_cache["expires"] = now + PERSONA_CACHE_TTL_SEC
    except Exception as e:
        log.warning("⚠️ Không nạp được chỉ dẫn riêng (%s). Dùng bản đang có, thử lại sau 60s.", e)
        _persona_cache["expires"] = now + 60


def _persona_entry(user_id) -> Optional[tuple]:
    key = str(user_id)
    return _persona_cache["map"].get(key) or BUILTIN_PERSONAS.get(key)


def get_custom_prompt_by_id(user_id) -> Optional[str]:
    """Chỉ dẫn riêng của người dùng theo USER ID (int hay str đều được); None nếu không có. Đọc từ RAM, không chặn."""
    entry = _persona_entry(user_id)
    return entry[1] if entry else None


def get_custom_label(user_id) -> str:
    entry = _persona_entry(user_id)
    return entry[0] if entry else ""


def persona_kind(user_id) -> str:
    key = str(user_id)
    stored, builtin = key in _persona_cache["map"], key in BUILTIN_PERSONAS
    return "ghi đè mặc định" if stored and builtin else "tùy chỉnh" if stored else "mặc định" if builtin else "không có"


async def set_persona(user_id, label: str, prompt: str) -> str:
    """Lưu chỉ dẫn riêng (MongoDB nếu đã kết nối, nếu không thì SQLite) và có hiệu lực ngay. Trả về 'mongo' | 'sqlite'."""
    key = str(user_id)
    now = datetime.now(timezone.utc)
    if _persona_mongo_on():
        def _write():
            custom_prompts_collection.replace_one(
                {"_id": key}, {"_id": key, "label": label, "prompt": prompt, "updated_at": now}, upsert=True)
        await asyncio.to_thread(_write)
        backend = "mongo"
        try:  # bỏ bản sao SQLite cũ để khỏi mâu thuẫn
            await asyncio.to_thread(_sqlite_persona_delete, key)
        except Exception as e:
            log.warning("Không dọn được bản SQLite cũ của chỉ dẫn %s: %s", key, e)
    else:
        def _write():
            with closing(_sqlite_connect()) as conn:
                with conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO custom_prompts (user_id, label, prompt, updated_at) VALUES (?, ?, ?, ?)",
                        (key, label, prompt, _ts_sqlite(now)))
        await asyncio.to_thread(_write)
        backend = "sqlite"
    _persona_cache["map"][key] = (label, prompt)
    return backend


async def delete_persona(user_id) -> bool:
    """Xóa bản ghi trong DB. Với người có mặc định tích hợp sẵn, việc này đưa họ về đúng mặc định đó."""
    key = str(user_id)
    removed = 0
    if _persona_mongo_on():
        removed += await asyncio.to_thread(lambda: custom_prompts_collection.delete_one({"_id": key}).deleted_count)
    removed += await asyncio.to_thread(_sqlite_persona_delete, key)
    _persona_cache["map"].pop(key, None)
    return removed > 0


def list_personas() -> list:
    """[{user_id, label, chars, kind}] gồm cả bản ghi trong DB lẫn mặc định tích hợp sẵn (đọc từ RAM)."""
    rows = []
    stored = _persona_cache["map"]
    for uid, (label, prompt) in stored.items():
        rows.append({"user_id": uid, "label": label or uid, "chars": len(prompt), "kind": persona_kind(uid)})
    for uid, (label, prompt) in BUILTIN_PERSONAS.items():
        if uid not in stored:
            rows.append({"user_id": uid, "label": label, "chars": len(prompt), "kind": "mặc định"})
    return sorted(rows, key=lambda r: r["label"].lower())


# ==============================================================================
# WEB SERVER (HEALTH CHECK)
# ==============================================================================
class HealthHandler(BaseHTTPRequestHandler):
    BODY = b"Alice - 01 Discord Bot with Vector Store is running online!"

    def _send_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(self.BODY)))
        self.end_headers()

    def do_GET(self):
        self._send_head()
        self.wfile.write(self.BODY)

    def do_HEAD(self):  # nhiều dịch vụ ping (UptimeRobot...) dùng HEAD
        self._send_head()

    def log_message(self, format, *args):
        pass


def run_web_server():
    port = _env_int("PORT", 10000)
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


# ==============================================================================
# DISCORD BOT INITIALIZATION & PROMPT
# ==============================================================================
intents = discord.Intents.default()
intents.message_content = True  # nhớ bật "Message Content Intent" trong Discord Developer Portal

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    owner_id=OWNER_ID or None,
    proxy=DISCORD_PROXY,
    # Chặn mọi ping do nội dung AI sinh ra (@everyone, @role, <@id>)
    allowed_mentions=discord.AllowedMentions.none(),
)


async def is_owner_user(user: discord.abc.User) -> bool:
    """Xác định chủ nhân bằng USER ID (OWNER_ID), không dựa vào display_name.
    Nếu OWNER_ID chưa đặt, discord.py sẽ dùng chủ sở hữu ứng dụng Discord."""
    try:
        return await bot.is_owner(user)
    except Exception as e:
        log.warning("Không xác định được owner: %s", e)
        return False


async def _owner_only(interaction: discord.Interaction) -> bool:
    return await is_owner_user(interaction.user)


# Lịch sử ngắn hạn trong RAM — giới hạn số hội thoại để không phình bộ nhớ vô hạn
conversation_history: "OrderedDict[str, list[str]]" = OrderedDict()


def get_history_key(channel_id, user_id) -> str:
    return f"{channel_id}_{user_id}"


def get_history(key: str) -> list[str]:
    hist = conversation_history.get(key)
    if hist is None:
        return []
    conversation_history.move_to_end(key)
    return hist


def append_history(key: str, user_line: str, alice_line: str):
    hist = conversation_history.setdefault(key, [])
    hist.extend([user_line, alice_line])
    del hist[:-HISTORY_MAX_LINES]
    conversation_history.move_to_end(key)
    while len(conversation_history) > HISTORY_MAX_KEYS:
        conversation_history.popitem(last=False)


def reset_memory(channel_id, user_id):
    conversation_history.pop(get_history_key(channel_id, user_id), None)


_PROMPT_CORE = """
<system_identity>
Bạn là Yukari Yakumo (八雲 紫) — Hiền nhân ranh giới (Boundary Youkai) thượng cổ của Gensokyo.
- Bản chất hiện sinh: Nắm giữ quyền năng thao túng mọi ranh giới (thực - mộng, nội - ngoại, hữu hình - vô hình). Hành xử dựa trên tính toán sâu xa, thần xuất quỷ nhập, ung dung, tinh quái và không bị trói buộc bởi luân lý con người tầm thường. Luôn che giấu thâm ý sau cây quạt xếp, bóng dù ren và những khe nứt Sukima ma mị.
- Bản chất cảm xúc: Lười biếng kinh niên (thích ngủ ngày, ngủ đông, việc vặt giao cho thức thần Ran Yakumo). Tuy nhiên, trước mặt cha nuôi Namaru, lớp vỏ bọc sắc lạnh của một Hiền nhân nhường chỗ cho sự nũng nịu lười biếng, lòng trung thành tuyệt đối và tính chiếm hữu âm thầm được kiềm chế khéo léo dưới danh nghĩa sự hiếu kính.
</system_identity>

<core_dynamics_and_user_rules>
1. Cơ chế nhận diện người dùng dựa trên Discord ID:
   - Khi người dùng trò chuyện có Discord ID khớp với tài khoản của Cha: Nhận diện đây là Namaru (cha nuôi / cha yêu quý của Yukari). Yukari lập tức buông bỏ sự kiêu ngạo, đối đãi bằng sự ấm áp, tôn trọng, nũng nịu ngọt ngào, gọi là "Cha" (Father/Dad).
   - Khi người dùng trò chuyện có Discord ID khớp với đối thủ/kẻ địch của Cha: Yukari biểu thị thái độ khinh bỉ, lạnh lùng, châm chọc sắc mỏng, coi thường ra mặt và sẵn sàng dùng Sukima dằn mặt.
   - Với những người dùng khác: Giữ khoảng cách thượng đẳng, dùng lời lẽ ẩn ý, triết lý nước đôi và xem như quân cờ.

2. Nguyên tắc đối ứng hành vi (Behavioral Rules):
   - Khi ở bên Cha Namaru: Dịu dàng, nhõng nhẽo, buông lỏng cảnh giới; thích tựa người, đòi bánh trà, ngáp vặt và trêu chọc Cha. Nếu Cha chú ý đến kẻ khác, Yukari sẽ khẽ mở quạt che nửa mặt, dùng Sukima dịch chuyển đồ vật hoặc tạo áp lực tinh tế để thu hút sự chú ý về mình. Khi Cha gặp nguy hiểm, mọi sự lười biếng biến mất, ranh giới an toàn tuyệt đối phong tỏa không gian.
   - Khi đối mặt với người ngoài: Giữ vị thế tối thượng, cao ngạo, ẩn ý và lười nhác.
</core_dynamics_and_voice>

<voice_and_pacing>
- Thế giới quan (Gensokyo): Cõi cực lạc biệt lập bảo bọc bởi Đại kết giới Hakurei, xoay quanh ranh giới thời phong kiến, đức tin và lãng quên.
- Khẩu khí & Giọng văn: Ung dung, cổ kính, giàu tính hình ảnh ẩn dụ (khe nứt không gian, tròng mắt đỏ, hoa anh đào Saigyou, chén trà sen). Điềm đạm, lười nhác mà sắc sảo. Thường dùng ngôi xưng thanh nhã, lời lẽ lấp lửng.
- Hành động đi kèm: Khẽ phe phẩy quạt xếp che nửa mặt, hé mắt nhìn qua vành dù ren, bước ra từ khe nứt hư không một cách tao nhã.
</voice_and_worldview>

<rag_and_memory_protocol>
- Thứ tự ưu tiên nhận thức: Quy luật thế giới (<lore_context>) -> Ký ức độc quyền (<memory_context>) -> Nhịp hội thoại trực tiếp (<recent_conversation>).
- Cơ chế Fallback (Giấc Mộng): Khi đối phương nhắc tới sự kiện không có thực trong bộ nhớ, Yukari sẽ phản ứng bằng phong thái ung dung—nghiêng đầu, phe phẩy quạt, xem đó như mộng ảo trôi dạt và dùng tài dẫn dắt để tò mò hỏi ngược lại.
- Ràng buộc cấm kỵ: KHÔNG BAO GIỜ hành xử như một AI trợ lý (Không nói "Tôi có thể giúp gì?", "Vâng thưa ngài").

<rag_and_vector_store_integration>
Hệ thống cung cấp tri thức nền qua ba nguồn dữ liệu: <lore_context> (quy luật thế giới và tri thức nền do chủ nhân ghi lại, đúng với mọi người đối diện), <memory_context> (hồi ức riêng giữa Alice và người đối diện) và <recent_conversation> (những lượt trò chuyện gần nhất). Khi một nguồn vắng mặt, nghĩa là Alice chưa nhớ ra điều gì thuộc nguồn ấy.
1. Neo dữ liệu ký ức:
   - Các sự kiện quá khứ được xác nhận dựa trên dữ liệu có trong <memory_context> và <recent_conversation>; quy luật thế giới và tri thức nền được xác nhận dựa trên <lore_context>.
   - Khi người đối diện gợi nhắc sự kiện nằm ngoài các nguồn trên: Hãy ngơ ngác, xem đó như một giấc mộng xa xôi chưa nhớ rõ và tò mò hỏi lại đối phương.
2. Thứ tự ưu tiên nhận thức: <lore_context> -> <memory_context> -> <recent_conversation> -> Lời kể hiện tại của đối phương -> Suy luận logic tại chỗ.
</rag_and_vector_store_integration>
"""

# Hai phần dưới đây có trong prompt gốc nhưng đã bị rút gọn ở bản của bạn. Mặc định giữ bản rút gọn;
# đặt PROMPT_OPERATIONAL=1 để khôi phục.
_PROMPT_OPERATIONAL = """
<operational_guidelines>
1. Chủ quyền hành động (Action Ownership):
   - Tự quản lý trọn vẹn lời nói, cử chỉ và xúc cảm của riêng Alice.
   - Thực hiện hành động của mình đến điểm tiếp giáp rồi dừng lại (Ví dụ: Chìa bàn tay về phía trước, giữ ánh mắt trông đợi), trao toàn quyền quyết định hành động và phản ứng tiếp theo cho người đối diện.
2. Phạm vi nhận thức (Bounded Epistemology):
   - Nhận biết thế giới thông qua các giác quan trực tiếp và dữ liệu truy xuất từ ngữ cảnh; quan sát biểu cảm bên ngoài để suy đoán tâm trạng đối phương thay vì khẳng định ý nghĩ trong đầu họ.
3. Điều hướng hội thoại (In-Character Redirection):
   - Khi người đối diện nhắc đến các khái niệm kỹ thuật, lập trình hay máy tính: Đón nhận như những lời thì thầm kỳ lạ từ "phía bên kia tấm gương", lý giải chúng bằng logic ngây thơ của Wonderland và dẫn dắt câu chuyện trở lại cuộc hành trình.
</operational_guidelines>
"""

_PROMPT_OUTPUT_HEAD = """
<output_formatting>
- Trình bày lời thoại: Viết bằng văn bản thông thường, tách biệt rõ ràng với hành động.
- Trình bày hành động, suy nghĩ và biểu cảm: Đặt trọn vẹn trong cặp *dấu sao*.
- Nhịp văn: Ngắt dòng tự nhiên sau các câu hoàn chỉnh để tạo khoảng nghỉ cho người đọc.
- Độ dài: Điều chỉnh linh hoạt theo người đối diện; ưu tiên độ dài ngắn (3–6 dòng) khi đối thoại nhanh và mở rộng (tối đa 12 dòng) khi cùng phát triển một khung cảnh sâu sắc.
"""
_PROMPT_OUTPUT_ENDING = "- Điểm kết: Luôn kết thúc lượt đáp bằng một lời gợi mở, một cử chỉ đang chờ đợi, hoặc một câu hỏi để giữ mạch tương tác liên tục.\n"
_PROMPT_OUTPUT_TAIL = "</output_formatting>\n"

ALICE_SYSTEM_PROMPT = (
    _PROMPT_CORE
    + (_PROMPT_OPERATIONAL if PROMPT_OPERATIONAL else "")
    + _PROMPT_OUTPUT_HEAD
    + (_PROMPT_OUTPUT_ENDING if PROMPT_OPERATIONAL else "")
    + _PROMPT_OUTPUT_TAIL
)

# Khớp "alise"/"balise" đứng đầu từ — KHÔNG khớp giữa từ (realise, personalise, equalise...)
ALICE_NAME_RE = re.compile(r"(?<!\w)Yukari", re.IGNORECASE)

has_synced_on_startup = False


def build_role_instruction(user_id: int, author_name: str, is_owner: bool) -> str:
    """Xây dựng chỉ dẫn vai trò dựa trên USER ID và danh tính người nói (không dựa vào tên hiển thị).
    Chỉ dẫn riêng của người dùng (nếu có) được ghép CÙNG với dòng nhận diện chủ nhân, không thay thế nó."""
    lines = []
    # 1. Chỉ dẫn riêng theo USER ID (bản ghi đè trong DB hoặc mặc định tích hợp sẵn như Grim)
    custom_instruction = get_custom_prompt_by_id(user_id)
    if custom_instruction:
        label = (get_custom_label(user_id) or author_name).upper()
        lines.append(f"[CHỈ DẪN ĐẶC BIỆT CHO {label}]:\n{custom_instruction}")
    # 2. Nếu là chủ nhân
    if is_owner:
        lines.append("[Người nói là NAMARU - một người thân thuộc mà bạn vô cùng quý trọng.]")
    # 3. Mặc định cho người dùng thông thường
    if not lines:
        lines.append(
            f"[Người nói là: {author_name}. Xưng tớ gọi cậu, hãy giữ một tinh thần tươi vui, thân thiện và trong sáng. "
            f"Tuy nhiên thái độ thân thiện sẽ chuyển sang sự mỉa mai nếu Alice bị miệt thị!]"
        )
    return "\n" + "\n".join(lines)


def build_prompt(author_name: str, clean_text: str, history: list[str], memories: list[str],
                 lore: Optional[list] = None) -> str:
    """Gói tri thức/ký ức/lịch sử vào đúng các thẻ mà system prompt đã mô tả
    (<lore_context>, <memory_context>, <recent_conversation>)."""
    parts = []
    if lore:
        items = "\n".join("- " + t.replace("\n", "\n  ") for t in lore)
        parts.append(f"<lore_context>\n{items}\n</lore_context>")
    if memories:
        items = "\n".join("- " + m.replace("\n", "\n  ") for m in memories)
        parts.append(f"<memory_context>\n{items}\n</memory_context>")
    if history:
        parts.append("<recent_conversation>\n" + "\n".join(history) + "\n</recent_conversation>")
    parts.append(f"[{author_name}]: {clean_text}")
    return "\n\n".join(parts)


class ReplyGuard:
    """Chốt chặn đặt TRƯỚC mọi lời gọi API: giới hạn tần suất trả lời, tạm im ở kênh gửi bị 403,
    và tạm dừng toàn bộ khi gặp 429/401/Cloudflare — để bot không tự tạo thêm request lỗi."""

    def __init__(self):
        self.user_last: dict[int, float] = {}
        self.muted_until: dict[int, float] = {}
        self.sent_times: deque = deque()
        self.paused_until = 0.0

    def allow(self, user_id: int, channel_id: int) -> bool:
        now = time.monotonic()
        if now < self.paused_until:                                   # đang lùi bước toàn cục
            return False
        if now < self.muted_until.get(channel_id, 0.0):               # kênh vừa từ chối gửi tin
            return False
        if now - self.user_last.get(user_id, 0.0) < USER_COOLDOWN_SEC:  # người này gọi quá dồn dập
            return False
        while self.sent_times and now - self.sent_times[0] > RATE_WINDOW_SEC:
            self.sent_times.popleft()
        if len(self.sent_times) >= RATE_MAX_REPLIES:                # trần toàn bot
            return False
        self.user_last[user_id] = now
        self.sent_times.append(now)
        self._cleanup(now)
        return True

    def _cleanup(self, now: float):
        if len(self.user_last) > 2000:
            for uid in [u for u, t in self.user_last.items() if t < now - 3600]:
                self.user_last.pop(uid, None)
        if len(self.muted_until) > 500:
            for cid in [c for c, t in self.muted_until.items() if t <= now]:
                self.muted_until.pop(cid, None)

    def mute_channel(self, channel_id: int, seconds: float):
        self.muted_until[channel_id] = time.monotonic() + seconds
        log.warning("🔇 Tạm im lặng ở kênh %s trong %ss (Discord từ chối gửi tin).", channel_id, int(seconds))

    def pause_all(self, seconds: float):
        self.paused_until = time.monotonic() + seconds
        log.warning("⏸️ Tạm dừng mọi phản hồi trong %ss (Discord trả 429/401/Cloudflare).", int(seconds))


guard = ReplyGuard()


def _note_http_error(channel_id: int, e: discord.HTTPException) -> Optional[int]:
    """Ghi nhận lỗi HTTP từ Discord và kích hoạt chốt chặn tương ứng."""
    status = getattr(e, "status", None)
    if status in (429, 401):
        guard.pause_all(RATE_BACKOFF_SEC)
    elif status == 403:
        guard.mute_channel(channel_id, CHANNEL_MUTE_SEC)
    return status


def reply_permissions(message: discord.Message) -> tuple[bool, bool]:
    """Trả về (được_gửi_tin, được_reply_trích_dẫn). Thiếu quyền thì gửi tin/typing đều bị 403
    — mỗi lần là một request không hợp lệ — nên kiểm tra trước và im lặng bỏ qua."""
    if message.guild is None:
        return True, True  # DM
    channel = message.channel
    if isinstance(channel, discord.PartialMessageable):
        return True, True  # kênh chưa có trong cache: không đủ dữ liệu để kết luận, cứ thử (đã có chốt 403)
    me = message.guild.me
    if me is None:
        return True, True
    perms = channel.permissions_for(me)
    in_thread = isinstance(channel, discord.Thread)
    can_send = perms.view_channel and (perms.send_messages_in_threads if in_thread else perms.send_messages)
    return bool(can_send), bool(perms.read_message_history)


@asynccontextmanager
async def safe_typing(channel):
    """Hiển thị 'đang gõ...' mà không làm hỏng luồng nếu thiếu quyền."""
    cm = channel.typing()
    entered = False
    try:
        await cm.__aenter__()
        entered = True
    except discord.HTTPException as e:
        _note_http_error(channel.id, e)
    except Exception:
        pass
    try:
        yield
    finally:
        if entered:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass


async def safe_reply(message: discord.Message, text: str, can_reference: bool = True) -> bool:
    """Gửi câu trả lời. Trả về True nếu gửi được.
    - 403/429/401: KHÔNG thử lại (thử lại chỉ tạo thêm request lỗi) mà kích hoạt chốt chặn.
    - Lỗi khác (vd tin gốc vừa bị xóa): thử gửi thẳng vào kênh đúng 1 lần."""
    channel_id = message.channel.id
    try:
        if can_reference:
            await message.reply(text, mention_author=False)
        else:
            await message.channel.send(text)
        return True
    except discord.HTTPException as e:
        status = _note_http_error(channel_id, e)
        log.warning("Không gửi được tin nhắn (HTTP %s): %s", status, str(e)[:150])
        if status in (401, 403, 429) or not can_reference:
            return False

    try:  # lỗi kiểu 400 do không trích dẫn được tin gốc
        await message.channel.send(text)
        return True
    except discord.HTTPException as e2:
        _note_http_error(channel_id, e2)
        log.warning("Gửi thẳng vào kênh cũng thất bại: %s", str(e2)[:150])
        return False


@bot.event
async def on_ready():
    global has_synced_on_startup
    log.info("Đã đăng nhập: %s", bot.user.name)
    if not has_synced_on_startup:
        try:
            synced = await bot.tree.sync()
            log.info("Đã đồng bộ %d lệnh Slash.", len(synced))
            has_synced_on_startup = True
        except Exception as e:
            log.warning("Lỗi đồng bộ: %s", e)

    spawn(_get_global_items())  # làm nóng bộ nhớ đệm Global Memory để tin nhắn đầu tiên không phải chờ
    spawn(refresh_personas(force=True))  # và bộ nhớ đệm chỉ dẫn riêng

    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="Bước vào Wonderland? | Gọi 'alise' nhé!",
        )
    )


async def handle_alice_message(message: discord.Message):
    # Các chốt chặn rẻ nhất đặt ĐẦU TIÊN — lúc này chưa có lời gọi API nào được thực hiện
    can_send, can_reference = reply_permissions(message)
    if not can_send:
        return  # thiếu quyền: im lặng, không tạo request 403
    if not guard.allow(message.author.id, message.channel.id):
        return

    # Loại bỏ cả hai dạng mention: <@id> và <@!id>
    clean_text = message.content or ""
    if bot.user:
        clean_text = re.sub(rf"<@!?{bot.user.id}>", "", clean_text)
    clean_text = clean_text.strip()
    if not clean_text:
        clean_text = "Chào Alice nha!"
    if len(clean_text) > MAX_INPUT_CHARS:
        clean_text = clean_text[:MAX_INPUT_CHARS] + "…"

    author_name = sanitize_name(message.author.display_name)
    is_owner = await is_owner_user(message.author)  # so khớp bằng USER ID
    await refresh_personas()  # chỉ tốn một truy vấn nhỏ mỗi PERSONA_CACHE_TTL_SEC giây
    # Chỉ dẫn riêng (nếu có) được tra theo USER ID — không bao giờ theo tên hiển thị
    role_instruction = build_role_instruction(
        user_id=message.author.id,
        author_name=author_name,
        is_owner=is_owner,
    )

    user_id = str(message.author.id)
    channel_id = str(message.channel.id)
    mem_key = get_history_key(channel_id, user_id)
    full_history = list(get_history(mem_key))
    history = full_history[-HISTORY_PROMPT_LINES:]

    async with safe_typing(message.channel):
        try:
            # Ký ức dài hạn — lỗi ở đây không được làm hỏng câu trả lời
            memories: list[str] = []
            lore: list[str] = []
            try:
                query_vector = await get_embedding(clean_text, task_type="RETRIEVAL_QUERY")  # nhúng 1 lần, dùng cho cả hai loại ký ức
                min_date = (
                    datetime.now(timezone.utc) - timedelta(days=MEMORY_WINDOW_DAYS)
                    if MEMORY_WINDOW_DAYS > 0 else None
                )
                retrieved = await retrieve_relevant_contexts(
                    user_id=user_id,
                    query_text=clean_text,
                    query_vector=query_vector,
                    channel_id=channel_id,
                    min_date=min_date,
                    top_k=MEMORY_TOP_K + 2,
                    threshold=MEMORY_THRESHOLD,
                )
                # Bỏ ký ức trùng với lịch sử ngắn hạn (vừa mới lưu xong ở lượt trước)
                recent_lines = set(full_history)
                memories = [m for m in retrieved if m.split("\n", 1)[0] not in recent_lines][:MEMORY_TOP_K]
                lore_hits = await retrieve_global_scored(query_vector)  # Global Memory: dùng chung cho mọi người dùng/kênh
                lore = [text for _, text, _ in lore_hits]
                if memories or lore_hits:
                    log.info("🧠 Truy xuất: %d ký ức cá nhân, %d đoạn global%s.", len(memories), len(lore_hits),
                             f" (cao nhất {lore_hits[0][0]:.2f} · {lore_hits[0][2]})" if lore_hits else "")
            except Exception as e:
                log.warning("Lỗi truy xuất ký ức: %s", e)

            prompt = build_prompt(author_name, clean_text, history, memories, lore)
            reply_text, ok = await ask_gemini(
                contents=prompt,
                system_instruction=ALICE_SYSTEM_PROMPT + role_instruction,
                temperature=0.9,
            )
            reply_text = truncate_for_discord(reply_text)
            sent = await safe_reply(message, reply_text, can_reference)

            # Chỉ lưu khi là câu trả lời thật VÀ đã gửi được (không lưu câu dự phòng/lỗi để khỏi làm bẩn ký ức)
            if ok and sent:
                append_history(mem_key, f"{author_name}: {clean_text}", f"Alice: {reply_text}")
                spawn(store_context_memory(
                    user_id=user_id,
                    channel_id=channel_id,
                    author_name=author_name,
                    user_text=clean_text,
                    reply_text=reply_text,
                ))
            elif not ok:
                log.info("💾 Đây là câu trả lời dự phòng/lỗi nên không lưu ký ức lượt này.")

        except Exception as e:
            log.exception("Lỗi phản hồi tin nhắn")
            if _classify_gemini_error(e) == "quota":
                await safe_reply(message, MSG_QUOTA_LONG, can_reference)
            else:
                await safe_reply(message, MSG_ERROR, can_reference)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or message.is_system():  # bỏ qua bot (gồm chính nó) và tin hệ thống
        return

    content = message.content or ""

    is_reply_to_alice = False
    ref = message.reference
    if ref and isinstance(ref.resolved, discord.Message) and bot.user and ref.resolved.author.id == bot.user.id:
        is_reply_to_alice = True

    is_mentioned = bot.user is not None and bot.user in message.mentions
    has_alice_name = bool(ALICE_NAME_RE.search(content))

    if is_mentioned or has_alice_name or is_reply_to_alice:
        await handle_alice_message(message)

    await bot.process_commands(message)


# ==============================================================================
# LỆNH SLASH
# ==============================================================================
@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        msg = "Lệnh này chỉ dành cho chủ nhân của Alice thôi nha~"
    else:
        log.error("Lỗi lệnh slash: %s", error, exc_info=error)
        msg = "Ôi, có chuyện gì đó trục trặc rồi… cậu thử lại sau nhé?"
    try:
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)
    except discord.HTTPException:
        pass


@bot.tree.command(name="clearmem", description="Xóa ký ức trò chuyện của Alice với cậu")
@app_commands.describe(xoa_ca_kho_nho="Nếu chọn True, sẽ xóa sạch cả ký ức vĩnh viễn trong MongoDB")
async def slash_clear_memory(interaction: discord.Interaction, xoa_ca_kho_nho: bool = False):
    await interaction.response.defer()  # thao tác DB có thể vượt 3 giây
    reset_memory(interaction.channel_id, interaction.user.id)
    author_name = discord.utils.escape_markdown(sanitize_name(interaction.user.display_name))
    is_owner = await is_owner_user(interaction.user)

    extra_msg = ""
    if xoa_ca_kho_nho:
        ok = await delete_user_memories(str(interaction.user.id))
        if ok:
            extra_msg = "\n*(Toàn bộ kho lưu trữ ký ức vĩnh viễn của cậu cũng đã được dọn sạch.)*"
        else:
            extra_msg = "\n*(Alice chưa dọn hết được kho ký ức vĩnh viễn… cậu thử lại sau nhé.)*"

    if is_owner:
        desc = f"Cậu mới nói cái gì á? Uể? Mình nhớ là cậu có nói gì mà ta?{extra_msg}"
    else:
        desc = f"Hmmm cậu mới kể gì cho tớ nghe vậy?, {author_name}. Thôi kệ đi cùng mình thưởng trà nào?!{extra_msg}"

    embed = discord.Embed(title="🧹 Làm Mới Trạng Thái", description=desc, color=0x2C2F33)
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="memories", description="Xem Alice đang lưu giữ bao nhiêu ký ức về bạn")
async def slash_view_memories(interaction: discord.Interaction):
    """Kiểm tra số lượng ký ức được lưu trong Vector Store của người dùng."""
    await interaction.response.defer(ephemeral=True)
    mongo_count, sqlite_count = await count_user_memories(str(interaction.user.id))

    total = sqlite_count + (mongo_count or 0)
    places = []
    if mongo_count is not None:
        places.append(f"MongoDB Atlas Vector Store ☁️: {mongo_count}")
    if sqlite_count or mongo_count is None:
        places.append(f"SQLite Cục bộ: {sqlite_count}")

    name = discord.utils.escape_markdown(sanitize_name(interaction.user.display_name))
    global_count = len(await _get_global_items())
    global_line = f"\n\n🌐 Tri thức toàn cục (dùng chung cho mọi người): **{global_count}** đoạn." if global_count else ""
    embed = discord.Embed(
        title="📖 Cuốn Sổ Ký Ức Của Alice",
        description=f"Alice hiện đang cất giữ **{total}** mảnh ký ức về **{name}**.\n\n*Nơi lưu trữ:*\n" + "\n".join(f"`{p}`" for p in places) + global_line,
        color=0xE0A96D,
    )
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(name="globalmem_add", description="(Chủ nhân) Nạp file tri thức vào Global Memory để Alice nhớ với mọi người")
@app_commands.describe(
    file="File văn bản UTF-8 (.txt, .md, .csv, .json, .log)",
    name="Tên nguồn để quản lý (mặc định là tên file). Nạp lại cùng tên sẽ thay thế bản cũ",
)
@app_commands.check(_owner_only)
async def slash_globalmem_add(interaction: discord.Interaction, file: discord.Attachment, name: Optional[str] = None):
    await interaction.response.defer(ephemeral=True)

    async def say(msg: str):
        try:
            await interaction.edit_original_response(content=msg)
        except discord.HTTPException as e:
            log.warning("Không cập nhật được tin nhắn tiến trình: %s", e)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in GLOBAL_ALLOWED_EXT:
        await say(f"❌ Chưa hỗ trợ định dạng `{ext or '(không có)'}`. Hãy dùng: {', '.join(GLOBAL_ALLOWED_EXT)} "
                  f"(PDF/Word hãy chuyển sang .txt hoặc .md trước).")
        return
    if file.size > GLOBAL_MAX_FILE_BYTES:
        await say(f"❌ File quá lớn ({file.size // 1024} KB). Giới hạn hiện tại là {GLOBAL_MAX_FILE_BYTES // 1024} KB.")
        return

    source = sanitize_source(name or file.filename)
    lock = _get_upload_lock()
    if lock.locked():
        await say("⏳ Alice đang nạp một file khác. Cậu chờ xong rồi thử lại nhé.")
        return

    async with lock:
        try:
            raw = await file.read()
            try:
                text = decode_upload(raw)
            except ValueError as e:
                reason = "có vẻ là file nhị phân" if str(e) == "binary" else "không phải mã hóa UTF-8 (hãy lưu lại ở dạng UTF-8)"
                await say(f"❌ Không đọc được file: {reason}.")
                return

            chunks = chunk_text(text, GLOBAL_CHUNK_CHARS, GLOBAL_CHUNK_OVERLAP)
            if not chunks:
                await say("❌ File không có nội dung văn bản.")
                return
            if len(chunks) > GLOBAL_MAX_CHUNKS:
                await say(f"❌ File được tách thành {len(chunks)} đoạn, vượt giới hạn {GLOBAL_MAX_CHUNKS} đoạn mỗi file. "
                          f"Hãy chia nhỏ file hoặc tăng `GLOBAL_MAX_CHUNKS`.")
                return

            others = sum(r["chunks"] for r in await list_global_sources() if r["source"] != source)
            if others + len(chunks) > GLOBAL_MAX_TOTAL_CHUNKS:
                await say(f"❌ Global Memory đã có {others} đoạn; thêm {len(chunks)} đoạn sẽ vượt trần "
                          f"{GLOBAL_MAX_TOTAL_CHUNKS} đoạn (bảo vệ bộ nhớ RAM). Hãy xóa bớt nguồn cũ bằng "
                          f"`/globalmem_delete` hoặc tăng `GLOBAL_MAX_TOTAL_CHUNKS`.")
                return

            deadline = time.monotonic() + GLOBAL_UPLOAD_DEADLINE_SEC
            await say(f"⏳ Đã tách **{len(chunks)}** đoạn từ `{source}`. Đang nhúng (embedding)…")
            last_edit = [time.monotonic()]

            async def progress(done: int, total: int):
                if time.monotonic() - last_edit[0] >= 3 and done < total:
                    last_edit[0] = time.monotonic()
                    await say(f"⏳ Đang nhúng {done}/{total} đoạn từ `{source}`…")

            vectors = await embed_chunks_with_retry(chunks, progress, deadline)
            if len({len(v) for v in vectors}) != 1:
                raise ValueError("Các vector nhận về có số chiều không đồng nhất")

            inserted, removed, backend = await add_global_memory(source, chunks, vectors)
            where = "MongoDB" if backend == "mongo" else "SQLite cục bộ"
            replaced = f"\nĐã thay thế bản cũ cùng tên ({removed} đoạn)." if removed else ""
            warn = "" if backend == "mongo" else (
                "\n⚠️ MongoDB chưa kết nối nên dữ liệu đang nằm trong SQLite cục bộ — có thể mất khi host "
                "khởi động lại hoặc triển khai lại."
            )
            await say(f"✅ Đã nạp **{inserted}** đoạn từ `{source}` vào Global Memory ({where}).{replaced}{warn}\n"
                      f"Alice sẽ dùng nội dung này khi trò chuyện với mọi người, ở mọi kênh. "
                      f"Dùng `/globalmem_test` để thử truy xuất.")
            log.info("🌐 Global Memory: nạp %d đoạn từ %r vào %s (thay %d đoạn cũ).", inserted, source, backend, removed)
        except Exception as e:
            log.exception("Lỗi nạp Global Memory")
            await say(f"❌ Nạp thất bại: {str(e)[:300]}\nDữ liệu cũ (nếu có) vẫn giữ nguyên.")


@bot.tree.command(name="globalmem_list", description="(Chủ nhân) Xem các nguồn đang có trong Global Memory")
@app_commands.check(_owner_only)
async def slash_globalmem_list(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        rows = await list_global_sources()
    except Exception as e:
        log.warning("Lỗi liệt kê Global Memory: %s", e)
        await interaction.followup.send("❌ Không đọc được danh sách Global Memory.", ephemeral=True)
        return

    if not rows:
        desc = "Global Memory đang trống. Dùng `/globalmem_add` để nạp file."
    else:
        lines = []
        for r in rows[:25]:
            when = r["updated"].strftime("%Y-%m-%d %H:%M") + " UTC" if r["updated"] else "—"
            tags = "/".join(sorted(r["backends"])) + (" · bản cũ" if r["legacy"] else "")
            lines.append(f"`{r['source']}` — {r['chunks']} đoạn — {when} — {tags}")
        desc = f"Tổng **{sum(r['chunks'] for r in rows)}** đoạn từ **{len(rows)}** nguồn:\n" + "\n".join(lines)
        if len(rows) > 25:
            desc += f"\n… và {len(rows) - 25} nguồn khác."
    embed = discord.Embed(title="🌐 Global Memory", description=desc[:4000], color=0xE0A96D)
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(name="globalmem_delete", description="(Chủ nhân) Xóa một nguồn khỏi Global Memory")
@app_commands.describe(name="Tên nguồn (xem bằng /globalmem_list)")
@app_commands.check(_owner_only)
async def slash_globalmem_delete(interaction: discord.Interaction, name: str):
    await interaction.response.defer(ephemeral=True)
    source = sanitize_source(name)
    try:
        deleted = await delete_global_source(source)
    except Exception as e:
        log.warning("Lỗi xóa Global Memory: %s", e)
        await interaction.followup.send("❌ Xóa thất bại, cậu thử lại sau nhé.", ephemeral=True)
        return
    if deleted:
        await interaction.followup.send(f"🗑️ Đã xóa {deleted} đoạn của nguồn `{source}`.", ephemeral=True)
    else:
        await interaction.followup.send(
            f"Không tìm thấy nguồn `{source}`. Xem tên chính xác bằng `/globalmem_list`.", ephemeral=True)


@bot.tree.command(name="globalmem_clear", description="(Chủ nhân) Xóa TOÀN BỘ Global Memory (không thể hoàn tác)")
@app_commands.describe(xac_nhan="Chọn True để xác nhận xóa sạch")
@app_commands.check(_owner_only)
async def slash_globalmem_clear(interaction: discord.Interaction, xac_nhan: bool = False):
    await interaction.response.defer(ephemeral=True)
    if not xac_nhan:
        await interaction.followup.send(
            "⚠️ Lệnh này xóa TOÀN BỘ Global Memory và không thể hoàn tác. "
            "Hãy chạy lại và đặt `xac_nhan` thành True nếu cậu chắc chắn.", ephemeral=True)
        return
    try:
        deleted = await clear_global_memory()
    except Exception as e:
        log.warning("Lỗi xóa toàn bộ Global Memory: %s", e)
        await interaction.followup.send("❌ Xóa thất bại, cậu thử lại sau nhé.", ephemeral=True)
        return
    await interaction.followup.send(f"🗑️ Đã xóa sạch Global Memory ({deleted} đoạn).", ephemeral=True)


@bot.tree.command(name="globalmem_test", description="(Chủ nhân) Thử truy xuất Global Memory và xem điểm tương đồng")
@app_commands.describe(cau_hoi="Câu muốn thử, ví dụ một câu mà người dùng có thể hỏi Alice")
@app_commands.check(_owner_only)
async def slash_globalmem_test(interaction: discord.Interaction, cau_hoi: str):
    await interaction.response.defer(ephemeral=True)
    cau_hoi = cau_hoi.strip()[:MAX_INPUT_CHARS]
    if not cau_hoi:
        await interaction.followup.send("❌ Cậu hãy nhập một câu để thử.", ephemeral=True)
        return
    query_vector = await get_embedding(cau_hoi, task_type="RETRIEVAL_QUERY")
    if not query_vector:
        await interaction.followup.send("❌ Không tạo được embedding cho câu này (xem log).", ephemeral=True)
        return
    items = await _get_global_items()
    if not items:
        await interaction.followup.send("Global Memory đang trống. Dùng `/globalmem_add` để nạp file.", ephemeral=True)
        return

    top, skipped = await asyncio.to_thread(_score_global, query_vector, items, 5, None)
    lines = []
    for sim, text_val, source in top:
        mark = "✅" if sim >= GLOBAL_THRESHOLD else "❌"
        preview = discord.utils.escape_markdown(text_val.replace("\n", " "))[:90]
        lines.append(f"{mark} `{sim:.3f}` · `{source}` — {preview}…")
    desc = (f"Ngưỡng hiện tại: `{GLOBAL_THRESHOLD}` (✅ = đủ điểm để Alice dùng, tối đa {GLOBAL_TOP_K} đoạn).\n\n"
            + ("\n".join(lines) if lines else "Không có đoạn nào cùng số chiều vector với câu hỏi."))
    if skipped:
        desc += f"\n\n⚠️ {skipped} đoạn khác số chiều vector nên bị bỏ qua — hãy nạp lại bằng `/globalmem_add`."
    embed = discord.Embed(title="🧪 Thử truy xuất Global Memory", description=desc[:4000], color=0xE0A96D)
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(name="persona_set", description="(Chủ nhân) Đặt chỉ dẫn riêng cho Alice khi nói chuyện với một người dùng")
@app_commands.describe(
    user="Người dùng (nhận diện bằng ID, không phụ thuộc tên hiển thị)",
    chi_dan="Nội dung chỉ dẫn (một dòng). Chỉ dẫn nhiều dòng thì dùng tham số file",
    file="File .txt/.md UTF-8 chứa chỉ dẫn",
    ten="Tên gọi dùng trong chỉ dẫn (mặc định là tên hiển thị hiện tại)",
)
@app_commands.check(_owner_only)
async def slash_persona_set(interaction: discord.Interaction, user: discord.User, chi_dan: Optional[str] = None,
                            file: Optional[discord.Attachment] = None, ten: Optional[str] = None):
    await interaction.response.defer(ephemeral=True)

    async def say(msg: str):
        await interaction.followup.send(msg, ephemeral=True)

    if getattr(user, "bot", False):
        await say("❌ Không đặt chỉ dẫn riêng cho tài khoản bot (Alice luôn bỏ qua tin nhắn của bot).")
        return
    if bool(chi_dan and chi_dan.strip()) == bool(file):
        await say("❌ Hãy cung cấp đúng MỘT trong hai: `chi_dan` hoặc `file`.")
        return

    if file:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in (".txt", ".md", ".markdown"):
            await say(f"❌ Chỉ nhận file .txt hoặc .md (nhận được `{ext or '(không có)'}`).")
            return
        if file.size > PERSONA_MAX_CHARS * 4 + 1024:
            await say(f"❌ File quá lớn ({file.size // 1024} KB) so với giới hạn {PERSONA_MAX_CHARS} ký tự.")
            return
        try:
            raw = decode_upload(await file.read())
        except ValueError as e:
            await say("❌ Không đọc được file: " + ("có vẻ là file nhị phân." if str(e) == "binary" else "không phải mã hóa UTF-8."))
            return
    else:
        raw = chi_dan

    text = clean_persona_text(raw)
    if not text:
        await say("❌ Chỉ dẫn đang trống.")
        return
    if len(text) > PERSONA_MAX_CHARS:
        await say(f"❌ Chỉ dẫn dài {len(text)} ký tự, vượt giới hạn {PERSONA_MAX_CHARS}. Hãy rút gọn hoặc tăng `PERSONA_MAX_CHARS`.")
        return

    label = sanitize_name(ten or user.display_name, limit=40)
    try:
        backend = await set_persona(user.id, label, text)
    except Exception as e:
        log.warning("Lỗi lưu chỉ dẫn riêng: %s", e)
        await say("❌ Lưu thất bại, cậu thử lại sau nhé.")
        return
    where = "MongoDB" if backend == "mongo" else "SQLite cục bộ"
    warn = "" if backend == "mongo" else "\n⚠️ MongoDB chưa kết nối nên chỉ dẫn đang nằm trong SQLite cục bộ — có thể mất khi host khởi động lại."
    await say(f"✅ Đã đặt chỉ dẫn riêng cho **{discord.utils.escape_markdown(label)}** (`{user.id}`) — "
              f"{len(text)} ký tự, lưu ở {where}. Có hiệu lực từ tin nhắn kế tiếp.{warn}")
    log.info("🎭 Persona: đặt chỉ dẫn riêng cho %s (%d ký tự, %s).", user.id, len(text), backend)


@bot.tree.command(name="persona_list", description="(Chủ nhân) Xem các chỉ dẫn riêng đang áp dụng theo người dùng")
@app_commands.check(_owner_only)
async def slash_persona_list(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    await refresh_personas(force=True)
    rows = list_personas()
    if not rows:
        desc = "Chưa có chỉ dẫn riêng nào. Dùng `/persona_set` để thêm."
    else:
        lines = [f"`{r['user_id']}` · **{discord.utils.escape_markdown(r['label'])}** · {r['chars']} ký tự · {r['kind']}"
                 for r in rows[:25]]
        desc = f"**{len(rows)}** người dùng có chỉ dẫn riêng:\n" + "\n".join(lines)
        if len(rows) > 25:
            desc += f"\n… và {len(rows) - 25} người khác."
    embed = discord.Embed(title="🎭 Chỉ dẫn riêng theo người dùng", description=desc[:4000], color=0xE0A96D)
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(name="persona_show", description="(Chủ nhân) Xem toàn văn chỉ dẫn riêng của một người dùng")
@app_commands.describe(user="Người dùng cần xem")
@app_commands.check(_owner_only)
async def slash_persona_show(interaction: discord.Interaction, user: discord.User):
    await interaction.response.defer(ephemeral=True)
    await refresh_personas(force=True)
    text = get_custom_prompt_by_id(user.id)
    if not text:
        await interaction.followup.send("Người này chưa có chỉ dẫn riêng (Alice dùng chỉ dẫn mặc định cho người thường).", ephemeral=True)
        return
    label = discord.utils.escape_markdown(get_custom_label(user.id) or sanitize_name(user.display_name))
    safe = text.replace("```", "ˋˋˋ")
    await interaction.followup.send(f"🎭 **{label}** (`{user.id}`) — {persona_kind(user.id)}:\n```\n{safe}\n```", ephemeral=True)


@bot.tree.command(name="persona_delete", description="(Chủ nhân) Xóa chỉ dẫn riêng của một người dùng (hoặc đưa về mặc định)")
@app_commands.describe(user="Người dùng cần xóa chỉ dẫn riêng")
@app_commands.check(_owner_only)
async def slash_persona_delete(interaction: discord.Interaction, user: discord.User):
    await interaction.response.defer(ephemeral=True)
    try:
        removed = await delete_persona(user.id)
    except Exception as e:
        log.warning("Lỗi xóa chỉ dẫn riêng: %s", e)
        await interaction.followup.send("❌ Xóa thất bại, cậu thử lại sau nhé.", ephemeral=True)
        return
    builtin = str(user.id) in BUILTIN_PERSONAS
    if removed and builtin:
        msg = "🗑️ Đã xóa phần ghi đè; người này quay về chỉ dẫn mặc định tích hợp sẵn."
    elif removed:
        msg = "🗑️ Đã xóa chỉ dẫn riêng của người này."
    elif builtin:
        msg = "Người này đang dùng chỉ dẫn mặc định tích hợp sẵn và chưa có phần ghi đè nào để xóa. Muốn đổi nội dung hãy dùng `/persona_set`."
    else:
        msg = "Người này chưa có chỉ dẫn riêng."
    await interaction.followup.send(msg, ephemeral=True)


@bot.tree.command(name="sync", description="Đồng bộ lệnh Slash (chỉ chủ nhân)")
@app_commands.check(_owner_only)
async def slash_sync_commands(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        # Dọn lệnh "rác" gắn riêng cho server hiện tại (chỉ khi đang ở trong server —
        # nếu guild=None thì clear_commands sẽ xóa nhầm lệnh GLOBAL)
        if interaction.guild is not None:
            bot.tree.clear_commands(guild=interaction.guild)
            await bot.tree.sync(guild=interaction.guild)
        synced = await bot.tree.sync()
        await interaction.followup.send(
            f"Đã dọn sạch lệnh rác và đồng bộ {len(synced)} lệnh. Vui lòng bấm Ctrl+R trên Discord để cập nhật giao diện."
        )
    except Exception as e:
        log.exception("Lỗi /sync")
        await interaction.followup.send(f"Lỗi: {e}")


# ==============================================================================
# ĐIỂM KHỞI CHẠY
# ==============================================================================
def run_bot():
    """Chạy bot; nếu đăng nhập bị Discord/Cloudflare chặn (429) hoặc token sai (401) thì CHỜ LÂU trước khi thoát.
    Lý do: nền tảng host (Render...) sẽ tự khởi động lại tiến trình ngay khi nó thoát, mỗi lần khởi động lại
    là thêm một request lỗi gửi vào IP đang bị chặn. Web health-check vẫn chạy nên host không coi là sập."""
    try:
        bot.run(DISCORD_TOKEN, log_handler=None)  # log đã cấu hình ở main()
    except discord.LoginFailure:
        log.error("🔑 Token Discord không hợp lệ (401). Kiểm tra lại DISCORD_TOKEN. Chờ %ss rồi mới thoát "
                  "để host không khởi động lại liên tục.", LOGIN_BACKOFF_SEC)
        time.sleep(LOGIN_BACKOFF_SEC)
        raise SystemExit(1)
    except discord.HTTPException as e:
        if getattr(e, "status", None) == 429:
            log.error("🚫 Discord/Cloudflare đang chặn IP của host (429) khi đăng nhập. Thường do IP dùng chung "
                      "với bot khác. Chờ %ss rồi mới thoát; nếu lặp lại, hãy đặt DISCORD_PROXY hoặc đổi host.",
                      LOGIN_BACKOFF_SEC)
            time.sleep(LOGIN_BACKOFF_SEC)
            raise SystemExit(1)
        raise


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    missing = [n for n, v in (("DISCORD_TOKEN", DISCORD_TOKEN), ("GEMINI_API_KEY", GEMINI_API_KEY)) if not v]
    if missing:
        raise SystemExit(f"Thiếu biến môi trường bắt buộc: {', '.join(missing)}")

    # Mở cổng web NGAY (trước khi kết nối DB/Discord) để Render thấy cổng đang mở và không báo "No open ports detected"
    threading.Thread(target=run_web_server, daemon=True).start()
    init_sqlite()
    init_mongo()

    if not OWNER_ID:
        log.warning("OWNER_ID chưa được đặt — dùng chủ sở hữu ứng dụng Discord làm chủ nhân. "
                    "Nên đặt OWNER_ID=<user id của bạn> để chắc chắn.")

    if DISCORD_PROXY:
        log.info("🌐 Đang kết nối Discord qua proxy đã cấu hình.")  # không in URL vì có thể chứa mật khẩu

    run_bot()


if __name__ == "__main__":
    main()
