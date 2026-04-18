"""멀티세션 RAG 챗봇 Streamlit 앱 (프롬프트: 7.MultiService/prompts/멀티세션 ref.txt)."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _REPO_ROOT / ".env"
_LOG_DIR = _REPO_ROOT / "logs"
_LOGO_PATH = _REPO_ROOT / "logo.png"

_ENV_KEYS = ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY")


def _secret_value(sec: Any, key: str) -> str | None:
    """st.secrets 최상위 또는 [env] 섹션에서 문자열 값 조회."""
    try:
        v = sec[key]
    except KeyError:
        v = None
    if (v is None or (isinstance(v, str) and not v.strip())) and "env" in sec:
        try:
            block = sec["env"]
            if isinstance(block, dict):
                v = block.get(key)
        except (KeyError, TypeError):
            pass
    if v is None:
        return None
    text = str(v).strip()
    return text or None


def _hydrate_env_from_streamlit_secrets() -> None:
    """Streamlit Cloud App Secrets 등을 os.environ에 반영(기존 env 덮어씀)."""
    try:
        sec = st.secrets
    except FileNotFoundError:
        return
    for key in _ENV_KEYS:
        val = _secret_value(sec, key)
        if val:
            os.environ[key] = val


load_dotenv(_ENV_PATH)
_hydrate_env_from_streamlit_secrets()

_logger = logging.getLogger("multi_session_rag")
_logging_done = False

MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
VECTOR_BATCH = 10
RETRIEVE_K = 8

ANSWER_SYSTEM = (
    "당신은 문서 기반 안내 챗봇입니다. "
    "답변은 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. "
    "주요 주제는 #, 세부는 ##, 구체적 설명은 ### 로 구분합니다. "
    "존대말로 완전한 문장으로 서술하세요. "
    "구분선(---, ===, ___)과 취소선(~~)은 사용하지 마세요. "
    "참조 표시나 출처 문구는 넣지 마세요."
)


def _configure_logging() -> None:
    global _logging_done
    if _logging_done:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    log_path = _LOG_DIR / f"multi_session_rag_{day}.log"
    _logger.handlers.clear()
    _logger.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    _logger.addHandler(fh)
    _logger.addHandler(ch)
    _logger.propagate = False
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _logging_done = True


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~.+?~~", "", text, flags=re.DOTALL)
    text = re.sub(r"^[\t ]*[=\-_]{3,}[\t ]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _env_status() -> tuple[str | None, str | None, str | None]:
    o = (os.getenv("OPENAI_API_KEY") or "").strip() or None
    u = (os.getenv("SUPABASE_URL") or "").strip() or None
    k = (os.getenv("SUPABASE_ANON_KEY") or "").strip() or None
    return o, u, k


def get_supabase() -> Client | None:
    _, url, key = _env_status()
    if not url or not key:
        return None
    return create_client(url, key)


def get_llm(temperature: float = 0.7) -> ChatOpenAI | None:
    openai_key, _, _ = _env_status()
    if not openai_key:
        return None
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature, api_key=openai_key)


def get_embeddings() -> OpenAIEmbeddings | None:
    openai_key, _, _ = _env_status()
    if not openai_key:
        return None
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=openai_key,
        chunk_size=30,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _parse_embedding(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            try:
                return [float(x) for x in json.loads(s)]
            except json.JSONDecodeError:
                return None
    return None


def fetch_sessions(sb: Client) -> list[dict[str, Any]]:
    res = (
        sb.table("chat_sessions")
        .select("id,title,updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return list(res.data or [])


def create_chat_session(sb: Client, title: str) -> str:
    ins = sb.table("chat_sessions").insert({"title": title}).execute()
    row = (ins.data or [{}])[0]
    return str(row["id"])


def update_session_title(sb: Client, session_id: str, title: str) -> None:
    sb.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()


def replace_session_messages(sb: Client, session_id: str, chat_history: list[dict[str, str]]) -> None:
    sb.table("session_messages").delete().eq("session_id", session_id).execute()
    rows: list[dict[str, Any]] = []
    for i, m in enumerate(chat_history):
        rows.append(
            {
                "session_id": session_id,
                "role": m["role"],
                "content": m["content"],
                "sort_order": i,
            }
        )
    if rows:
        sb.table("session_messages").insert(rows).execute()


def load_session_messages(sb: Client, session_id: str) -> list[dict[str, str]]:
    res = (
        sb.table("session_messages")
        .select("role,content,sort_order")
        .eq("session_id", session_id)
        .order("sort_order")
        .execute()
    )
    out: list[dict[str, str]] = []
    for row in res.data or []:
        out.append({"role": str(row["role"]), "content": str(row["content"])})
    return out


def insert_vector_batch(
    sb: Client,
    session_id: str,
    items: list[tuple[str, str, list[float]]],
) -> None:
    """(file_name, content, embedding) 배치 INSERT."""
    payload = [
        {
            "session_id": session_id,
            "file_name": fn,
            "content": text,
            "embedding": emb,
            "metadata": {},
        }
        for fn, text, emb in items
    ]
    for i in range(0, len(payload), VECTOR_BATCH):
        sb.table("vector_documents").insert(payload[i : i + VECTOR_BATCH]).execute()


def copy_vectors_to_session(sb: Client, source_session_id: str, target_session_id: str) -> None:
    res = (
        sb.table("vector_documents")
        .select("content,embedding,file_name,metadata")
        .eq("session_id", source_session_id)
        .execute()
    )
    batch: list[tuple[str, str, list[float]]] = []
    for row in res.data or []:
        emb = _parse_embedding(row.get("embedding"))
        if emb is None:
            continue
        fn = str(row.get("file_name") or "document.pdf")
        batch.append((fn, str(row.get("content") or ""), emb))
    insert_vector_batch(sb, target_session_id, batch)


def delete_session(sb: Client, session_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).execute()


def list_vector_filenames(sb: Client, session_id: str | None) -> list[str]:
    if not session_id:
        return []
    res = sb.table("vector_documents").select("file_name").eq("session_id", session_id).execute()
    names: list[str] = []
    for row in res.data or []:
        fn = str(row.get("file_name") or "")
        if fn and fn not in names:
            names.append(fn)
    return names


def retrieve_with_rpc(
    sb: Client,
    emb: OpenAIEmbeddings,
    query: str,
    session_id: str,
    k: int = RETRIEVE_K,
) -> list[Document]:
    qv = emb.embed_query(query)
    try:
        res = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": qv,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in res.data or []:
            docs.append(
                Document(
                    page_content=str(row.get("content") or ""),
                    metadata={
                        "file_name": row.get("file_name"),
                        "session_id": str(row.get("session_id")),
                    },
                )
            )
        return docs
    except Exception as exc:  # noqa: BLE001
        _logger.warning("RPC 검색 실패, 로컬 필터 폴백: %s", exc)
        return _retrieve_fallback(sb, qv, session_id, k)


def _retrieve_fallback(
    sb: Client,
    query_embedding: list[float],
    session_id: str,
    k: int,
) -> list[Document]:
    res = (
        sb.table("vector_documents")
        .select("content,file_name,session_id,embedding")
        .eq("session_id", session_id)
        .execute()
    )
    scored: list[tuple[float, Document]] = []
    for row in res.data or []:
        ev = _parse_embedding(row.get("embedding"))
        if ev is None:
            continue
        sim = _cosine(query_embedding, ev)
        scored.append(
            (
                sim,
                Document(
                    page_content=str(row.get("content") or ""),
                    metadata={
                        "file_name": row.get("file_name"),
                        "session_id": str(row.get("session_id")),
                    },
                ),
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]


def generate_short_title(user_q: str, answer: str, api_key: str) -> str:
    prompt = (
        "다음은 사용자의 첫 질문과 챗봇의 첫 답변입니다. "
        "이 대화를 대표하는 한글 세션 제목을 28자 이내로 한 줄만 출력하세요. "
        "따옴표나 부가 설명 없이 제목만 쓰세요.\n\n"
        f"[질문]\n{user_q[:2500]}\n\n[답변]\n{answer[:2500]}"
    )
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.35, api_key=api_key)
    try:
        r = llm.invoke([HumanMessage(content=prompt)])
        t = str(r.content).strip().splitlines()[0].strip()
        return t[:80] if t else "새 세션"
    except Exception as exc:  # noqa: BLE001
        _logger.warning("제목 생성 실패: %s", exc)
        return "새 세션"


def _followup_block(user_q: str, answer: str, openai_key: str, llm: ChatOpenAI | None) -> str:
    prompt = (
        "다음은 사용자 질문과 챗봇 답변입니다.\n"
        "이어서 물어보기 좋은 후속 질문을 정확히 3개만 한글로 만드세요.\n"
        "형식은 반드시:\n1. 첫째 질문\n2. 둘째 질문\n3. 셋째 질문\n"
        "질문만 출력하고 다른 문장은 쓰지 마세요.\n\n"
        f"[질문]\n{user_q}\n\n[답변]\n{answer[:8000]}"
    )
    caller: Any = llm
    if caller is None:
        caller = ChatOpenAI(model=MODEL_NAME, temperature=0.4, api_key=openai_key)
    try:
        r = caller.invoke([HumanMessage(content=prompt)])
        body = str(r.content).strip()
    except Exception as exc:  # noqa: BLE001
        _logger.warning("후속 질문 생성 실패: %s", exc)
        body = (
            "1. 문서의 핵심 요지를 더 설명해 주세요.\n"
            "2. 관련 근거나 예시가 더 있나요?\n"
            "3. 실무에 적용하려면 어떻게 하면 될까요?"
        )
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{body}"


def _init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []
    if "processed_pdf_names" not in st.session_state:
        st.session_state.processed_pdf_names = []
    if "db_session_id" not in st.session_state:
        st.session_state.db_session_id = None
    if "dropdown_last_sid" not in st.session_state:
        st.session_state.dropdown_last_sid = None


def _append_turn(role: str, content: str) -> None:
    st.session_state.chat_history.append({"role": role, "content": content})
    st.session_state.conversation_memory.append({"role": role, "content": content})
    mem = st.session_state.conversation_memory
    if len(mem) > 50:
        st.session_state.conversation_memory = mem[-50:]


def _build_memory_messages() -> list[HumanMessage | AIMessage]:
    out: list[HumanMessage | AIMessage] = []
    for m in st.session_state.conversation_memory[:-1]:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        else:
            out.append(AIMessage(content=m["content"]))
    return out


def _ensure_db_session(sb: Client, default_title: str) -> str:
    if st.session_state.db_session_id:
        return st.session_state.db_session_id
    sid = create_chat_session(sb, default_title)
    st.session_state.db_session_id = sid
    return sid


def _autosave_chat(sb: Client) -> None:
    sid = st.session_state.db_session_id
    if not sid:
        return
    try:
        replace_session_messages(sb, sid, st.session_state.chat_history)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("자동 저장 실패: %s", exc)


def _maybe_update_title_from_first_turn(sb: Client, openai_key: str) -> None:
    hist = st.session_state.chat_history
    if len(hist) < 2:
        return
    sid = st.session_state.db_session_id
    if not sid:
        return
    first_u = next((h["content"] for h in hist if h["role"] == "user"), "")
    first_a = next((h["content"] for h in hist if h["role"] == "assistant"), "")
    if not first_u or not first_a:
        return
    if st.session_state.get("_title_done_for_sid") == sid:
        return
    title = generate_short_title(first_u, first_a, openai_key)
    try:
        update_session_title(sb, sid, title)
        st.session_state._title_done_for_sid = sid
    except Exception as exc:  # noqa: BLE001
        _logger.warning("제목 갱신 실패: %s", exc)


def _process_pdfs(sb: Client, uploaded_files: list[Any], emb: OpenAIEmbeddings) -> None:
    sid = _ensure_db_session(sb, "문서 임베딩 세션")
    raw_docs: list[Any] = []
    names: list[str] = []
    for f in uploaded_files:
        names.append(getattr(f, "name", "document.pdf"))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getbuffer())
            path = tmp.name
        try:
            docs = PyPDFLoader(path).load()
            for d in docs:
                d.metadata = dict(d.metadata or {})
                d.metadata["source_file"] = names[-1]
            raw_docs.extend(docs)
        finally:
            os.unlink(path)
    if not raw_docs:
        st.error("PDF에서 텍스트를 추출하지 못했습니다.")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    to_insert: list[tuple[str, str, list[float]]] = []
    for ch in chunks:
        fname = str(ch.metadata.get("source_file") or names[0] if names else "document.pdf")
        text = ch.page_content
        vec = emb.embed_documents([text])[0]
        to_insert.append((fname, text, vec))
    insert_vector_batch(sb, sid, to_insert)
    st.session_state.processed_pdf_names = names
    st.success(f"처리 완료: {len(names)}개 파일, {len(chunks)}개 청크. (세션 ID에 연결됨)")
    _autosave_chat(sb)


def _load_session_into_ui(sb: Client, session_id: str) -> None:
    msgs = load_session_messages(sb, session_id)
    st.session_state.chat_history = msgs
    st.session_state.conversation_memory = msgs[-50:]
    st.session_state.db_session_id = session_id
    st.session_state.processed_pdf_names = list_vector_filenames(sb, session_id)
    st.session_state._title_done_for_sid = session_id


def main() -> None:
    _configure_logging()
    st.set_page_config(page_title="멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
    _init_state()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div[data-testid="stChatMessage"] { border-radius: 12px; padding: 0.75rem; }
div.stButton > button {
  background-color: #ff69b4 !important;
  color: #ffffff !important;
  border: none !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 2.2, 1])
    with c1:
        if _LOGO_PATH.is_file():
            st.image(str(_LOGO_PATH), width=180)
        else:
            st.markdown("## 📚")
    with c2:
        st.markdown(
            """
<div style="text-align:center;font-size:4rem !important;line-height:1.1;">
<span style="color:#1f77b4 !important;">멀티세션</span>
<span style="color:#ffd700 !important;">RAG 챗봇</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.write("")

    openai_key, sup_url, sup_key = _env_status()
    sb = get_supabase()

    missing: list[str] = []
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not sup_url:
        missing.append("SUPABASE_URL")
    if not sup_key:
        missing.append("SUPABASE_ANON_KEY")
    if missing:
        st.warning(
            "다음 값이 설정되지 않았습니다: "
            + ", ".join(missing)
            + "\n\n**로컬:** 프로젝트 루트 `.env` 파일 (`"
            + str(_ENV_PATH)
            + "`)\n\n**Streamlit Cloud:** App settings → **Secrets** 에 위 키 이름으로 "
            "추가하세요. (선택: `[env]` 섹션 아래에 동일 키로 넣어도 됩니다.)"
        )

    sessions: list[dict[str, Any]] = []
    if sb:
        try:
            sessions = fetch_sessions(sb)
        except Exception as exc:  # noqa: BLE001
            st.error(f"세션 목록을 불러오지 못했습니다: {exc}")
            _logger.warning("세션 목록 오류: %s", exc)

    with st.sidebar:
        st.markdown("**LLM 모델**")
        st.radio(
            "LLM",
            (MODEL_NAME,),
            index=0,
            label_visibility="collapsed",
            key="ms_llm_model",
            disabled=True,
        )
        st.markdown("**RAG (PDF)**")
        rag_on = st.radio(
            "RAG",
            ("사용 안 함", "RAG 사용"),
            index=1,
            label_visibility="collapsed",
            key="ms_rag",
        )
        st.markdown("**PDF 업로드**")
        uploads = st.file_uploader(
            "PDF",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="ms_pdf_upload",
        )
        if st.button("파일 처리하기"):
            if not openai_key or not sb:
                st.error("OPENAI·Supabase 설정이 필요합니다.")
            elif not uploads:
                st.warning("업로드할 PDF를 선택해 주세요.")
            else:
                emb = get_embeddings()
                if emb is None:
                    st.error("임베딩을 초기화할 수 없습니다.")
                else:
                    try:
                        _process_pdfs(sb, list(uploads), emb)
                        _autosave_chat(sb)
                    except Exception as exc:  # noqa: BLE001
                        _logger.warning("PDF 처리 오류: %s", exc)
                        st.error(f"파일 처리 중 오류: {exc}")

        st.divider()
        st.markdown("**세션 관리**")

        labels = ["— 선택 —"]
        ids: list[str | None] = [None]
        for s in sessions:
            labels.append(str(s.get("title") or "제목 없음"))
            ids.append(str(s.get("id")))

        def _fmt(i: int) -> str:
            return labels[i]

        pick = st.selectbox(
            "저장된 세션",
            range(len(labels)),
            format_func=_fmt,
            key="ms_session_select_idx",
        )
        selected_id = ids[pick] if pick < len(ids) else None

        if selected_id and sb and st.session_state.dropdown_last_sid != selected_id:
            try:
                _load_session_into_ui(sb, selected_id)
                st.session_state.dropdown_last_sid = selected_id
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.session_state.dropdown_last_sid = selected_id
                st.error(f"세션 자동 로드 실패: {exc}")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("세션로드"):
                if not sb:
                    st.error("Supabase가 없습니다.")
                elif not selected_id:
                    st.warning("풀다운에서 세션을 먼저 선택하세요.")
                else:
                    try:
                        _load_session_into_ui(sb, selected_id)
                        st.session_state.dropdown_last_sid = selected_id
                        st.success("세션을 불러왔습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

        with col_b:
            if st.button("세션저장"):
                if not sb or not openai_key:
                    st.error("OPENAI·Supabase 설정이 필요합니다.")
                elif not st.session_state.chat_history:
                    st.warning("저장할 대화가 없습니다.")
                else:
                    hist = st.session_state.chat_history
                    first_u = next((h["content"] for h in hist if h["role"] == "user"), "새 세션")
                    first_a = next((h["content"] for h in hist if h["role"] == "assistant"), "")
                    title = generate_short_title(first_u, first_a or first_u, openai_key)
                    try:
                        new_sid = create_chat_session(sb, title)
                        replace_session_messages(sb, new_sid, hist)
                        src = st.session_state.db_session_id
                        if src:
                            copy_vectors_to_session(sb, src, new_sid)
                        st.success("새 세션으로 저장했습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

        if st.button("세션삭제"):
            if not sb:
                st.error("Supabase가 없습니다.")
            elif not selected_id:
                st.warning("삭제할 세션을 선택하세요.")
            else:
                try:
                    delete_session(sb, selected_id)
                    if st.session_state.db_session_id == selected_id:
                        st.session_state.db_session_id = None
                        st.session_state.chat_history = []
                        st.session_state.conversation_memory = []
                        st.session_state.processed_pdf_names = []
                    st.session_state.dropdown_last_sid = None
                    st.session_state.ms_session_select_idx = 0
                    st.success("삭제했습니다.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        if st.button("화면초기화"):
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.session_state.processed_pdf_names = []
            st.session_state.db_session_id = None
            st.session_state.dropdown_last_sid = None
            st.session_state.ms_session_select_idx = 0
            st.session_state._title_done_for_sid = None
            st.rerun()

        if st.button("vectordb"):
            if not sb:
                st.error("Supabase가 없습니다.")
            else:
                sid = st.session_state.db_session_id
                names = list_vector_filenames(sb, sid)
                if not names:
                    st.info("현재 세션에 저장된 벡터 문서 파일명이 없습니다.")
                else:
                    st.markdown("**현재 세션 벡터 DB 파일명**")
                    for n in names:
                        st.text(n)

        if st.session_state.processed_pdf_names:
            st.caption("처리·연결된 PDF 이름")
            for n in st.session_state.processed_pdf_names:
                st.text(n)

        n_chat = len(st.session_state.chat_history)
        st.text(
            f"모델: {MODEL_NAME}\n"
            f"RAG: {rag_on}\n"
            f"활성 DB 세션: {st.session_state.db_session_id or '없음'}\n"
            f"대화 수: {n_chat}"
        )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    if not openai_key:
        st.chat_input("OPENAI_API_KEY를 설정하면 채팅할 수 있습니다.", disabled=True)
        return

    if prompt := st.chat_input("질문을 입력하세요"):
        if sb and not st.session_state.db_session_id:
            try:
                _ensure_db_session(sb, "새 대화")
            except Exception as exc:  # noqa: BLE001
                st.error(f"세션을 만들 수 없습니다: {exc}")
                return

        _append_turn("user", prompt)
        with st.chat_message("user"):
            st.markdown(remove_separators(prompt))

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_reply = ""
            llm = get_llm()
            try:
                if rag_on == "RAG 사용":
                    if not sb or not st.session_state.db_session_id:
                        full_reply = (
                            "RAG를 쓰려면 Supabase 연결 후 PDF를 업로드하고 "
                            "「파일 처리하기」로 임베딩을 만든 뒤 질문해 주세요."
                        )
                        placeholder.markdown(full_reply)
                    elif not get_embeddings():
                        full_reply = "임베딩 모델을 초기화할 수 없습니다."
                        placeholder.markdown(full_reply)
                    else:
                        emb = get_embeddings()
                        assert emb is not None
                        assert sb is not None
                        sid = st.session_state.db_session_id
                        assert sid is not None
                        docs = retrieve_with_rpc(sb, emb, prompt, sid)
                        ctx = "\n\n".join(d.page_content for d in docs)
                        msgs: list[Any] = [SystemMessage(content=ANSWER_SYSTEM)]
                        msgs.extend(_build_memory_messages())
                        msgs.append(
                            HumanMessage(
                                content=(
                                    "다음은 참고 자료에서 검색된 내용입니다.\n\n"
                                    f"{ctx}\n\n사용자 질문:\n{prompt}"
                                )
                            )
                        )
                        if llm is None:
                            full_reply = "LLM을 초기화할 수 없습니다."
                            placeholder.markdown(full_reply)
                        else:
                            acc = ""
                            for chunk in llm.stream(msgs):
                                if chunk.content:
                                    acc += str(chunk.content)
                                    placeholder.markdown(remove_separators(acc))
                            full_reply = acc
                            full_reply += _followup_block(prompt, full_reply, openai_key, llm)
                            full_reply = remove_separators(full_reply)
                            placeholder.markdown(full_reply)
                else:
                    if llm is None:
                        full_reply = "LLM을 초기화할 수 없습니다."
                        placeholder.markdown(full_reply)
                    else:
                        msgs2: list[Any] = [SystemMessage(content=ANSWER_SYSTEM)]
                        msgs2.extend(_build_memory_messages())
                        msgs2.append(HumanMessage(content=prompt))
                        acc2 = ""
                        for chunk in llm.stream(msgs2):
                            if chunk.content:
                                acc2 += str(chunk.content)
                                placeholder.markdown(remove_separators(acc2))
                        full_reply = acc2
                        full_reply += _followup_block(prompt, full_reply, openai_key, llm)
                        full_reply = remove_separators(full_reply)
                        placeholder.markdown(full_reply)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("응답 생성 오류: %s", exc)
                full_reply = f"오류가 발생했습니다: {exc}"
                placeholder.markdown(full_reply)

            _append_turn("assistant", full_reply)
            if sb:
                _autosave_chat(sb)
                if openai_key:
                    _maybe_update_title_from_first_turn(sb, openai_key)


if __name__ == "__main__":
    main()
