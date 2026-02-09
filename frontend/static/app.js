// Simple ChatGPT-like UI for MSRA
// Uses your FastAPI endpoints + SQLite persistence in backend.

const chatListEl = document.getElementById("chatList");
const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");

const newChatBtn = document.getElementById("newChatBtn");
const renameBtn = document.getElementById("renameBtn");
const deleteBtn = document.getElementById("deleteBtn");
const exportPdfBtn = document.getElementById("exportPdfBtn");
const exportDocxBtn = document.getElementById("exportDocxBtn");

const activeChatTitleEl = document.getElementById("activeChatTitle");
const activeChatMetaEl = document.getElementById("activeChatMeta");

let activeChatId = localStorage.getItem("msra_active_chat_id") || null;
let chatsCache = []; // [{id,title,updated_at}]

// NEW: selected turn = the user message id you clicked
let selectedUserMessageId = null;

function fmtTime(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString() + " " + d.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
  } catch {
    return "";
  }
}

async function api(path, opts={}) {
  const res = await fetch(path, {
    headers: {"Content-Type":"application/json"},
    ...opts
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j?.detail) msg = j.detail;
    } catch {}
    throw new Error(msg);
  }
  return res;
}

async function loadChats() {
  const res = await api("/chats");
  chatsCache = await res.json();
  renderChatList();
  if (!activeChatId) {
    if (chatsCache.length === 0) {
      await createNewChat();
    } else {
      setActiveChat(chatsCache[0].id);
    }
  } else {
    if (!chatsCache.find(c => c.id === activeChatId)) {
      if (chatsCache.length) setActiveChat(chatsCache[0].id);
      else await createNewChat();
    } else {
      setActiveChat(activeChatId);
    }
  }
}

function renderChatList() {
  chatListEl.innerHTML = "";
  chatsCache.forEach(c => {
    const item = document.createElement("div");
    item.className = "chat-item" + (c.id === activeChatId ? " active" : "");
    item.onclick = () => setActiveChat(c.id);

    const title = document.createElement("div");
    title.className = "title";
    title.textContent = c.title || "New Chat";

    const time = document.createElement("div");
    time.className = "time";
    time.textContent = c.updated_at ? fmtTime(c.updated_at) : "";

    item.appendChild(title);
    item.appendChild(time);
    chatListEl.appendChild(item);
  });
}

function setActiveChat(chatId) {
  activeChatId = chatId;
  selectedUserMessageId = null; // reset selection when switching chat
  localStorage.setItem("msra_active_chat_id", chatId);
  renderChatList();
  loadMessages(chatId);
  const meta = chatsCache.find(c => c.id === chatId);
  activeChatTitleEl.textContent = meta?.title || "New Chat";
  activeChatMetaEl.textContent = meta?.updated_at ? `Updated: ${fmtTime(meta.updated_at)}` : "";
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(str) {
  return (str || "").replace(/[&<>"']/g, (m) => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"
  }[m]));
}

function selectTurn(userMessageId) {
  selectedUserMessageId = userMessageId;
  // highlight selected user bubble
  [...messagesEl.querySelectorAll(".bubble.user")].forEach(el => {
    el.classList.toggle("selected", el.dataset.messageId === userMessageId);
  });
}

function exportTurn(format, userMessageId) {
  if (!activeChatId) return;
  if (!userMessageId) {
    // fallback: whole chat export (keeps old behavior)
    window.open(`/chats/${activeChatId}/export?format=${format}`, "_blank");
    return;
  }
  window.open(`/chats/${activeChatId}/turns/${userMessageId}/export?format=${format}`, "_blank");
}

function renderMessageBubble(messageId, role, content, references=[]) {
  const wrap = document.createElement("div");
  wrap.className = "bubble " + (role === "user" ? "user" : "assistant");
  wrap.dataset.messageId = messageId || "";

  // content
  const body = document.createElement("div");
  body.className = "bubble-body";
  body.innerHTML = escapeHtml(content);
  wrap.appendChild(body);

  // NEW: per-user-message export controls
  if (role === "user") {
    const actions = document.createElement("div");
    actions.className = "turn-actions";

    const hint = document.createElement("span");
    hint.className = "turn-hint";
    hint.textContent = "Selected for export when clicked";
    actions.appendChild(hint);

    const btnPdf = document.createElement("button");
    btnPdf.className = "btn tiny";
    btnPdf.textContent = "PDF";
    btnPdf.onclick = (e) => {
      e.stopPropagation();
      selectTurn(messageId);
      exportTurn("pdf", messageId);
    };

    const btnDocx = document.createElement("button");
    btnDocx.className = "btn tiny";
    btnDocx.textContent = "DOCX";
    btnDocx.onclick = (e) => {
      e.stopPropagation();
      selectTurn(messageId);
      exportTurn("docx", messageId);
    };

    actions.appendChild(btnPdf);
    actions.appendChild(btnDocx);
    wrap.appendChild(actions);

    // click bubble selects this turn
    wrap.onclick = () => selectTurn(messageId);
  }

  if (role === "assistant" && references && references.length) {
    const refs = document.createElement("div");
    refs.className = "refs";
    refs.innerHTML = "<div><b>References</b></div>";
    const ul = document.createElement("ul");
    ul.style.margin = "8px 0 0 18px";
    references.forEach(u => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = u;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = u;
      li.appendChild(a);
      ul.appendChild(li);
    });
    refs.appendChild(ul);
    wrap.appendChild(refs);
  }

  messagesEl.appendChild(wrap);
}

async function loadMessages(chatId) {
  messagesEl.innerHTML = "";
  const res = await api(`/chats/${chatId}`);
  const msgs = await res.json();
  msgs.forEach(m => renderMessageBubble(m.id, m.role, m.content, m.references || []));
  scrollToBottom();
}

async function createNewChat() {
  const res = await api("/chats", {method:"POST", body: JSON.stringify({title:"New Chat"})});
  const chat = await res.json();
  await loadChats();
  setActiveChat(chat.id);
}

async function renameActiveChat() {
  if (!activeChatId) return;
  const current = chatsCache.find(c => c.id === activeChatId)?.title || "New Chat";
  const title = prompt("Rename chat:", current);
  if (!title) return;
  await api(`/chats/${activeChatId}`, {method:"PATCH", body: JSON.stringify({title})});
  await loadChats();
  setActiveChat(activeChatId);
}

async function deleteActiveChat() {
  if (!activeChatId) return;
  const ok = confirm("Delete this chat? This cannot be undone.");
  if (!ok) return;
  await api(`/chats/${activeChatId}`, {method:"DELETE"});
  activeChatId = null;
  selectedUserMessageId = null;
  localStorage.removeItem("msra_active_chat_id");
  await loadChats();
}

async function sendMessage() {
  const text = (inputEl.value || "").trim();
  if (!text || !activeChatId) return;

  // optimistic UI
  renderMessageBubble("tmp-user", "user", text, []);
  scrollToBottom();
  inputEl.value = "";

  // temporary assistant bubble
  const pending = document.createElement("div");
  pending.className = "bubble assistant";
  pending.textContent = "Thinking...";
  messagesEl.appendChild(pending);
  scrollToBottom();

  try {
    const res = await api(`/chats/${activeChatId}/messages`, {
      method:"POST",
      body: JSON.stringify({message: text})
    });
    const data = await res.json();

    pending.remove();

    // authoritative history
    messagesEl.innerHTML = "";
    (data.all_messages || []).forEach(m => renderMessageBubble(m.id, m.role, m.content, m.references || []));
    scrollToBottom();

    await loadChats();
    setActiveChat(activeChatId);

    // auto-select the latest user message for export convenience
    const all = data.all_messages || [];
    for (let i = all.length - 1; i >= 0; i--) {
      if (all[i].role === "user") {
        selectTurn(all[i].id);
        break;
      }
    }
  } catch (e) {
    pending.remove();
    renderMessageBubble("tmp-err", "assistant", `Error: ${e.message}`, []);
    scrollToBottom();
  }
}

function exportFromTop(format) {
  // If user has selected a query, export only that turn; otherwise export whole chat
  exportTurn(format, selectedUserMessageId);
}

newChatBtn?.addEventListener("click", createNewChat);
renameBtn?.addEventListener("click", renameActiveChat);
deleteBtn?.addEventListener("click", deleteActiveChat);

exportPdfBtn?.addEventListener("click", () => exportFromTop("pdf"));
exportDocxBtn?.addEventListener("click", () => exportFromTop("docx"));

sendBtn?.addEventListener("click", sendMessage);

// Enter to send (Shift+Enter = newline)
inputEl?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Boot
loadChats().catch(err => {
  alert("Failed to load chats: " + err.message);
});
