const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const clearButton = document.querySelector("#clear-button");
const messageStream = document.querySelector("#message-stream");
let emptyState = document.querySelector("#empty-state");
const characterCount = document.querySelector("#character-count");
const serviceState = document.querySelector("#service-state");
const serviceLabel = document.querySelector("#service-label");
const resultBadge = document.querySelector("#result-badge");
const evidenceCaption = document.querySelector("#evidence-caption");
const gateValue = document.querySelector("#gate-value");
const scoreValue = document.querySelector("#score-value");
const sourceCount = document.querySelector("#source-count");
const sourceList = document.querySelector("#source-list");
const topKInput = document.querySelector("#top-k");
const similarityInput = document.querySelector("#min-similarity");
const similarityValue = document.querySelector("#similarity-value");

let requestInProgress = false;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const REQUEST_TIMEOUT_MS = 120_000;

function setServiceState(state, label) {
  serviceState.classList.remove("online", "offline");
  if (state) {
    serviceState.classList.add(state);
  }
  serviceLabel.textContent = label;
}

async function checkHealth() {
  try {
    const response = await fetch("/health", {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error("服务未就绪");
    }
    const health = await response.json();
    setServiceState(
      health.pipeline_loaded ? "online" : "offline",
      health.pipeline_loaded ? "模型已就绪" : "模型正在加载",
    );
  } catch {
    setServiceState("offline", "无法连接服务");
  }
}

function autoResizeInput() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 170)}px`;
  characterCount.textContent = `${questionInput.value.length} / 500`;
}

function removeEmptyState() {
  if (emptyState && emptyState.isConnected) {
    emptyState.remove();
  }
}

function createMessage(role, text, variant = "") {
  const article = document.createElement("article");
  article.className = `message ${role} ${variant}`.trim();

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "你的问题" : "知识库回答";

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;

  article.append(label, body);
  messageStream.append(article);
  messageStream.scrollTo({
    top: messageStream.scrollHeight,
    behavior: reduceMotion.matches ? "auto" : "smooth",
  });

  return article;
}

function createLoadingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.id = "loading-message";

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = "正在检索并生成回答";

  const lines = document.createElement("div");
  lines.className = "loading-lines";
  lines.setAttribute("aria-label", "正在处理，请稍候");
  lines.append(
    document.createElement("span"),
    document.createElement("span"),
    document.createElement("span"),
  );

  article.append(label, lines);
  messageStream.append(article);
  messageStream.scrollTop = messageStream.scrollHeight;
  return article;
}

function setResultState(kind, label) {
  resultBadge.className = `result-badge ${kind}`;
  resultBadge.textContent = label;
}

function resetEvidence() {
  setResultState("neutral", "待检索");
  evidenceCaption.textContent = "等待一次有效提问";
  gateValue.textContent = "—";
  scoreValue.textContent = "—";
  sourceCount.textContent = "—";
  sourceList.replaceChildren();

  const placeholder = document.createElement("li");
  placeholder.className = "source-placeholder";
  placeholder.textContent =
    "完成提问后，这里会显示标题、文本块编号和检索分数。";
  sourceList.append(placeholder);
}

function renderEvidence(result) {
  const sources = Array.isArray(result.sources) ? result.sources : [];
  const passed = Boolean(result.passed);

  setResultState(passed ? "success" : "warning", passed ? "已通过" : "资料不足");
  evidenceCaption.textContent = passed
    ? "本次回答已通过相关性门槛"
    : "本次检索未达到生成门槛";
  gateValue.textContent = passed ? "允许生成" : "已安全拒答";
  scoreValue.textContent = Number(result.max_score || 0).toFixed(4);
  sourceCount.textContent = `${sources.length} 条`;
  sourceList.replaceChildren();

  if (!sources.length) {
    const placeholder = document.createElement("li");
    placeholder.className = "source-placeholder";
    placeholder.textContent = "没有达到门槛的来源。请换一种问法或检查知识库内容。";
    sourceList.append(placeholder);
    return;
  }

  sources.forEach((source, index) => {
    const item = document.createElement("li");
    item.className = "source-item";

    const rank = document.createElement("span");
    rank.className = "source-rank";
    rank.textContent = String(source.rank ?? index + 1).padStart(2, "0");

    const main = document.createElement("div");
    main.className = "source-main";

    const title = document.createElement("div");
    title.className = "source-title";
    title.textContent = source.title || "未命名来源";
    title.title = source.title || "未命名来源";

    const meta = document.createElement("div");
    meta.className = "source-meta";

    const chunkId = document.createElement("span");
    chunkId.className = "source-id";
    chunkId.textContent = source.chunk_id || "unknown_chunk";
    chunkId.title = source.chunk_id || "unknown_chunk";

    const score = document.createElement("span");
    score.className = "source-score";
    score.textContent = Number(source.score || 0).toFixed(4);

    meta.append(chunkId, score);
    main.append(title, meta);
    item.append(rank, main);
    sourceList.append(item);
  });
}

function setBusy(isBusy) {
  requestInProgress = isBusy;
  sendButton.disabled = isBusy;
  clearButton.disabled = isBusy;
  questionInput.disabled = isBusy;
  messageStream.setAttribute("aria-busy", String(isBusy));
  sendButton.querySelector("span").textContent = isBusy ? "处理中" : "发送";
}

async function askQuestion(query) {
  removeEmptyState();
  createMessage("user", query);
  const loadingMessage = createLoadingMessage();
  setBusy(true);
  setResultState("neutral", "检索中");
  evidenceCaption.textContent = "正在计算相关性并整理来源";
  const controller = new AbortController();
  const timeoutId = window.setTimeout(
    () => controller.abort(),
    REQUEST_TIMEOUT_MS,
  );

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      signal: controller.signal,
      body: JSON.stringify({
        query,
        top_k: Number(topKInput.value),
        min_similarity: Number(similarityInput.value),
        max_new_tokens: 200,
      }),
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `请求失败（${response.status}）`);
    }

    loadingMessage.remove();
    const variant = payload.passed ? "entering" : "refusal entering";
    createMessage("assistant", payload.answer, variant);
    renderEvidence(payload);
    setServiceState("online", "模型已就绪");
  } catch (error) {
    loadingMessage.remove();
    const message = error?.name === "AbortError"
      ? "请求超过 2 分钟，已自动取消"
      : error instanceof Error
        ? error.message
        : "发生未知错误，请稍后重试。";
    createMessage(
      "assistant",
      `暂时无法完成回答：${message}\n请确认服务仍在运行，然后重新发送问题。`,
      "error entering",
    );
    setResultState("error", "请求失败");
    evidenceCaption.textContent = "没有取得可用的检索结果";
    gateValue.textContent = "未完成";
    scoreValue.textContent = "—";
    sourceCount.textContent = "—";
    setServiceState("offline", "服务响应异常");
    questionInput.value = query;
    autoResizeInput();
  } finally {
    window.clearTimeout(timeoutId);
    setBusy(false);
    questionInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (requestInProgress) {
    return;
  }

  const query = questionInput.value.trim();
  if (!query) {
    questionInput.focus();
    return;
  }

  questionInput.value = "";
  autoResizeInput();
  askQuestion(query);
});

questionInput.addEventListener("input", autoResizeInput);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.example || "";
    autoResizeInput();
    questionInput.focus();
  });
});

clearButton.addEventListener("click", () => {
  messageStream.replaceChildren();

  const restoredEmptyState = emptyState.cloneNode(true);
  restoredEmptyState.id = "empty-state";
  messageStream.append(restoredEmptyState);
  emptyState = restoredEmptyState;
  restoredEmptyState.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      questionInput.value = button.dataset.example || "";
      autoResizeInput();
      questionInput.focus();
    });
  });

  resetEvidence();
  questionInput.focus();
});

similarityInput.addEventListener("input", () => {
  similarityValue.textContent = Number(similarityInput.value).toFixed(2);
});

autoResizeInput();
checkHealth();
