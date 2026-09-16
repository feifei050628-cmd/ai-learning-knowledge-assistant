<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue";
import { marked } from "marked";
import {
  BookOpen, Bot, Check, ChevronDown, ChevronRight, CircleUserRound,
  Clipboard, Database, FileText, FolderOpen, History, Library, Menu,
  MessageSquareText, PanelRightClose, PanelRightOpen, Plus, RefreshCw,
  Search, Send, Settings, Sparkles, Square, ThumbsDown, ThumbsUp,
  Trash2, Upload, X, AlertCircle, CircleCheck, LoaderCircle,
} from "lucide-vue-next";
import { api, ApiError } from "./api";
import { demoDocuments, demoHistory } from "./mock";
import type { AskResponse, ChatMessage, DemoDocument, SourceItem } from "./types";

type ViewName = "chat" | "knowledge" | "documents" | "settings";
const view = ref<ViewName>("chat");
const service = ref<"checking" | "online" | "offline">("checking");
const messages = ref<ChatMessage[]>([]);
const query = ref("");
const topK = ref(5);
const minSimilarity = ref(0.48);
const controller = ref<AbortController | null>(null);
const streamEl = ref<HTMLElement | null>(null);
const evidenceOpen = ref(window.innerWidth > 850);
const mobileNavOpen = ref(false);
const activeResult = ref<AskResponse | null>(null);
const expandedSource = ref<string | null>(null);
const copied = ref<string | null>(null);
const documents = ref<DemoDocument[]>(structuredClone(demoDocuments));
const documentSearch = ref("");
const confirmDelete = ref<DemoDocument | null>(null);
const toast = ref("");

const busy = computed(() => Boolean(controller.value));
const serviceLabel = computed(() => service.value === "online" ? "知识服务在线" : service.value === "offline" ? "知识服务离线" : "正在检查服务");
const filteredDocuments = computed(() => documents.value.filter((item) => item.name.toLowerCase().includes(documentSearch.value.toLowerCase())));
const readyCount = computed(() => documents.value.filter((item) => item.status === "ready").length);
const totalChunks = computed(() => documents.value.reduce((total, item) => total + item.chunks, 0));

marked.setOptions({ breaks: true, gfm: true });
function markdown(text: string) {
  return marked.parse(text) as string;
}

function uid() { return `${Date.now()}-${Math.random().toString(36).slice(2)}`; }
function notify(message: string) {
  toast.value = message;
  window.setTimeout(() => { if (toast.value === message) toast.value = ""; }, 2600);
}
async function checkHealth() {
  try {
    const result = await api.health();
    service.value = result.pipeline_loaded ? "online" : "checking";
  } catch { service.value = "offline"; }
}
async function scrollToBottom() {
  await nextTick();
  streamEl.value?.scrollTo({ top: streamEl.value.scrollHeight, behavior: "smooth" });
}
function newChat() {
  controller.value?.abort();
  messages.value = [];
  activeResult.value = null;
  query.value = "";
  view.value = "chat";
  mobileNavOpen.value = false;
}
function stopGeneration() {
  controller.value?.abort();
  controller.value = null;
}
async function submitQuestion(forcedQuery?: string) {
  const text = (forcedQuery ?? query.value).trim();
  if (!text || busy.value) return;
  view.value = "chat";
  query.value = "";
  const user: ChatMessage = { id: uid(), role: "user", content: text };
  const answer: ChatMessage = { id: uid(), role: "assistant", content: "", status: "loading", feedback: null };
  messages.value.push(user, answer);
  activeResult.value = null;
  controller.value = new AbortController();
  scrollToBottom();
  try {
    const result = await api.ask({ query: text, top_k: topK.value, min_similarity: minSimilarity.value, max_new_tokens: 200 }, controller.value.signal);
    answer.content = result.answer;
    answer.result = result;
    answer.status = result.passed ? "done" : "refused";
    activeResult.value = result;
    service.value = "online";
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      answer.content = "已停止本次生成。你可以调整问题后重新发送。";
    } else {
      const detail = error instanceof ApiError ? error.message : "无法连接知识服务，请确认 FastAPI 已启动。";
      answer.content = `请求未完成：${detail}`;
      service.value = "offline";
    }
    answer.status = "error";
  } finally {
    controller.value = null;
    scrollToBottom();
  }
}
function regenerate(index: number) {
  const previous = [...messages.value].slice(0, index).reverse().find((item) => item.role === "user");
  if (previous) submitQuestion(previous.content);
}
async function copyText(text: string, id: string) {
  await navigator.clipboard.writeText(text);
  copied.value = id;
  window.setTimeout(() => { if (copied.value === id) copied.value = null; }, 1600);
}
function setFeedback(message: ChatMessage, feedback: "up" | "down") {
  message.feedback = message.feedback === feedback ? null : feedback;
  notify("已在本地记录反馈（后端反馈接口尚未接入）");
}
function openSource(source: SourceItem) {
  expandedSource.value = expandedSource.value === source.chunk_id ? null : source.chunk_id;
}
function chooseView(next: ViewName) {
  view.value = next;
  mobileNavOpen.value = false;
}
function handleUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  const files = Array.from(input.files || []);
  files.forEach((file) => {
    const extension = file.name.split(".").pop()?.toUpperCase();
    if (!extension || !["PDF", "TXT", "MD"].includes(extension)) return;
    documents.value.unshift({ id: uid(), name: file.name, type: extension as DemoDocument["type"], size: formatBytes(file.size), updatedAt: "刚刚", status: "processing", chunks: 0 });
  });
  if (files.length) notify("文件已加入本地演示列表，尚未上传后端");
  input.value = "";
}
function formatBytes(bytes: number) { return bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`; }
function reparse(document: DemoDocument) {
  document.status = "processing";
  document.updatedAt = "刚刚";
  notify("已模拟重新解析，真实接口尚未接入");
}
function deleteDocument() {
  if (!confirmDelete.value) return;
  documents.value = documents.value.filter((item) => item.id !== confirmDelete.value?.id);
  confirmDelete.value = null;
  notify("已从本地演示列表移除");
}
function onComposerKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuestion();
  }
}
onMounted(checkHealth);
</script>

<template>
  <div class="app-shell">
    <a class="skip-link" href="#main-content">跳到主内容</a>
    <header class="topbar">
      <button class="icon-button mobile-only" type="button" aria-label="打开导航" @click="mobileNavOpen = true"><Menu :size="20" /></button>
      <button class="brand" type="button" @click="chooseView('chat')">
        <span class="brand-mark"><Sparkles :size="19" /></span>
        <span><strong>AI 技术学习助手</strong><small>可追溯知识工作台</small></span>
      </button>
      <div class="knowledge-switch"><Database :size="16" /><span>AI 技术学习库</span><ChevronDown :size="16" /></div>
      <div class="top-actions">
        <span class="service-pill" :class="service"><span></span>{{ serviceLabel }}</span>
        <a class="docs-link" href="/docs" target="_blank" rel="noreferrer">API 文档</a>
        <button class="user-menu" type="button"><span>学</span><span class="user-copy"><b>学习者</b><small>本地工作区</small></span><ChevronDown :size="16" /></button>
      </div>
    </header>

    <div v-if="mobileNavOpen" class="scrim" @click="mobileNavOpen = false"></div>
    <aside class="sidebar" :class="{ open: mobileNavOpen }">
      <div class="sidebar-mobile-head"><strong>导航</strong><button class="icon-button" @click="mobileNavOpen = false"><X :size="19" /></button></div>
      <button class="new-chat" type="button" @click="newChat"><Plus :size="19" />新建对话<span>Ctrl K</span></button>
      <nav aria-label="主导航">
        <button :class="{ active: view === 'chat' }" @click="chooseView('chat')"><MessageSquareText :size="19" />智能问答</button>
        <button :class="{ active: view === 'knowledge' }" @click="chooseView('knowledge')"><Library :size="19" />知识库<span class="count">1</span></button>
        <button :class="{ active: view === 'documents' }" @click="chooseView('documents')"><FolderOpen :size="19" />文档管理<span class="count">{{ documents.length }}</span></button>
        <button :class="{ active: view === 'settings' }" @click="chooseView('settings')"><Settings :size="19" />个人设置</button>
      </nav>
      <section class="history-list">
        <div class="section-label"><span><History :size="15" />历史会话</span><em>Mock</em></div>
        <button v-for="item in demoHistory" :key="item" type="button" @click="query = item; chooseView('chat')">{{ item }}</button>
      </section>
      <div class="sidebar-foot"><BookOpen :size="17" /><div><strong>29 篇学习笔记</strong><small>本地知识源</small></div></div>
    </aside>

    <main id="main-content" :class="['main', `view-${view}`, { 'evidence-collapsed': !evidenceOpen }]">
      <template v-if="view === 'chat'">
        <section class="chat-workspace">
          <header class="page-head">
            <div><p class="page-context">智能问答</p><h1>{{ messages.length ? '对话详情' : '开始一次可验证的问答' }}</h1><p>从 AI 技术学习库中检索依据，回答后附上来源。</p></div>
            <button class="panel-toggle" type="button" @click="evidenceOpen = !evidenceOpen" :aria-label="evidenceOpen ? '收起检索详情' : '展开检索详情'">
              <PanelRightClose v-if="evidenceOpen" :size="19" /><PanelRightOpen v-else :size="19" />
            </button>
          </header>

          <div ref="streamEl" class="message-stream" aria-live="polite" :aria-busy="busy">
            <section v-if="!messages.length" class="welcome-state">
              <div class="assistant-intro"><span class="ai-avatar"><Sparkles :size="20" /></span><div><h2>你好，我是你的 AI 技术学习助手</h2><p>我会先检索本地学习资料，再给出带引用的回答。如果资料不足，我会明确拒答。</p></div></div>
              <div class="prompt-guide"><span>试试这些问题</span><button @click="submitQuestion('RAG 的完整处理流程是什么？')"><b>RAG 的完整处理流程是什么？</b><small>检索基础原理和项目实现</small><ChevronRight :size="18" /></button><button @click="submitQuestion('为什么要设置相关性门槛？')"><b>为什么要设置相关性门槛？</b><small>了解低置信度拒答机制</small><ChevronRight :size="18" /></button><button @click="submitQuestion('如何评估生成答案的质量？')"><b>如何评估生成答案的质量？</b><small>查看评估指标与实践方法</small><ChevronRight :size="18" /></button></div>
            </section>

            <article v-for="(message, index) in messages" :key="message.id" :class="['message', message.role, message.status]">
              <div class="message-avatar"><CircleUserRound v-if="message.role === 'user'" :size="20" /><Sparkles v-else :size="19" /></div>
              <div class="message-content">
                <div class="message-meta"><strong>{{ message.role === 'user' ? '你' : 'AI 学习助手' }}</strong><span v-if="message.role === 'assistant' && message.status === 'done'" class="grounded"><Check :size="13" />由知识库资料生成</span><span v-if="message.status === 'refused'" class="low-confidence"><AlertCircle :size="13" />低置信度·已安全拒答</span></div>
                <div v-if="message.status === 'loading'" class="answer-loading"><span></span><span></span><span></span><p>正在检索知识库并组织回答…</p></div>
                <div v-else class="markdown-body" v-html="markdown(message.content)"></div>
                <div v-if="message.result?.sources.length" class="inline-sources"><button v-for="source in message.result.sources.slice(0, 3)" :key="source.chunk_id" @click="activeResult = message.result || null; evidenceOpen = true"><FileText :size="14" /><span>{{ source.title }}</span><em>{{ Math.round(source.score * 100) }}%</em></button></div>
                <div v-if="message.role === 'assistant' && message.status !== 'loading'" class="message-actions"><button :class="{ selected: message.feedback === 'up' }" @click="setFeedback(message, 'up')" aria-label="有帮助"><ThumbsUp :size="15" /></button><button :class="{ selected: message.feedback === 'down' }" @click="setFeedback(message, 'down')" aria-label="无帮助"><ThumbsDown :size="15" /></button><button @click="copyText(message.content, message.id)"><Check v-if="copied === message.id" :size="15" /><Clipboard v-else :size="15" />{{ copied === message.id ? '已复制' : '复制' }}</button><button @click="regenerate(index)"><RefreshCw :size="15" />重新生成</button></div>
              </div>
            </article>
          </div>

          <div class="composer-wrap">
            <div class="composer"><textarea v-model="query" rows="1" maxlength="500" aria-label="输入问题" placeholder="输入一个 AI 技术问题…" @keydown="onComposerKeydown"></textarea><span class="mode-chip"><Database :size="13" />知识库增强</span><button v-if="busy" class="stop-button" type="button" aria-label="停止生成" @click="stopGeneration"><Square :size="17" /></button><button v-else class="send-button" type="button" :disabled="!query.trim()" aria-label="发送" @click="submitQuestion()"><Send :size="18" /></button></div>
            <div class="composer-foot"><span>Enter 发送 · Shift + Enter 换行</span><span>{{ query.length }} / 500</span></div>
          </div>
        </section>

        <aside class="evidence-panel" :class="{ hidden: !evidenceOpen }">
          <header><div><p class="page-context">可追溯性</p><h2>检索详情</h2></div><button class="icon-button" @click="evidenceOpen = false"><X :size="18" /></button></header>
          <template v-if="activeResult">
            <div :class="['confidence-summary', activeResult.passed ? 'passed' : 'blocked']"><CircleCheck v-if="activeResult.passed" :size="20" /><AlertCircle v-else :size="20" /><div><strong>{{ activeResult.passed ? '相关性门控已通过' : '未达到生成门槛' }}</strong><span>最高相似度 {{ activeResult.max_score.toFixed(4) }}</span></div></div>
            <div class="retrieval-flow"><div class="flow-step done"><span>1</span><div><b>问题向量化</b><small>BGE 中文语义表示</small></div><Check :size="15" /></div><div class="flow-step done"><span>2</span><div><b>Top-K 召回</b><small>检索 {{ activeResult.sources.length }} 个相关片段</small></div><Check :size="15" /></div><div class="flow-step done"><span>3</span><div><b>相关性门控</b><small>当前门槛 {{ minSimilarity.toFixed(2) }}</small></div><Check :size="15" /></div></div>
            <div class="source-panel-title"><h3>召回资料</h3><span>{{ activeResult.sources.length }} 个片段</span></div>
            <ol class="evidence-list"><li v-for="source in activeResult.sources" :key="source.chunk_id"><button class="source-row" @click="openSource(source)"><span class="source-rank">{{ source.rank }}</span><span class="source-copy"><b>{{ source.title }}</b><small>{{ source.chunk_id }}</small></span><em>{{ (source.score * 100).toFixed(1) }}%</em><ChevronDown :class="{ rotated: expandedSource === source.chunk_id }" :size="16" /></button><div v-if="expandedSource === source.chunk_id" class="source-preview"><p>当前后端只返回标题、片段 ID 和相似度，未返回片段原文。</p><span>API 适配层已保留原文展示位置。</span></div></li></ol>
          </template>
          <div v-else class="evidence-empty"><Search :size="28" /><h3>等待检索</h3><p>发送问题后，这里会显示召回过程、来源和相似度。</p></div>
          <details class="retrieval-settings"><summary>检索参数<ChevronDown :size="16" /></summary><label>Top-K <select v-model="topK"><option :value="3">3</option><option :value="5">5</option><option :value="8">8</option><option :value="10">10</option></select></label><label>最低相似度 <b>{{ minSimilarity.toFixed(2) }}</b></label><input v-model.number="minSimilarity" type="range" min="0" max="1" step="0.01" /></details>
        </aside>
      </template>

      <section v-else-if="view === 'documents'" class="management-page">
        <header class="management-head"><div><p class="page-context">知识资产</p><h1>文档管理</h1><p>查看资料处理状态，管理用于问答的知识来源。</p></div><label class="upload-button"><Upload :size="17" />上传文档<input type="file" accept=".pdf,.txt,.md" multiple @change="handleUpload" /></label></header>
        <div class="mock-notice"><AlertCircle :size="17" /><span><strong>演示模式</strong>文档 CRUD 后端接口尚未接入，本页操作仅影响当前浏览器会话。</span></div>
        <div class="stats-strip"><div><span>文档总数</span><strong>{{ documents.length }}</strong></div><div><span>可用文档</span><strong>{{ readyCount }}</strong></div><div><span>可检索片段</span><strong>{{ totalChunks }}</strong></div><div><span>支持格式</span><strong>PDF · TXT · MD</strong></div></div>
        <div class="table-toolbar"><label><Search :size="17" /><input v-model="documentSearch" placeholder="搜索文件名" /></label><span>{{ filteredDocuments.length }} 个文档</span></div>
        <div class="document-table" role="table" aria-label="文档列表"><div class="table-row table-header" role="row"><span>文件</span><span>类型</span><span>大小</span><span>更新时间</span><span>处理状态</span><span>操作</span></div><div v-for="document in filteredDocuments" :key="document.id" class="table-row" role="row"><span class="file-cell"><FileText :size="19" /><span><b>{{ document.name }}</b><small>{{ document.chunks ? `${document.chunks} 个片段` : '尚无可检索片段' }}</small></span></span><span><em class="file-type">{{ document.type }}</em></span><span>{{ document.size }}</span><span>{{ document.updatedAt }}</span><span><em :class="['status-label', document.status]"><LoaderCircle v-if="document.status === 'processing'" :size="13" /><CircleCheck v-else-if="document.status === 'ready'" :size="13" /><AlertCircle v-else :size="13" />{{ document.status === 'ready' ? '可用' : document.status === 'processing' ? '处理中' : '失败' }}</em></span><span class="row-actions"><button title="重新解析" @click="reparse(document)"><RefreshCw :size="16" /></button><button title="删除" @click="confirmDelete = document"><Trash2 :size="16" /></button></span></div><div v-if="!filteredDocuments.length" class="table-empty"><Search :size="25" /><p>没有找到匹配的文档</p></div></div>
      </section>

      <section v-else-if="view === 'knowledge'" class="management-page">
        <header class="management-head"><div><p class="page-context">知识空间</p><h1>AI 技术学习库</h1><p>集中管理项目资料与 Day 1–29 学习笔记。</p></div><button class="primary-button" @click="chooseView('documents')"><FolderOpen :size="17" />管理文档</button></header>
        <div class="knowledge-overview"><div class="knowledge-icon"><Database :size="28" /></div><div><h2>本地中文 RAG 知识库</h2><p>使用 BAAI/bge-small-zh-v1.5 构建向量表示，通过相关性门控决定是否生成回答。</p></div><span class="status-label ready"><CircleCheck :size="13" />可用</span></div>
        <div class="knowledge-grid"><article><Database :size="20" /><span>知识源</span><strong>29</strong><small>4 篇基础资料 · 25 篇学习笔记</small></article><article><FileText :size="20" /><span>文本片段</span><strong>810</strong><small>最大 500 字符 · 80 字符重叠</small></article><article><Search :size="20" /><span>默认检索参数</span><strong>Top 5</strong><small>相关性门槛 0.48</small></article></div>
      </section>

      <section v-else class="management-page settings-page"><header class="management-head"><div><p class="page-context">工作区</p><h1>个人设置</h1><p>调整本地问答偏好。</p></div></header><div class="settings-card"><h2>默认检索设置</h2><label><span><b>召回数量</b><small>每次问答最多取回的文本片段数</small></span><select v-model="topK"><option :value="3">3 条</option><option :value="5">5 条</option><option :value="8">8 条</option><option :value="10">10 条</option></select></label><label><span><b>相关性门槛</b><small>低于门槛时明确拒答，避免无依据生成</small></span><b>{{ minSimilarity.toFixed(2) }}</b></label><input v-model.number="minSimilarity" type="range" min="0" max="1" step="0.01" /></div></section>
    </main>

    <div v-if="confirmDelete" class="modal-backdrop" @click.self="confirmDelete = null"><section class="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title"><span class="danger-icon"><Trash2 :size="20" /></span><h2 id="delete-title">删除演示文档？</h2><p>将从当前浏览器的演示列表中移除“{{ confirmDelete.name }}”，不会删除本地真实文件。</p><div><button class="secondary-button" @click="confirmDelete = null">取消</button><button class="danger-button" @click="deleteDocument">确认移除</button></div></section></div>
    <transition name="toast"><div v-if="toast" class="toast-message" role="status"><Check :size="16" />{{ toast }}</div></transition>
  </div>
</template>
