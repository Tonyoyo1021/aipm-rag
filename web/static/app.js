const elements = {
  form: document.querySelector('#question-form'),
  input: document.querySelector('#question-input'),
  send: document.querySelector('#send-button'),
  messages: document.querySelector('#messages'),
  empty: document.querySelector('#empty-state'),
  conversation: document.querySelector('#conversation'),
  sourcesContent: document.querySelector('#sources-content'),
  sourcesSummary: document.querySelector('#sources-summary'),
  sourcesPanel: document.querySelector('#sources-panel'),
  sourcesToggle: document.querySelector('#sources-toggle'),
  sourcesClose: document.querySelector('#sources-close'),
  sourcesBackdrop: document.querySelector('#sources-backdrop'),
  sourceCountMobile: document.querySelector('#source-count-mobile'),
  errorBanner: document.querySelector('#error-banner'),
  errorText: document.querySelector('#error-text'),
  retry: document.querySelector('#retry-button'),
  newChat: document.querySelector('#new-chat'),
  topK: document.querySelector('#top-k'),
};

let loading = false;
let lastQuestion = '';
let loadingElement = null;

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function inlineFormat(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[(\d+)\]/g, '<button type="button" class="citation" data-source="$1" aria-label="查看来源 $1">[$1]</button>');
}

function renderMarkdown(text) {
  const lines = text.split('\n');
  const html = [];
  let listType = null;
  let paragraph = [];

  const closeParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${paragraph.map(inlineFormat).join('<br>')}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (!trimmed) {
      closeParagraph();
      closeList();
    } else if (heading) {
      closeParagraph();
      closeList();
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${inlineFormat(heading[2])}</h${level}>`);
    } else if (unordered || ordered) {
      closeParagraph();
      const nextType = unordered ? 'ul' : 'ol';
      if (listType !== nextType) {
        closeList();
        html.push(`<${nextType}>`);
        listType = nextType;
      }
      html.push(`<li>${inlineFormat((unordered || ordered)[1])}</li>`);
    } else {
      closeList();
      paragraph.push(trimmed);
    }
  }
  closeParagraph();
  closeList();
  return html.join('');
}

function documentIcon() {
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h9l4 4v14H6zM14 3v5h5M9 12h6M9 16h6"/></svg>';
}

async function loadStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('status failed');
    const status = await response.json();
    document.querySelector('#document-count').textContent = `${status.document_count} 份`;
    document.querySelector('#chunk-count').textContent = status.chunk_count;
    document.querySelector('#document-list').innerHTML = status.documents
      .map((name) => `<li>${documentIcon()}<span>${escapeHtml(name)}</span></li>`)
      .join('');
    document.querySelector('#top-k').value = String(status.top_k);
    document.querySelector('#model-names').textContent = `${status.embedding_model} · ${status.chat_model}`;
    const ready = status.ready && status.configured;
    const label = status.ready ? (status.configured ? '本地索引已就绪' : '等待模型配置') : '等待构建索引';
    for (const id of ['#side-status', '#header-status']) document.querySelector(id).textContent = label;
    for (const id of ['#side-status-dot', '#header-status-dot']) {
      document.querySelector(id).classList.toggle('ready', ready);
      document.querySelector(id).classList.toggle('error', !status.configured);
    }
  } catch {
    document.querySelector('#side-status').textContent = '状态读取失败';
    document.querySelector('#header-status').textContent = '连接失败';
  }
}

function resizeInput() {
  elements.input.style.height = 'auto';
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 150)}px`;
}

function showError(message) {
  elements.errorText.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  elements.errorBanner.hidden = true;
  elements.errorText.textContent = '';
}

function setLoading(value) {
  loading = value;
  elements.send.disabled = value;
  elements.input.disabled = value;
  elements.topK.disabled = value;
  elements.newChat.disabled = value;
  elements.send.querySelector('span').textContent = value ? '正在生成' : '发送问题';
}

function scrollToLatest() {
  requestAnimationFrame(() => {
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
  });
}

function appendUserMessage(question) {
  const article = document.createElement('article');
  article.className = 'message user-message';
  const bubble = document.createElement('div');
  bubble.className = 'user-bubble';
  bubble.textContent = question;
  article.append(bubble);
  elements.messages.append(article);
}

function assistantShell() {
  const article = document.createElement('article');
  article.className = 'message assistant-message';
  article.innerHTML = `
    <div class="assistant-header">
      <div class="assistant-identity">
        <span class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></span>
        <strong>知库</strong>
      </div>
      <span class="assistant-meta"></span>
    </div>
    <div class="answer-body"></div>`;
  return article;
}

function showLoading() {
  loadingElement = assistantShell();
  loadingElement.querySelector('.assistant-meta').textContent = '检索中';
  loadingElement.querySelector('.answer-body').innerHTML = `
    <div class="loading-answer" aria-label="正在检索并生成回答">
      <p class="loading-label">正在检索资料并生成回答，免费模型可能需要一些时间…</p>
      <div class="skeleton-line"></div><div class="skeleton-line"></div>
      <div class="skeleton-line"></div><div class="skeleton-line"></div>
    </div>`;
  elements.messages.append(loadingElement);
  elements.sourcesSummary.textContent = '正在检索知识库';
  elements.sourcesContent.innerHTML = '<div class="loading-sources"><div class="skeleton-line"></div><div class="skeleton-line"></div><div class="skeleton-line"></div></div>';
  scrollToLatest();
}

function renderSources(results) {
  elements.sourcesSummary.textContent = `本次检索 · ${results.length} 段资料`;
  elements.sourceCountMobile.textContent = `(${results.length})`;
  elements.sourcesContent.innerHTML = '';
  results.forEach((result, index) => {
    const number = index + 1;
    const headings = result.heading_path.split(' > ');
    const title = headings.at(-1) || '未命名章节';
    const item = document.createElement('details');
    item.className = 'source-item';
    item.id = `source-${number}`;
    item.innerHTML = `
      <summary>
        <div class="source-topline"><span class="source-number">${number}</span><span class="source-score">相似度 ${(result.score * 100).toFixed(1)}%</span></div>
        <h3>${escapeHtml(title)}</h3>
        <span class="source-file">${escapeHtml(result.source)}</span>
        <div class="score-track" aria-hidden="true"><span style="width:${Math.max(0, Math.min(100, result.score * 100))}%"></span></div>
      </summary>
      <div class="source-text"><p>${escapeHtml(result.heading_path)}</p><pre></pre></div>`;
    item.querySelector('pre').textContent = result.text;
    elements.sourcesContent.append(item);
  });
}

function focusSource(number) {
  const item = document.querySelector(`#source-${number}`);
  if (!item) return;
  item.open = true;
  document.querySelectorAll('.source-item.highlight').forEach((source) => source.classList.remove('highlight'));
  item.classList.add('highlight');
  if (window.innerWidth <= 1180) openSources();
  requestAnimationFrame(() => item.scrollIntoView({ behavior: 'smooth', block: 'start' }));
}

async function copyAnswer(button, text) {
  try {
    await navigator.clipboard.writeText(text);
    button.setAttribute('aria-label', '已复制');
    button.textContent = '已复制';
    setTimeout(() => {
      button.textContent = '';
      button.append(createCopyIcon());
      button.setAttribute('aria-label', '复制回答');
    }, 1600);
  } catch {
    showError('浏览器未允许复制，请手动选择回答文本。');
  }
}

function createCopyIcon() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  svg.innerHTML = '<rect x="8" y="8" width="11" height="12" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h3"/>';
  return svg;
}

function appendAssistantMessage(data) {
  const article = assistantShell();
  const seconds = (data.meta.elapsed_ms / 1000).toFixed(1);
  article.querySelector('.assistant-meta').textContent = `${seconds}s · ${data.results.length} 段资料`;
  const body = article.querySelector('.answer-body');
  body.className = 'answer-content';
  body.innerHTML = renderMarkdown(data.answer);
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'icon-button';
  copy.setAttribute('aria-label', '复制回答');
  copy.append(createCopyIcon());
  copy.addEventListener('click', () => copyAnswer(copy, data.answer));
  article.querySelector('.assistant-header').append(copy);
  body.querySelectorAll('.citation').forEach((button) => {
    button.addEventListener('click', () => focusSource(button.dataset.source));
  });
  loadingElement.replaceWith(article);
  loadingElement = null;
  scrollToLatest();
}

async function askQuestion(question) {
  const cleaned = question.trim();
  if (!cleaned || loading) return;
  lastQuestion = cleaned;
  clearError();
  elements.empty.hidden = true;
  appendUserMessage(cleaned);
  elements.input.value = '';
  resizeInput();
  setLoading(true);
  showLoading();
  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: cleaned, top_k: Number(elements.topK.value) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '请求失败，请稍后重试。');
    appendAssistantMessage(data);
    renderSources(data.results);
  } catch (error) {
    if (loadingElement) {
      loadingElement.remove();
      loadingElement = null;
    }
    elements.sourcesSummary.textContent = '本次检索未完成';
    elements.sourcesContent.innerHTML = '<div class="sources-empty"><h3>暂时没有检索结果</h3><p>修正错误后重试，历史问题仍会保留。</p></div>';
    showError(error.message || '网络连接失败，请确认 Web 服务正在运行。');
  } finally {
    setLoading(false);
    elements.input.focus();
    scrollToLatest();
  }
}

function openSources() {
  elements.sourcesPanel.classList.add('is-open');
  elements.sourcesToggle.setAttribute('aria-expanded', 'true');
  if (window.innerWidth <= 1180) elements.sourcesBackdrop.hidden = false;
}

function closeSources() {
  elements.sourcesPanel.classList.remove('is-open');
  elements.sourcesToggle.setAttribute('aria-expanded', 'false');
  elements.sourcesBackdrop.hidden = true;
}

function resetConversation() {
  if (loading) return;
  elements.messages.innerHTML = '';
  elements.empty.hidden = false;
  elements.sourcesSummary.textContent = '等待一次提问';
  elements.sourcesContent.innerHTML = '<div class="sources-empty"><h3>答案旁边，就是依据</h3><p>提问后会展示 Top-K 片段、相似度和标题路径。点击引用编号即可定位。</p></div>';
  elements.sourceCountMobile.textContent = '';
  elements.input.value = '';
  clearError();
  closeSources();
  window.history.replaceState({}, '', '/');
  resizeInput();
  elements.input.focus();
}

elements.form.addEventListener('submit', (event) => {
  event.preventDefault();
  askQuestion(elements.input.value);
});
elements.input.addEventListener('input', resizeInput);
elements.input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    askQuestion(elements.input.value);
  }
});
elements.retry.addEventListener('click', () => askQuestion(lastQuestion));
elements.newChat.addEventListener('click', resetConversation);
elements.sourcesToggle.addEventListener('click', () => {
  if (elements.sourcesPanel.classList.contains('is-open')) closeSources();
  else openSources();
});
elements.sourcesClose.addEventListener('click', closeSources);
elements.sourcesBackdrop.addEventListener('click', closeSources);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeSources();
});
document.querySelectorAll('.suggestion').forEach((button) => {
  button.addEventListener('click', () => askQuestion(button.dataset.question));
});

loadStatus();
resizeInput();
const initialQuestion = new URLSearchParams(window.location.search).get('q');
if (initialQuestion) askQuestion(initialQuestion);
