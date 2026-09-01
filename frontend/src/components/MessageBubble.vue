<template>
  <div class="msg" :class="msg.role">
    <el-avatar v-if="msg.role === 'assistant'" :size="34" class="avatar">客</el-avatar>
    <div class="bubble chat-bubble" :class="msg.role">
      <template v-if="msg.role === 'user'">{{ msg.content }}</template>
      <template v-else>
        <div v-html="rendered"></div>
        <span v-if="msg.streaming" class="stream-cursor" />

        <el-collapse v-if="msg.citations && msg.citations.length" class="citations">
          <el-collapse-item :title="`参考知识(${msg.citations.length})`">
            <div v-for="(c, i) in msg.citations" :key="i" class="citation">
              <div class="citation-q">{{ i + 1 }}. {{ c.question || c.source }}</div>
              <div class="citation-meta">
                <el-tag size="small" effect="plain">{{ c.source }}</el-tag>
                <span class="score">相关度 {{ (c.score * 100).toFixed(0) }}%</span>
              </div>
              <div class="citation-snippet">{{ c.snippet }}</div>
            </div>
          </el-collapse-item>
        </el-collapse>
      </template>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'

const props = defineProps({
  msg: { type: Object, required: true },
})

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return `<pre class="hljs"><code>${hljs.highlight(code, { language: lang }).value}</code></pre>`
      } catch {
        /* fallthrough */
      }
    }
    return `<pre class="hljs"><code>${md.utils.escapeHtml(code)}</code></pre>`
  },
})

const rendered = computed(() =>
  props.msg.role === 'assistant' ? md.render(props.msg.content || '') : ''
)
</script>

<style scoped>
.msg {
  display: flex;
  margin-bottom: 16px;
  gap: 10px;
}
.msg.user {
  justify-content: flex-end;
}
.avatar {
  flex-shrink: 0;
  background: #409eff;
}
.bubble {
  flex-shrink: 1;
}
.citations {
  margin-top: 10px;
  border-left: 3px solid #d9ecff;
  --el-collapse-header-height: 34px;
}
.citations :deep(.el-collapse-item__header) {
  font-size: 13px;
  color: #606266;
  background: #f8fafc;
  padding: 0 10px;
  border-radius: 6px;
}
.citation {
  padding: 6px 2px;
  border-bottom: 1px dashed #ebeef5;
}
.citation:last-child {
  border-bottom: none;
}
.citation-q {
  font-size: 13px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 4px;
}
.citation-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}
.score {
  font-size: 12px;
  color: #909399;
}
.citation-snippet {
  font-size: 12px;
  color: #606266;
  line-height: 1.5;
}
</style>
