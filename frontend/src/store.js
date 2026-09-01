/* 全局状态 + 动作:会话列表、当前会话、消息、流式状态 */

import { reactive } from 'vue'
import { api, streamChat } from './api'

export const store = reactive({
  sessions: [],
  currentId: null,
  messages: [],
  streaming: false,
  error: '',
  hasKey: true,
})

export function currentSession() {
  return store.sessions.find((s) => s.id === store.currentId) || null
}

export async function loadSessions() {
  store.sessions = await api.listSessions()
}

export async function selectSession(id) {
  store.currentId = id
  store.error = ''
  store.messages = await api.getMessages(id)
}

export async function newSession() {
  const s = await api.createSession()
  store.sessions.unshift(s)
  store.currentId = s.id
  store.messages = []
  return s
}

export async function send(content) {
  const text = content.trim()
  if (!text || store.streaming) return
  if (!store.currentId) await newSession()
  const sid = store.currentId

  store.messages.push({
    id: -1,
    role: 'user',
    content: text,
    citations: null,
    created_at: new Date().toISOString(),
  })
  store.streaming = true
  store.error = ''

  let assistant = null
  const ensureAssistant = () => {
    if (!assistant) {
      assistant = {
        id: -2,
        role: 'assistant',
        content: '',
        citations: [],
        streaming: true,
        created_at: new Date().toISOString(),
      }
      store.messages.push(assistant)
    }
    return assistant
  }

  await streamChat({
    sessionId: sid,
    content: text,
    onDelta: (t) => {
      ensureAssistant().content += t
    },
    onCitations: (c) => {
      if (c.length) ensureAssistant().citations = c
    },
    onDone: () => {
      if (assistant) assistant.streaming = false
      loadSessions().catch(() => {}) // 刷新标题/转人工标记/排序
    },
    onError: (m) => {
      store.error = m
      if (assistant) assistant.streaming = false
    },
  })

  // 以服务端为准刷新消息(拿到真实 id 与持久化的引用)
  try {
    store.messages = await api.getMessages(sid)
  } catch {
    /* 会话可能已被删除 */
  }
  store.streaming = false
}
