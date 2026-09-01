/* 后端 API 封装(经 Vite 代理 /api → 127.0.0.1:8000) */

async function jfetch(url, opts = {}) {
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${resp.status}`)
  }
  return resp.status === 204 ? null : resp.json()
}

export const api = {
  health: () => jfetch('/api/health'),
  listSessions: () => jfetch('/api/sessions'),
  createSession: (title) =>
    jfetch('/api/sessions', { method: 'POST', body: JSON.stringify({ title }) }),
  renameSession: (id, title) =>
    jfetch(`/api/sessions/${id}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  deleteSession: (id) => jfetch(`/api/sessions/${id}`, { method: 'DELETE' }),
  getMessages: (id) => jfetch(`/api/sessions/${id}/messages`),
}

/**
 * SSE 流式对话。POST /api/chat,解析 event/data 帧:
 * delta(增量文本) / citations(引用来源) / done(收尾) / error
 */
export async function streamChat({ sessionId, content, onDelta, onCitations, onDone, onError }) {
  let resp
  try {
    resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, content }),
    })
  } catch (e) {
    onError?.(`网络错误:${e.message}`)
    return
  }
  if (!resp.ok || !resp.body) {
    const err = await resp.json().catch(() => ({}))
    onError?.(err.detail || `HTTP ${resp.status}`)
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const block = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        let event = 'message'
        let data = ''
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          else if (line.startsWith('data:')) data += line.slice(5).trim()
        }
        if (!data) continue
        let payload
        try {
          payload = JSON.parse(data)
        } catch {
          continue
        }
        if (event === 'delta') onDelta?.(payload.text || '')
        else if (event === 'citations') onCitations?.(payload.citations || [])
        else if (event === 'done') onDone?.(payload)
        else if (event === 'error') onError?.(payload.message || '未知错误')
      }
    }
  } catch (e) {
    onError?.(`流中断:${e.message}`)
  }
}
