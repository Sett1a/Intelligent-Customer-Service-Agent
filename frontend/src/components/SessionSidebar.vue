<template>
  <el-aside width="280px" class="sidebar">
    <div class="head">
      <span class="brand">🤖 客服 Agent</span>
      <el-button type="primary" size="small" :icon="Plus" @click="onNew">新建会话</el-button>
    </div>

    <div class="list">
      <div
        v-for="s in store.sessions"
        :key="s.id"
        class="item"
        :class="{ active: s.id === store.currentId }"
        @click="onSelect(s)"
      >
        <div class="row1">
          <span class="title">{{ s.title }}</span>
          <el-tag v-if="s.handoff" type="warning" size="small" effect="light">已转人工</el-tag>
        </div>
        <div class="row2">
          <span class="time">{{ fmtTime(s.updated_at) }}</span>
          <span class="ops">
            <el-button link size="small" :icon="Edit" title="重命名" @click.stop="onRename(s)" />
            <el-button link size="small" type="danger" :icon="Delete" title="删除" @click.stop="onDelete(s)" />
          </span>
        </div>
      </div>
      <el-empty v-if="!store.sessions.length" description="暂无会话" :image-size="70" />
    </div>
  </el-aside>
</template>

<script setup>
import { Delete, Edit, Plus } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { loadSessions, newSession, selectSession, store } from '../store'

function fmtTime(iso) {
  const d = new Date(iso)
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getMonth() + 1}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

async function onNew() {
  try {
    await newSession()
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function onSelect(s) {
  if (s.id === store.currentId) return
  try {
    await selectSession(s.id)
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function onRename(s) {
  try {
    const { value } = await ElMessageBox.prompt('输入新的会话名称', '重命名', {
      inputValue: s.title,
      inputPattern: /\S+/,
      inputErrorMessage: '名称不能为空',
    })
    await import('../api').then(({ api }) => api.renameSession(s.id, value.trim()))
    s.title = value.trim()
  } catch (e) {
    if (e !== 'cancel' && e?.message) ElMessage.error(e.message)
  }
}

async function onDelete(s) {
  try {
    await ElMessageBox.confirm(`确定删除会话「${s.title}」及其全部消息?`, '删除会话', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  try {
    await import('../api').then(({ api }) => api.deleteSession(s.id))
    store.sessions = store.sessions.filter((x) => x.id !== s.id)
    if (store.currentId === s.id) {
      store.currentId = null
      store.messages = []
    }
    ElMessage.success('已删除')
  } catch (e) {
    ElMessage.error(e.message)
  }
}
</script>

<style scoped>
.sidebar {
  background: #fff;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.head {
  padding: 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #f0f2f5;
}
.brand {
  font-weight: 600;
  font-size: 15px;
}
.list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}
.item {
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 4px;
}
.item:hover {
  background: #f5f7fa;
}
.item.active {
  background: #ecf5ff;
}
.row1 {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.title {
  font-size: 14px;
  color: #303133;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}
.row2 {
  margin-top: 4px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.time {
  font-size: 12px;
  color: #909399;
}
.ops {
  visibility: hidden;
}
.item:hover .ops {
  visibility: visible;
}
</style>
