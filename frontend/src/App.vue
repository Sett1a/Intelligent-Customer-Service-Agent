<template>
  <el-container class="layout">
    <SessionSidebar />
    <el-main class="main">
      <ChatWindow v-if="store.currentId" :key="store.currentId" />
      <div v-else class="placeholder">
        <el-empty description="选择左侧会话,或点击「新建会话」开始咨询" />
      </div>
    </el-main>
  </el-container>
</template>

<script setup>
import { onMounted } from 'vue'
import SessionSidebar from './components/SessionSidebar.vue'
import ChatWindow from './components/ChatWindow.vue'
import { api } from './api'
import { loadSessions, store } from './store'

onMounted(async () => {
  try {
    const h = await api.health()
    store.hasKey = h.has_key
  } catch {
    /* 后端未启动时侧栏/输入自行报错 */
  }
  loadSessions().catch(() => {})
})
</script>

<style scoped>
.layout {
  height: 100%;
}
.main {
  padding: 0;
  display: flex;
  flex-direction: column;
}
.placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
