<template>
  <div class="chat">
    <el-alert
      v-if="!store.hasKey"
      type="warning"
      :closable="false"
      show-icon
      class="banner"
      title="未配置 ZHIPU_API_KEY"
      description="请编辑 backend/.env 填入 API Key 后重启后端;也可置 EMBED_FAKE=1 干跑检索链路(对话仍需 Key)。"
    />
    <el-alert
      v-else-if="current?.handoff"
      type="warning"
      :closable="false"
      show-icon
      class="banner"
      title="本会话已转接人工客服"
      description="AI 客服将继续响应,人工客服接入前请耐心等待。"
    />

    <div ref="scrollEl" class="messages">
      <MessageBubble v-for="m in store.messages" :key="m.id" :msg="m" />
      <el-empty
        v-if="!store.messages.length && !store.streaming"
        description="有什么可以帮您?"
        :image-size="90"
      />
    </div>

    <div v-if="store.error" class="error-bar">
      <el-alert type="error" :title="store.error" @close="store.error = ''" />
    </div>

    <div class="input-area">
      <el-input
        v-model="draft"
        type="textarea"
        resize="none"
        :autosize="{ minRows: 1, maxRows: 5 }"
        placeholder="输入消息,Enter 发送,Shift+Enter 换行"
        @keydown="onKeydown"
      />
      <el-button
        type="primary"
        :icon="Promotion"
        :disabled="store.streaming || !draft.trim()"
        @click="onSend"
      >
        发送
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { Promotion } from '@element-plus/icons-vue'
import MessageBubble from './MessageBubble.vue'
import { currentSession, send, store } from '../store'

const draft = ref('')
const scrollEl = ref(null)
const current = computed(() => currentSession())

function onKeydown(e) {
  // 中文输入法组词阶段的 Enter 不发送
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    onSend()
  }
}

async function onSend() {
  const text = draft.value.trim()
  if (!text || store.streaming) return
  draft.value = ''
  await send(text)
}

watch(
  () => store.messages.map((m) => m.content.length + m.role).join('|') + store.messages.length,
  () => nextTick(() => scrollEl.value?.scrollTo({ top: scrollEl.value.scrollHeight }))
)
</script>

<style scoped>
.chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.banner {
  border-radius: 0;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
}
.error-bar {
  padding: 0 24px 6px;
}
.input-area {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  padding: 12px 24px 16px;
  border-top: 1px solid #e4e7ed;
  background: #fff;
}
</style>
