import { useEffect, useMemo, useRef, useState } from 'react'

type ChatSource = { title: string; url?: string; section_hint?: string; score?: number }

type ChatResponse = {
  query: string
  visa_type: string
  answer: string
  sources: ChatSource[]
  num_sources: number
}

type Message = {
  role: 'user' | 'assistant'
  text: string
  visa?: string
  sources?: ChatSource[]
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([{
    role: 'assistant',
    text: "Hello! I'm your Immigration Guardian. Ask about F-1, F-2, H-1B, H-4, J-1, or J-2 laws, and I'll answer with citations.",
  }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const chatEndRef = useRef<HTMLDivElement | null>(null)

  const API_BASE = useMemo(() => (import.meta.env.VITE_API_BASE as string) || 'http://127.0.0.1:8000', [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage() {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', text: q }])
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data: ChatResponse = await res.json()
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text: data.answer,
          visa: data.visa_type,
          sources: data.sources || [],
        },
      ])
    } catch (err: any) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: `Error: ${err?.message || 'Failed to contact API'}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') sendMessage()
  }

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <div style={styles.header}>🤖 Immigration Guardian</div>
        <div style={styles.chat}>
          {messages.map((m, i) => (
            <div key={i} style={{ ...styles.message, ...(m.role === 'user' ? styles.user : styles.bot) }}>
              <div style={styles.messageText}>{m.text}</div>
              {m.role === 'assistant' && m.visa ? (
                <span style={styles.badge}>{m.visa}</span>
              ) : null}
              {m.role === 'assistant' && m.sources && m.sources.length > 0 ? (
                <div style={styles.sources}>
                  <div style={{ fontWeight: 600, marginBottom: 6 }}>Sources</div>
                  <ul style={{ paddingLeft: 18, margin: 0 }}>
                    {m.sources.slice(0, 3).map((s, idx) => (
                      <li key={idx} style={{ marginBottom: 4 }}>
                        {s.url ? (
                          <a href={s.url} target="_blank" rel="noreferrer" style={{ color: '#2563eb' }}>
                            {s.title || s.url}
                          </a>
                        ) : (
                          <span>{s.title || s.section_hint}</span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
        <div style={styles.inputRow}>
          <input
            style={styles.input}
            placeholder="Ask about immigration laws..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={loading}
          />
          <button style={styles.button} onClick={sendMessage} disabled={loading}>
            {loading ? 'Thinking…' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg,rgb(247, 247, 247) 0%,rgb(133, 133, 201) 100%)',
    padding: 16,
  },
  card: {
    width: '100%',
    maxWidth: 900,
    background: '#fff',
    borderRadius: 16,
    boxShadow: '0 20px 40px rgba(45, 164, 220, 0.2)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    padding: '16px 20px',
    background: 'linear-gradient(135deg, #1f2937 0%,rgb(69, 110, 175) 100%)',
    color: '#fff',
    fontWeight: 700,
    fontSize: 18,
  },
  chat: {
    padding: 16,
    height: 520,
    overflowY: 'auto',
    background: '#f8fafc',
  },
  message: {
    marginBottom: 12,
    padding: '12px 14px',
    borderRadius: 12,
    maxWidth: '80%',
    position: 'relative',
  },
  user: {
    marginLeft: 'auto',
    background: '#2563eb',
    color: '#fff',
  },
  bot: {
    background: '#ffffff',
    border: '1px solid #e5e7eb',
  },
  messageText: {
    whiteSpace: 'pre-wrap',
    lineHeight: 1.5,
  },
  badge: {
    position: 'absolute',
    top: 8,
    right: 8,
    background: '#10b981',
    color: '#fff',
    borderRadius: 12,
    padding: '2px 8px',
    fontSize: 12,
    fontWeight: 700,
  },
  sources: {
    marginTop: 10,
    fontSize: 13,
    color: '#374151',
  },
  inputRow: {
    display: 'flex',
    gap: 8,
    padding: 12,
    borderTop: '1px solid #e5e7eb',
  },
  input: {
    flex: 1,
    padding: '12px 14px',
    borderRadius: 999,
    border: '1px solid #e5e7eb',
    outline: 'none',
    fontSize: 16,
  },
  button: {
    padding: '12px 16px',
    borderRadius: 999,
    background: '#111827',
    color: '#fff',
    border: 'none',
    cursor: 'pointer',
    fontWeight: 700,
  },
}
