import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL = API_URL.replace("http", "ws");

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [csvInfo, setCsvInfo] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("idle");
  const [loading, setLoading] = useState(false);
  const [reuploadReady, setReuploadReady] = useState(false);
  const wsRef = useRef(null);
  const chatEndRef = useRef(null);

  const statusLabel = useMemo(() => {
    if (status === "connected") return "✅ Conectado";
    if (status === "connecting") return "⏳ Conectando";
    if (status === "error") return "⚠️ Erro";
    return "🕒 Aguardando";
  }, [status]);

  const connectWebSocket = (sid) => {
    setStatus("connecting");
    wsRef.current = new WebSocket(`${WS_URL}/chat/${sid}`);

    wsRef.current.onopen = () => setStatus("connected");
    wsRef.current.onerror = () => setStatus("error");
    wsRef.current.onclose = () => setStatus("idle");

    wsRef.current.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "response") {
        setMessages((prev) => [
          ...prev,
          {
            id: `${Date.now()}-${prev.length}`,
            role: "assistant",
            content: payload.content,
            action: payload.action,
          },
        ]);
        if (
          payload.reupload_required ||
          payload.action === "mark_solved" ||
          payload.content.toLowerCase().includes("faça upload do dataset atualizado")
        ) {
          setReuploadReady(true);
        }
        setLoading(false);
      }
    };
  };

  const handleCsvUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const endpoint = sessionId
        ? `${API_URL}/reupload/${sessionId}`
        : `${API_URL}/upload`;
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.success) {
        setSessionId(data.session_id);
        setCsvInfo(data.dataset_info);
        setMessages([
          {
            id: "initial",
            role: "assistant",
            content: data.message,
            action: "analyze",
          },
        ]);
        if (!sessionId) {
          connectWebSocket(data.session_id);
        }
        setReuploadReady(false);
      } else {
        setStatus("error");
      }
    } catch (err) {
      console.error(err);
      setStatus("error");
    } finally {
      setLoading(false);
    }
  };

  const handleSend = () => {
    if (!input.trim() || !wsRef.current) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);

    setMessages((prev) => [
      ...prev,
      { id: `${Date.now()}-${prev.length}`, role: "user", content: userMessage },
    ]);

    wsRef.current.send(JSON.stringify({ message: userMessage }));
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, loading]);

  return (
    <div className="app">
      <div className="hero">
        <div className="brand">
          <span className="badge">AcademicLab Edu</span>
          <h1>Data Preparation Tutor</h1>
          <p>
            Plataforma educativa com LangGraph para guiar estudantes na preparação
            de dados antes da análise.
          </p>
        </div>
        <div className="hero-card">
          <h2>Fluxo guiado</h2>
          <ul>
            <li>Diagnóstico P01-P35</li>
            <li>Explicação estruturada</li>
            <li>Exemplos e reflexão</li>
            <li>Checklist CHK-001-035</li>
          </ul>
        </div>
      </div>

      <main className="workspace">
        {!sessionId && (
          <section className="upload-card">
            <h3>1. Faça upload do CSV</h3>
            <p>O tutor analisa o dataset e organiza os problemas por severidade.</p>
            <label className="upload-button">
              <input type="file" accept=".csv" onChange={handleCsvUpload} />
              Selecionar arquivo
            </label>
            {loading && <span className="status">Analisando...</span>}
          </section>
        )}

        {sessionId && (
          <section className="chat-card">
            <div className="chat-header">
              <div>
                <h3>2. Converse com o tutor</h3>
                <span className="status-pill">{statusLabel}</span>
              </div>
              {csvInfo && (
                <div className="dataset-meta">
                  <span>{csvInfo.rows} linhas</span>
                  <span>{csvInfo.columns} colunas</span>
                  <span>{csvInfo.memory_mb} MB</span>
                </div>
              )}
            </div>

            <div className="chat-body">
              {messages.map((msg) => (
                <div key={msg.id} className={`bubble ${msg.role}`}>
                  <div className="bubble-header">
                    <span>{msg.role === "user" ? "👤 Você" : "🤖 Tutor"}</span>
                    {msg.action && <span className="action-tag">{msg.action}</span>}
                  </div>
                  <pre>{msg.content}</pre>
                </div>
              ))}
              {loading && (
                <div className="bubble assistant">
                  <div className="bubble-header">🤖 Tutor</div>
                  <pre>⏳ Pensando...</pre>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="chat-input">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Faça uma pergunta ou escolha um P##..."
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                disabled={loading || status !== "connected"}
              />
              <button onClick={handleSend} disabled={loading || status !== "connected"}>
                Enviar
              </button>
              <label className="reupload-button">
                <input type="file" accept=".csv" onChange={handleCsvUpload} />
                Reavaliar CSV
              </label>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
