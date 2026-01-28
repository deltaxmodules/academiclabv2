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
  const [showReuploadModal, setShowReuploadModal] = useState(false);

  const statusLabel = useMemo(() => {
    if (status === "connected") return "✅ Connected";
    if (status === "connecting") return "⏳ Connecting";
    if (status === "error") return "⚠️ Error";
    return "🕒 Waiting";
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
          payload.content.toLowerCase().includes("upload the updated dataset")
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
        setCsvInfo({
          ...data.dataset_info,
          csv_version: data.csv_version || 1,
        });
        setMessages((prev) => {
          const next = sessionId ? [...prev] : [];
          const action = sessionId ? "reupload" : "analyze";
          const last = next[next.length - 1];
          if (last && last.action === action && last.content === data.message) {
            return next;
          }
          next.push({
            id: `${Date.now()}-upload`,
            role: "assistant",
            content: data.message,
            action,
          });
          return next;
        });
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

  const handleOpenReupload = () => {
    setShowReuploadModal(true);
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
            An educational LangGraph tutor to guide students through data
            preparation before analysis.
          </p>
        </div>
        <div className="hero-card">
          <h2>Guided flow</h2>
          <ul>
            <li>P01-P35 diagnostics</li>
            <li>Structured explanations</li>
            <li>Examples and reflection</li>
            <li>Checklist CHK-001-035</li>
          </ul>
        </div>
      </div>

      <main className="workspace">
        {!sessionId && (
          <section className="upload-card">
            <h3>1. Upload your CSV</h3>
            <p>The tutor analyzes the dataset and orders issues by severity.</p>
            <label className="upload-button">
              <input
                type="file"
                accept=".csv"
                onChange={handleCsvUpload}
              />
              Select file
            </label>
            {loading && <span className="status">Analyzing...</span>}
          </section>
        )}

        {sessionId && (
          <section className="chat-card">
            <div className="chat-header">
              <div>
                <h3>2. Chat with the tutor</h3>
                <span className="status-pill">{statusLabel}</span>
              </div>
              {csvInfo && (
                <div className="dataset-meta">
                  <span>{csvInfo.rows} rows</span>
                  <span>{csvInfo.columns} columns</span>
                  <span>{csvInfo.memory_mb} MB</span>
                </div>
              )}
            </div>
            <div className="chat-body">
              {messages.map((msg) => (
                <div key={msg.id} className={`bubble ${msg.role}`}>
                  <div className="bubble-header">
                    <span>{msg.role === "user" ? "👤 You" : "🤖 Tutor"}</span>
                    {msg.action && <span className="action-tag">{msg.action}</span>}
                  </div>
                  <pre>{msg.content}</pre>
                </div>
              ))}
              {loading && (
                <div className="bubble assistant">
                  <div className="bubble-header">🤖 Tutor</div>
                  <pre>⏳ Thinking...</pre>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="chat-input">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question or choose a P##..."
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                disabled={loading || status !== "connected"}
              />
              <button onClick={handleSend} disabled={loading || status !== "connected"}>
                Send
              </button>
              <button
                className="reupload-button"
                type="button"
                onClick={handleOpenReupload}
              >
                Re-evaluate CSV
              </button>
            </div>
          </section>
        )}
      </main>

      {showReuploadModal && sessionId && (
        <div className="modal-backdrop" onClick={() => setShowReuploadModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3>📁 File to edit:</h3>
            <p className="path">{`dataset_v${csvInfo?.csv_version || "?"}.csv`}</p>
            <p>📝 Fix the issue in Jupyter and bring the updated file here.</p>
            <p className="path-hint">
              Save the updated file in the same folder you selected for the first upload.
            </p>
            <pre className="code-block">{`import pandas as pd

raw_df = pd.read_csv("dataset_v${csvInfo?.csv_version || "?"}.csv")
df = raw_df.copy()
# ... apply fixes on df ...
df.to_csv("dataset_v${(csvInfo?.csv_version || 1) + 1}.csv", index=False)
`}</pre>
            <div className="modal-actions">
              <label className="upload-button">
                <input type="file" accept=".csv" onChange={(e) => { handleCsvUpload(e); setShowReuploadModal(false); }} />
                ✅ I have the new file
              </label>
              <button className="ghost" onClick={() => setShowReuploadModal(false)}>
                ❌ Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
