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
  const [copyNotice, setCopyNotice] = useState("");
  const [showResetModal, setShowResetModal] = useState(false);
  const [showNotesModal, setShowNotesModal] = useState(false);
  const [notesText, setNotesText] = useState("");
  const [notesTitle, setNotesTitle] = useState("");
  const [notesIndex, setNotesIndex] = useState([]);
  const [selectedNoteId, setSelectedNoteId] = useState(null);
  const [responseStyle, setResponseStyle] = useState("fast");
  const [expertMode, setExpertMode] = useState(false);
  const [showImageModal, setShowImageModal] = useState(false);
  const [imageDataUrl, setImageDataUrl] = useState("");
  const [imageAnalysis, setImageAnalysis] = useState("");
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState("");
  const imageModalRef = useRef(null);

  const statusLabel = useMemo(() => {
    if (status === "connected") return "Connected";
    if (status === "connecting") return "Connecting";
    if (status === "error") return "Error";
    return "Waiting";
  }, [status]);

  const StatusIcon = () => {
    if (status === "connected") return <Icon name="check" />;
    if (status === "connecting") return <Icon name="spinner" />;
    if (status === "error") return <Icon name="alert" />;
    return <Icon name="clock" />;
  };

  const Icon = ({ name, className = "" }) => {
    const icons = {
      check: (
        <path d="M5 13l4 4L19 7" />
      ),
      alert: (
        <>
          <path d="M12 9v4" />
          <path d="M12 17h.01" />
          <path d="M10.29 3.86l-7.4 12.8A1 1 0 003.76 18h16.48a1 1 0 00.87-1.5l-7.4-12.8a1 1 0 00-1.73 0z" />
        </>
      ),
      clock: (
        <>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7v5l3 2" />
        </>
      ),
      spinner: (
        <path d="M12 3a9 9 0 109 9" />
      ),
      user: (
        <>
          <path d="M20 21a8 8 0 10-16 0" />
          <circle cx="12" cy="8" r="4" />
        </>
      ),
      bot: (
        <>
          <rect x="4" y="6" width="16" height="12" rx="3" />
          <path d="M8 6V4" />
          <path d="M16 6V4" />
          <circle cx="9" cy="12" r="1" />
          <circle cx="15" cy="12" r="1" />
        </>
      ),
      file: (
        <>
          <path d="M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9z" />
          <path d="M14 3v6h6" />
        </>
      ),
      edit: (
        <>
          <path d="M12 20h9" />
          <path d="M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4 12.5-12.5z" />
        </>
      ),
      warn: (
        <>
          <path d="M12 9v4" />
          <path d="M12 17h.01" />
          <path d="M10.29 3.86l-7.4 12.8A1 1 0 003.76 18h16.48a1 1 0 00.87-1.5l-7.4-12.8a1 1 0 00-1.73 0z" />
        </>
      ),
      help: (
        <>
          <circle cx="12" cy="12" r="9" />
          <path d="M9.09 9a3 3 0 115.82 1c0 2-3 2-3 4" />
          <path d="M12 17h.01" />
        </>
      ),
      reset: (
        <>
          <path d="M3 12a9 9 0 0115-6" />
          <path d="M18 3v6h-6" />
        </>
      ),
      dot: (
        <circle cx="12" cy="12" r="4" />
      ),
      copy: (
        <>
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <rect x="4" y="4" width="11" height="11" rx="2" />
        </>
      ),
    };
    return (
      <span className={`icon ${className}`.trim()}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          {icons[name]}
        </svg>
      </span>
    );
  };


  const renderTextBlock = (text) => {
    const lines = text.split("\n");
    return (
      <div className="text-block">
        {lines.map((line, idx) => {
          const trimmed = line.trim();
          if (trimmed === "CRITICAL:") {
            return (
              <div key={`line-${idx}`} className="severity-line severity-critical">
                <Icon name="dot" />
                <span>CRITICAL</span>
              </div>
            );
          }
          if (trimmed === "HIGH:") {
            return (
              <div key={`line-${idx}`} className="severity-line severity-high">
                <Icon name="dot" />
                <span>HIGH</span>
              </div>
            );
          }
          if (trimmed === "MEDIUM:") {
            return (
              <div key={`line-${idx}`} className="severity-line severity-medium">
                <Icon name="dot" />
                <span>MEDIUM</span>
              </div>
            );
          }
          if (trimmed === "LOW:") {
            return (
              <div key={`line-${idx}`} className="severity-line severity-low">
                <Icon name="dot" />
                <span>LOW</span>
              </div>
            );
          }
          if (trimmed.startsWith("•") || trimmed.startsWith("-")) {
            const text = trimmed.replace(/^•\s?/, "").replace(/^-+\s?/, "");
            return (
              <div key={`line-${idx}`} className="bullet-line">
                <span className="bullet">•</span>
                <span>{text}</span>
              </div>
            );
          }
          return (
            <p key={`line-${idx}`} className="text-line">
              {line}
            </p>
          );
        })}
      </div>
    );
  };

  const renderMessage = (content) => {
    const parts = content.split("```");
    return parts.map((part, idx) => {
      if (idx % 2 === 1) {
        const cleaned = part.trim().replace(/^python\\s*\\n/i, "");
        return (
          <div key={`code-${idx}`} className="code-block-wrapper">
            <div className="code-label">Code</div>
            <button
              className="copy-button"
              type="button"
              onClick={() => {
                navigator.clipboard.writeText(cleaned.trim());
                setCopyNotice("Code copied to clipboard");
                setTimeout(() => setCopyNotice(""), 2000);
              }}
            >
              <Icon name="copy" />
              Copy
            </button>
            <pre className="code-block">{cleaned.trim()}</pre>
          </div>
        );
      }
      return <React.Fragment key={`text-${idx}`}>{renderTextBlock(part)}</React.Fragment>;
    });
  };

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

    const payloadMessage = expertMode ? `expert: ${userMessage}` : userMessage;
    wsRef.current.send(JSON.stringify({ message: payloadMessage }));
  };

  const handleOpenReupload = () => {
    setShowReuploadModal(true);
  };

  const handleOpenNotes = () => {
    const savedIndex = JSON.parse(localStorage.getItem("academiclab_notes_index") || "[]");
    setNotesIndex(savedIndex);
    if (savedIndex.length > 0) {
      const first = savedIndex[0];
      const saved = localStorage.getItem(`academiclab_notes_${first.id}`) || "";
      setSelectedNoteId(first.id);
      setNotesTitle(first.title || "Untitled");
      setNotesText(saved);
    } else {
      setSelectedNoteId(null);
      setNotesTitle("");
      setNotesText("");
    }
    setShowNotesModal(true);
  };

  const handleSaveNotes = () => {
    const id = selectedNoteId || `note_${Date.now()}`;
    const title = notesTitle.trim() || "Untitled";
    const updatedIndex = notesIndex.some((n) => n.id === id)
      ? notesIndex.map((n) => (n.id === id ? { ...n, title } : n))
      : [{ id, title }, ...notesIndex];

    localStorage.setItem(`academiclab_notes_${id}`, notesText);
    localStorage.setItem("academiclab_notes_index", JSON.stringify(updatedIndex));
    setNotesIndex(updatedIndex);
    setSelectedNoteId(id);
    setShowNotesModal(false);
  };

  useEffect(() => {
    if (!showNotesModal) return;
    const id = selectedNoteId || "draft";
    localStorage.setItem(`academiclab_notes_draft_${id}`, notesText);
  }, [notesText, showNotesModal, selectedNoteId]);

  const handleClearNotes = () => {
    if (!selectedNoteId) return;
    localStorage.removeItem(`academiclab_notes_${selectedNoteId}`);
    const updated = notesIndex.filter((n) => n.id !== selectedNoteId);
    localStorage.setItem("academiclab_notes_index", JSON.stringify(updated));
    setNotesIndex(updated);
    if (updated.length > 0) {
      const first = updated[0];
      const saved = localStorage.getItem(`academiclab_notes_${first.id}`) || "";
      setSelectedNoteId(first.id);
      setNotesTitle(first.title || "Untitled");
      setNotesText(saved);
    } else {
      setSelectedNoteId(null);
      setNotesTitle("");
      setNotesText("");
    }
  };

  const handleNewNote = () => {
    setSelectedNoteId(null);
    setNotesTitle("");
    setNotesText("");
  };

  const handleSelectNote = (id) => {
    const note = notesIndex.find((n) => n.id === id);
    const saved = localStorage.getItem(`academiclab_notes_${id}`) || "";
    setSelectedNoteId(id);
    setNotesTitle(note?.title || "Untitled");
    setNotesText(saved);
  };

  const handleExportNote = () => {
    const title = notesTitle.trim() || "notes";
    const blob = new Blob([notesText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleOpenImageModal = () => {
    setImageDataUrl("");
    setImageAnalysis("");
    setImageError("");
    setShowImageModal(true);
    setTimeout(() => imageModalRef.current?.focus(), 50);
  };

  const handlePasteImage = (event) => {
    const items = event.clipboardData?.items || [];
    const fileItem = Array.from(items).find((item) => item.type.startsWith("image/"));
    if (!fileItem) return;
    const file = fileItem.getAsFile();
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setImageDataUrl(reader.result);
      setImageError("");
    };
    reader.readAsDataURL(file);
    event.preventDefault();
  };

  useEffect(() => {
    if (!showImageModal) return;
    const handler = (event) => handlePasteImage(event);
    window.addEventListener("paste", handler);
    return () => window.removeEventListener("paste", handler);
  }, [showImageModal]);

  const handleAnalyzeImage = async () => {
    if (!imageDataUrl) return;
    setImageLoading(true);
    setImageError("");
    try {
      const base64 = imageDataUrl.split(",")[1] || "";
      const response = await fetch(`${API_URL}/analyze-image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64 }),
      });
      const data = await response.json();
      if (data.success) {
        setImageAnalysis(data.analysis || "");
      } else {
        setImageError(data.detail || "Failed to analyze image.");
      }
    } catch (err) {
      setImageError("Failed to analyze image.");
    } finally {
      setImageLoading(false);
    }
  };

  const handleCopyAnalysis = () => {
    if (!imageAnalysis) return;
    navigator.clipboard.writeText(imageAnalysis);
    setCopyNotice("Analysis copied to clipboard");
    setTimeout(() => setCopyNotice(""), 2000);
  };

  const handleResponseStyle = async (style) => {
    setResponseStyle(style);
    if (!sessionId) return;
    try {
      await fetch(`${API_URL}/session/${sessionId}/response-style`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ style }),
      });
    } catch (err) {
      console.error(err);
    }
  };

  const handleResetSession = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/session/${sessionId}/reset`, {
        method: "POST",
      });
      const data = await response.json();
      if (data.success) {
        setSessionId(data.session_id);
        setCsvInfo(null);
        setMessages([
          {
            id: `${Date.now()}-reset`,
            role: "assistant",
            content: data.message,
            action: "reset",
          },
        ]);
        setReuploadReady(false);
        setShowResetModal(false);
        if (wsRef.current) {
          wsRef.current.close();
        }
        connectWebSocket(data.session_id);
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
          <div className="brand-row">
            <svg className="brand-mark" viewBox="0 0 64 64" aria-hidden="true">
              <defs>
                <linearGradient id="flask-glass" x1="0" x2="1" y1="0" y2="1">
                  <stop offset="0%" stopColor="#d1fae5" />
                  <stop offset="100%" stopColor="#a7f3d0" />
                </linearGradient>
              </defs>
              <path d="M26 6h12v10l10 18c5 8 0 18-9 18H25c-9 0-14-10-9-18l10-18V6z" fill="url(#flask-glass)" stroke="#064e3b" strokeWidth="2.5" strokeLinejoin="round"></path>
              <path d="M22 24h20" stroke="#064e3b" strokeWidth="2.5" strokeLinecap="round"></path>
              <path d="M20 38c6-4 18-4 24 0" stroke="#064e3b" strokeWidth="2.5" strokeLinecap="round"></path>
            </svg>
            <span className="brand-title">AcademicLab Edu</span>
          </div>
          <h1>Data Preparation Tutor</h1>
          <p>
            An educational LangGraph tutor to guide students through data
            preparation before analysis.
          </p>
        </div>
        <div className="hero-card">
          <div className="hero-card-header">
            <h2>Guided flow</h2>
            {sessionId && (
              <button
                className="outline-button danger small"
                type="button"
                onClick={() => setShowResetModal(true)}
                disabled={loading}
              >
                Start new session
              </button>
            )}
          </div>
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
                <span className="status-pill">
                  <StatusIcon />
                  {statusLabel}
                </span>
              </div>
              <div className="response-style">
                <button
                  className={responseStyle === "fast" ? "pill active" : "pill"}
                  type="button"
                  onClick={() => handleResponseStyle("fast")}
                >
                  Fast
                </button>
                <button
                  className={responseStyle === "detailed" ? "pill active" : "pill"}
                  type="button"
                  onClick={() => handleResponseStyle("detailed")}
                >
                  Detailed
                </button>
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
                <div
                  key={msg.id}
                  className={`bubble ${msg.role}${msg.action === "expert_help" ? " expert" : ""}`}
                >
                  <div className="bubble-header">
                    <span>
                      {msg.role === "user" ? <Icon name="user" /> : <Icon name="bot" />}
                      {msg.role === "user" ? "You" : msg.action === "expert_help" ? "Expert" : "Tutor"}
                    </span>
                    {msg.action && <span className="action-tag">{msg.action}</span>}
                  </div>
                  {renderMessage(msg.content)}
                </div>
              ))}
              {loading && (
                <div className="bubble assistant">
                  <div className="bubble-header">
                    <Icon name="bot" /> Tutor
                  </div>
                  <pre>Thinking...</pre>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {copyNotice && <div className="copy-toast">{copyNotice}</div>}

            <div className="chat-input">
              <div className="chat-input-field">
                <div className="input-label input-label-inline">
                  Activate the expert agent for technical questions.
                </div>
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a question or choose a P##..."
                  disabled={loading || status !== "connected"}
                  rows={2}
                />
                <button
                  className={expertMode ? "expert-switch active" : "expert-switch"}
                  type="button"
                  onClick={() => setExpertMode((prev) => !prev)}
                  disabled={loading || status !== "connected"}
                  aria-pressed={expertMode}
                  aria-label="Expert mode"
                  title="Expert mode"
                />
              </div>
              <button
                className="send-button"
                onClick={handleSend}
                disabled={loading || status !== "connected"}
                aria-label="Send message"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M2 12L22 3l-6 18-4.5-6L2 12z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                  <path d="M22 3L10 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <div className="chat-actions">
              <button
                className="outline-button"
                type="button"
                onClick={handleOpenNotes}
                disabled={loading}
              >
                Notes
              </button>
              <button
                className="reupload-button"
                type="button"
                onClick={handleOpenReupload}
              >
                Re-evaluate CSV
              </button>
              <button
                className="outline-button info"
                type="button"
                disabled={loading}
                onClick={handleOpenImageModal}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <rect x="3" y="5" width="18" height="14" rx="2" ry="2" fill="none" stroke="currentColor" strokeWidth="2" />
                  <circle cx="8" cy="9" r="2" fill="none" stroke="currentColor" strokeWidth="2" />
                  <path d="M21 17l-5.5-5.5L9 18l-2-2-4 4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                </svg>
                Analyse image
              </button>
            </div>

          </section>
        )}
      </main>

      {showReuploadModal && sessionId && (
        <div className="modal-backdrop" onClick={() => setShowReuploadModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3><Icon name="file" /> File to edit:</h3>
            <p className="path">{`dataset_v${csvInfo?.csv_version || "?"}.csv`}</p>
            <p><Icon name="edit" /> Fix the issue in Jupyter and bring the updated file here.</p>
            <p className="path-hint">
              Save the updated file in the same folder you selected for the first upload.
            </p>
            <div className="modal-actions">
              <label className="upload-button">
                <input type="file" accept=".csv" onChange={(e) => { handleCsvUpload(e); setShowReuploadModal(false); }} />
                I have the new file
              </label>
              <button className="ghost" onClick={() => setShowReuploadModal(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}


      {showResetModal && sessionId && (
        <div className="modal-backdrop" onClick={() => setShowResetModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3><Icon name="reset" /> Start a new session</h3>
            <p>This will permanently clear the current chat and session data.</p>
            <p>You will need to upload a new CSV to continue.</p>
            <div className="modal-actions">
              <button className="upload-button" onClick={handleResetSession} disabled={loading}>
                Reset session
              </button>
              <button className="ghost" onClick={() => setShowResetModal(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {showNotesModal && (
        <div className="modal-backdrop" onClick={() => setShowNotesModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3>Notes</h3>
            <p>Write your personal study notes here. They are saved locally in this browser.</p>
            <div className="notes-toolbar">
              <button className="outline-button" onClick={handleNewNote}>
                New note
              </button>
              <button className="outline-button" onClick={handleExportNote} disabled={!notesText.trim()}>
                Export .txt
              </button>
              <button className="outline-button" onClick={handleSaveNotes} disabled={!notesText.trim()}>
                Save
              </button>
            </div>
            <div className="notes-layout">
              <div className="notes-list">
                {notesIndex.length === 0 && <p className="muted">No notes yet.</p>}
                {notesIndex.map((note) => (
                  <button
                    key={note.id}
                    className={note.id === selectedNoteId ? "note-item active" : "note-item"}
                    onClick={() => handleSelectNote(note.id)}
                  >
                    {note.title || "Untitled"}
                  </button>
                ))}
              </div>
              <div className="notes-editor">
                <input
                  className="text-input"
                  placeholder="Title"
                  value={notesTitle}
                  onChange={(e) => setNotesTitle(e.target.value)}
                />
                <textarea
                  className="reason-input"
                  placeholder="Your notes..."
                  value={notesText}
                  onChange={(e) => setNotesText(e.target.value)}
                  rows={10}
                />
              </div>
            </div>
            <div className="modal-actions">
              <button className="ghost" onClick={handleClearNotes} disabled={!selectedNoteId}>
                Clear
              </button>
              <button className="ghost" onClick={() => setShowNotesModal(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {showImageModal && (
        <div className="modal-backdrop" onClick={() => setShowImageModal(false)}>
          <div
            className="modal-card image-modal"
            onClick={(e) => e.stopPropagation()}
            tabIndex={0}
            ref={imageModalRef}
          >
            <h3>Analyse image</h3>
            <p>Paste a chart from your clipboard (Ctrl+V / ⌘V).</p>
            <div className="image-drop">
              {imageDataUrl ? (
                <img src={imageDataUrl} alt="Pasted chart" />
              ) : (
                <span>Paste an image to preview it here.</span>
              )}
            </div>
            <div className="modal-actions">
              <button
                className="upload-button"
                onClick={handleAnalyzeImage}
                disabled={!imageDataUrl || imageLoading}
              >
                {imageLoading ? "Analyzing..." : "Analyze"}
              </button>
              <button className="ghost" onClick={() => setShowImageModal(false)}>
                Close
              </button>
            </div>
            {imageError && <p className="error-text">{imageError}</p>}
            {imageAnalysis && (
              <div className="analysis-box">
                <div className="analysis-header">
                  <span>Analysis</span>
                  <button className="ghost" onClick={handleCopyAnalysis}>
                    Copy
                  </button>
                </div>
                <pre>{imageAnalysis}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
