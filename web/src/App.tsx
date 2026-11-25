import { useEffect, useState } from "react";
import { Chat } from "./components/Chat";
import { Dashboard } from "./components/Dashboard";

type Tab = "chat" | "eval";
type Ready = { ok: boolean; detail: string } | null;

export function App() {
  const [tab, setTab] = useState<Tab>(location.hash === "#eval" ? "eval" : "chat");
  const [ready, setReady] = useState<Ready>(null);

  useEffect(() => {
    location.hash = tab;
  }, [tab]);

  useEffect(() => {
    const onHash = () => setTab(location.hash === "#eval" ? "eval" : "chat");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    fetch("/readyz")
      .then(async (r) => {
        const body = await r.json();
        setReady({ ok: r.ok, detail: r.ok ? body.tools.join(", ") : body.reason });
      })
      .catch((e) => setReady({ ok: false, detail: String(e) }));
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <strong className="brand">Triage Agent</strong>
        <nav className="tabs">
          <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>
            Chat
          </button>
          <button className={tab === "eval" ? "active" : ""} onClick={() => setTab("eval")}>
            Eval results
          </button>
        </nav>
        {ready && (
          <span className={`badge ${ready.ok ? "good" : "bad"}`} title={ready.detail}>
            {ready.ok ? "✓ agent ready" : "✕ agent not ready"}
          </span>
        )}
      </header>
      <main>{tab === "chat" ? <Chat /> : <Dashboard />}</main>
    </div>
  );
}
