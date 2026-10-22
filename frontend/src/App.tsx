import { useState, useEffect } from "react";
import { apiClient, HealthResponse } from "./api/client";

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.getHealth();
      setHealth(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="container">
      <header style={{ marginBottom: "2rem" }}>
        <h1>Kronos Tensor Lab</h1>
        <p style={{ color: "#6b7280" }}>A Foundation Model for the Language of Financial Markets</p>
      </header>

      <div className="card">
        <h2>System Health</h2>
        {loading && <p>Checking backend connection...</p>}
        {error && (
          <p className="status-unhealthy">Error: {error}</p>
        )}
        {health && (
          <div>
            <p>
              Status: <span className={health.status === "healthy" ? "status-healthy" : "status-unhealthy"}>
                {health.status}
              </span>
            </p>
            {health.service && <p>Service: {health.service}</p>}
            {health.version && <p>Version: {health.version}</p>}
          </div>
        )}
        <button onClick={fetchHealth} disabled={loading}>
          {loading ? "Checking..." : "Refresh"}
        </button>
      </div>

      <div className="card">
        <h2>Available Workflows</h2>
        <ul style={{ lineHeight: 2 }}>
          <li><strong>Tokenizer:</strong> Encode/decode OHLCV windows (planned)</li>
          <li><strong>Prediction:</strong> Kronos model inference with sampling (planned)</li>
          <li><strong>Fine-tuning:</strong> LoRA adapter training on custom data (planned)</li>
          <li><strong>Validation:</strong> Backtest metrics and visualization (planned)</li>
        </ul>
      </div>

      <footer style={{ marginTop: "2rem", color: "#9ca3af", fontSize: "0.875rem" }}>
        <p>Alan Vo | AI & Machine Learning</p>
        <p>alvo@gmail.com</p>
      </footer>
    </div>
  );
}

export default App;