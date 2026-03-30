import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
} from "recharts";

function SHAPChart({ data }) {
  const chartData = Object.entries(data)
    .map(([word, value]) => ({ word, value }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="card">
      <h2>SHAP Explanation</h2>
      <p style={{ fontSize: "0.82rem", color: "#94a3b8", marginBottom: 12 }}>
        Positive values (red) push toward <em>Stressed</em>; negative (blue) push
        toward <em>Not Stressed</em>.
      </p>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
            <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 12 }} />
            <YAxis
              dataKey="word"
              type="category"
              tick={{ fill: "#e2e8f0", fontSize: 13 }}
              width={80}
            />
            <Tooltip
              contentStyle={{
                background: "#1e1b4b",
                border: "1px solid rgba(255,255,255,0.15)",
                borderRadius: 8,
                color: "#e2e8f0",
              }}
              formatter={(v) => [v.toFixed(4), "SHAP value"]}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, idx) => (
                <Cell
                  key={idx}
                  fill={entry.value >= 0 ? "#ef4444" : "#3b82f6"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default SHAPChart;
