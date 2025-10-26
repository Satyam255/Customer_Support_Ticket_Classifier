import React, { useState } from "react";
import "./App.css";

const Spinner = () => (
  <svg
    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
  >
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
    />
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    ></path>
  </svg>
);

const ResultBadge = ({ label }) => {
  const baseStyle =
    "px-5 py-2 rounded-full text-sm font-semibold shadow-md uppercase tracking-wide";
  const colorMap = {
    "Billing Question": "bg-blue-100 text-blue-800",
    "Technical Issue": "bg-red-100 text-red-800",
    "General Inquiry": "bg-green-100 text-green-800",
  };
  return (
    <div
      className={`${baseStyle} ${
        colorMap[label] || "bg-gray-100 text-gray-800"
      }`}
    >
      {label}
    </div>
  );
};

function App() {
  const [text, setText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:3000/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error("Server error");
      const data = await response.json();
      setResult(data.prediction);
    } catch (err) {
      console.error(err);
      setError(
        "Failed to classify. Check if Flask & Node servers are running."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-purple-50 to-blue-50 flex items-center justify-center font-sans p-4">
      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-2xl p-10 md:p-16">
        <header className="text-center mb-10">
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-blue-500 mb-2">
            Customer Support Ticket Classifier
          </h1>
          <p className="text-lg md:text-xl text-gray-600">
            Paste your support ticket below and let AI classify it instantly.
          </p>
        </header>

        <form onSubmit={handleSubmit}>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full h-44 p-5 border border-gray-300 rounded-xl shadow-md focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all resize-none mb-6 text-gray-700"
            placeholder="e.g., 'My credit card was charged twice for the same transaction...'"
            required
          />
          <div className="flex justify-center">
            <button
              type="submit"
              disabled={isLoading || !text}
              className="flex items-center justify-center px-10 py-3 bg-gradient-to-r from-purple-600 to-blue-500 text-white font-semibold rounded-xl shadow-lg hover:scale-105 transform transition-all disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <Spinner /> Classifying...
                </>
              ) : (
                "Classify Ticket"
              )}
            </button>
          </div>
        </form>

        <div className="mt-12 min-h-[120px] flex flex-col items-center justify-center space-y-4">
          {error && (
            <div className="text-center">
              <p className="text-red-600 font-semibold text-lg">Error</p>
              <p className="text-gray-600">{error}</p>
            </div>
          )}

          {result && (
            <div className="text-center space-y-3">
              <p className="text-lg md:text-xl font-medium text-gray-700">
                Predicted Category:
              </p>
              <ResultBadge label={result.label} />
              <p className="text-gray-500 text-sm md:text-base">
                Confidence: {(result.score * 100).toFixed(2)}%
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
