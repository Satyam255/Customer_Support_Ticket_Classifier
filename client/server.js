import express from "express";
import axios from "axios";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

// 🔹 Simple GET route to test connection
app.get("/api/flask", async (req, res) => {
  try {
    const response = await axios.get("http://127.0.0.1:5000/");
    res.send(`Flask Response: ${response.data.message}`);
  } catch (error) {
    console.error("Error contacting Flask:", error.message);
    res.status(500).send("Error contacting Flask");
  }
});

// 🔹 Route to send text to Flask classify API
app.post("/api/classify", async (req, res) => {
  try {
    const flaskRes = await axios.post(
      "http://127.0.0.1:5000/classify",
      req.body
    );
    res.json(flaskRes.data);
  } catch (error) {
    console.error("Error contacting Flask:", error.message);
    res.status(500).json({ error: "Error contacting Flask" });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`🚀 Node.js server running at http://localhost:${PORT}`);
});
