const express = require("express");
const cors = require("cors");

const app = express();

require("dotenv").config();
const PORT = process.env.PORT || 4000;

app.use(cors());

app.use(express.json());

const studface = require("./routes/studface");
app.use("/api/v1", studface);

const connectWithDb = require("./config/database");
connectWithDb();

app.listen(PORT, () => {
  console.log(`App is started at PORT number ${PORT}`);
});

app.get("/", (req, res) => {
  res.send(`<h1>Welcome to StudFace</h1>`);
});
