const express = require("express");
const app = express();

require("dotenv").config();
const PORT = process.env.PORT || 4000;

app.use(express.json());

const studface = require("./routes/auth");

app.get("/", (req, res) => {
  res.send("<h1>Welcome to StudFace</h1>");
});

app.use("/api/v1", studface);

app.use("/profile-picture", express.static("C:\Users\91820\OneDrive\Desktop\FYProject\App\studFace\assets\images\profilepic"));

const connectWithDb = require("./config/database");
connectWithDb();

app.listen(PORT, () => {
  console.log(`App is started at PORT number ${PORT}`);
});
