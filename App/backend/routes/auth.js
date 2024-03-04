const express = require("express");
const router = express.Router();

const { login } = require("../controllers/Auth");
const { getUserData } = require("../controllers/getUserData");
const authMiddleware = require("../middlewares/auth");

router.post("/login", login);
router.get("/user", authMiddleware, getUserData);

module.exports = router;
