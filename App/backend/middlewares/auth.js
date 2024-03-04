const jwt = require("jsonwebtoken");
require("dotenv").config();

const authMiddleware = (req, res, next) => {
  try {
    const token = req.headers.authorization.split(" ")[1];
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: "Authorization token not found",
      });
    }
    
    const decodedToken = jwt.verify(token, process.env.JWT_SECRET);
    
    req.user = decodedToken;
    
    next();
  } catch (error) {
    console.log(error);
    return res.status(401).json({
      success: false,
      message: "Invalid or expired token",
    });
  }
};

module.exports = authMiddleware;
