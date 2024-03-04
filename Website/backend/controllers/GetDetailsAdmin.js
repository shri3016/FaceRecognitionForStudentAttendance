const jwt = require("jsonwebtoken");
const Admin = require("../models/addAdminModel");

const getAdminDetails = async (req, res) => {
  try {
    // Extract the token from the request headers
    const token = req.headers.authorization.split(" ")[1];

    // Verify the token and extract the user ID from the payload
    const { id: userId } = jwt.verify(token, process.env.JWT_SECRET);

    // Fetch the user details based on the user ID
    const user = await Admin.findById(userId);

    if (user) {
      // Return the user details
      res.json({ email: user.email });
    } else {
      res.status(404).json({ message: "User not found" });
    }
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: "Internal server error" });
  }
};

module.exports = {
  getAdminDetails,
};
