const User = require("../models/User");

exports.getUserData = async (req, res) => {
  try {
    const userId = req.user.id;

    const user = await User.findById(userId);

    if (!user) {
      return res.status(404).json({
        success: false,
        message: "User not found",
      });
    }

    const {
      firstName,
      lastName,
      email,
      gender,
      dateOfBirth,
      phoneNumber,
      profilePicture,
      department,
      subjects,
    } = user;

    return res.status(200).json({
      success: true,
      firstName,
      lastName,
      email,
      gender,
      dateOfBirth,
      phoneNumber,
      profilePicture,
      department,
      subjects,
    });
  } catch (error) {
    console.log(error);
    return res.status(500).json({
      success: false,
      message: "Failed to fetch user data",
    });
  }
};
