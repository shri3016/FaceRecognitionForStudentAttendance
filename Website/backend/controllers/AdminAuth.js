const bcrypt = require("bcrypt");
const Admin = require("../models/addAdminModel");
const jwt = require("jsonwebtoken");
require("dotenv").config();

exports.adminSignup = async (req, res) => {
  try {
    const { firstName, lastName, email, password, phoneNumber, gender, dateOfBirth } = req.body;
    const existingAdmin = await Admin.findOne({ email });

    if (existingAdmin) {
      return res.status(400).json({
        success: false,
        message: "Admin already exists",
      });
    }

    let hashedPassword;
    try {
      hashedPassword = await bcrypt.hash(password, 10);
    } catch (err) {
      return res.status(500).json({
        success: false,
        message: "Error in hashing password",
      });
    }

    const admin = await Admin.create({
      firstName,
      lastName,
      email,
      password: hashedPassword,
      phoneNumber,
      gender,
      dateOfBirth
    });

    return res.status(200).json({
      success: true,
      message: "Admin created successfully",
      admin: {
        _id: admin._id,
        firstName: admin.firstName,
        lastName: admin.lastName,
        email: admin.email,
        phoneNumber: admin.phoneNumber,
        gender: admin.gender,
        dateOfBirth: admin.dateOfBirth,
      },
    });
  } catch (error) {
    console.error("Error occurred while submitting form", error);
    return res.status(500).json({
      success: false,
      message: "Admin registration failed, please try again later",
    });
  }
};


exports.adminLogin = async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: "Please fill in all the details carefully",
      });
    }

    let admin = await Admin.findOne({ email });
    if (!admin) {
      return res.status(401).json({
        success: false,
        message: "Admin not registered",
      });
    }

    const payload = {
      email: admin.email,
      id: admin._id,
      role: "admin",
    };

    if (await bcrypt.compare(password, admin.password)) {
      let token = jwt.sign(payload, process.env.JWT_SECRET, {
        expiresIn: "2h",
      });

      admin = admin.toObject();
      admin.token = token;
      admin.password = undefined;

      res.status(200).json({
        success: true,
        token,
        admin,
        message: "Admin logged in successfully",
      });
    } else {
      return res.status(403).json({
        success: false,
        message: "Password incorrect",
      });
    }
  } catch (error) {
    console.log(error);
    return res.status(500).json({
      success: false,
      message: "Admin login failed",
    });
  }
};
