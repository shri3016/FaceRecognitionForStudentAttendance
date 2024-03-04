const mongoose = require("mongoose");

const addStudentSchema = new mongoose.Schema({
  firstName: {
    type: String,
    required: true,
  },
  lastName: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
    match: /^([\w-\.]+@([\w-]+\.)+[\w-]{2,4})?$/,
  },
  rollNumber: {
    type: String,
    required: true,
    unique: true, // Make the rollNumber field unique
  },
  department: {
    type: String,
    required: true,
  },
  year: {
    type: String,
    required: true,
  },
  dateOfBirth: {
    type: Date,
    required: true,
  },
  phoneNumber: {
    type: String,
    match: /^[0-9]{10}$/,
  },
});

module.exports = mongoose.model("AddStudent", addStudentSchema);
