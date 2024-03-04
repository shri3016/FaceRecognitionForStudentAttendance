const mongoose = require("mongoose");

const addTeachersSchema = new mongoose.Schema({
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
  password: {
    type: String,
    required: true,
  },
  phoneNumber: {
    type: String,
    match: /^[0-9]{10}$/,
  },
  profilePicture: {
    type: String,
  },
  department: {
    type: String,
    enum: ["Comp", "EComp", "IT", "Mech"], // Enum field with the fixed departments
    required: true,
  },
  subjects: {
    type: [String],
    default: [],
  },
  gender: {
    type: String,
    enum: ["male", "female", "others"],
    required: true,
  },
  dateOfBirth: {
    type: Date,
    required: true,
  },
});

module.exports = mongoose.model("AddTeachers", addTeachersSchema);
