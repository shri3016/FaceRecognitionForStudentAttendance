const Student = require("../models/addStudentModel");

const addStudent = async (req, res) => {
  try {
    const {
      firstName,
      lastName,
      email,
      rollNumber,
      department,
      year,
      dateOfBirth,
      phoneNumber,
    } = req.body;

    const existingStudent = await Student.findOne({ rollNumber });
    if (existingStudent) {
      return res.status(409).json({ message: "Roll number already exists" });
    }

    const newStudent = new Student({
      firstName,
      lastName,
      email,
      rollNumber,
      department,
      year,
      dateOfBirth,
      phoneNumber,
    });

    await newStudent.save();

    res.status(201).json({ message: "Student added successfully" });
  } catch (err) {
    res.status(500).json({ message: "Error adding student", error: err });
  }
};

const getAllStudents = async (req, res) => {
  try {
    const students = await Student.find();
    res.send(students);
  } catch (error) {
    res.status(500).send(error);
  }
};

const updateStudent = async (req, res) => {
  try {
    const { id } = req.params;
    const student = await Student.findByIdAndUpdate(id, req.body, {
      new: true,
    });
    if (!student) {
      return res.status(404).send();
    }
    res.send(student);
  } catch (error) {
    res.status(400).send(error);
  }
};

const deleteStudent = async (req, res) => {
  try {
    const { id } = req.params;
    const student = await Student.findByIdAndDelete(id);
    if (!student) {
      return res.status(404).send();
    }
    res.send(student);
  } catch (error) {
    res.status(500).send(error);
  }
};

module.exports = {
  addStudent,
  getAllStudents,
  updateStudent,
  deleteStudent,
};
