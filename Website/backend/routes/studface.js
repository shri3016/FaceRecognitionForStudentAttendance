const express = require("express");
const router = express.Router();
const { getAdminDetails } = require("../controllers/GetDetailsAdmin");
const { login, signup,getAllTeachers } = require("../controllers/Auth");
const { auth, isAdmin, isTeachers } = require("../middlewares/auth");
const { adminLogin, adminSignup } = require("../controllers/AdminAuth");
const teacherController = require("../controllers/TeachersController");
const studentController = require("../controllers/StudentController");

// Get admin details
router.get("/user-details", getAdminDetails);

// Teacher routes
router.post("/login", login);
router.post("/signup", signup);

// Admin routes
router.post("/adminlogin", adminLogin);
router.post("/adminsignup", adminSignup);

router.get("/getallteachers",getAllTeachers);





// Protected teacher route
router.get("/teacherauth", auth, isTeachers, (req, res) => {
  res.json({
    success: true,
    message: "Welcome to the protected teachers route",
  });
});

// Protected admin route
router.get("/adminauth", auth, isAdmin, (req, res) => {
  res.json({
    success: true,
    message: "Welcome to the protected admin routes",
  });
});



// Teacher routes
router.post("/addteachers", teacherController.addTeacher);
router.get("/teachers", teacherController.getAllTeachers);
router.patch("/teacher/:id", teacherController.updateTeacher);
router.delete("/teacher/:id", teacherController.deleteTeacher);

// Student routes
router.post("/addstudent", studentController.addStudent);
router.get("/students", studentController.getAllStudents);
router.patch("/student/:id", studentController.updateStudent);
router.delete("/student/:id", studentController.deleteStudent);

module.exports = router;
