import { Routes, Route } from 'react-router-dom';
import Login from './pages/Login';
import AdminLogin from './pages/AdminLogin';
import TeacherLogin from './pages/TeachersLogin';
import AdminDashboard from './pages/AdminDashboard';
import TeacherDashboard from './pages/TeacherDashboard';
import AddAdmin from './pages/AddAdmin';
import AddTeachers from './pages/AddTeachers';
import AddStudents from './pages/AddStudents';
import EditTeachers from './pages/EditTeachers';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/adminlogin" element={<AdminLogin />} />
      {/* <Route path="/adminsignup" element={<AdminSignup />} /> */}
      <Route path="/teacherlogin" element={<TeacherLogin />} />
      {/* <Route path="/teachersignup" element={<TeacherSignup />} /> */}
      <Route path="/admindashboard" element={<AdminDashboard />} />
      <Route path="/admindashboard/addadmin" element={<AddAdmin />} />
      <Route
        path="/admindashboard/addstudents"
        element={<AddStudents />}
      />
      <Route
        path="/admindashboard/addteachers"
        element={<AddTeachers />}
      />
      <Route
        path="/admindashboard/editteachers"
        element={<EditTeachers />}
      />
      <Route
        path="/teacherlogin/teacherdashboard"
        element={<TeacherDashboard />}
      />
    </Routes>
  );
}

export default App;
