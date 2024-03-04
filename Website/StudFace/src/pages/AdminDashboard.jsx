import React, { useState, useEffect } from "react";
import profilepic from "../assets/images/emailavatar.png";
import Navbar from "../components/Navbar";
import { Link } from "react-router-dom";

const AdminDashboard = () => {
  const [userEmail, setUserEmail] = useState("");

  useEffect(() => {
    const fetchUserDetails = async () => {
      try {
        const response = await fetch("http://localhost:4000/api/v1/user-details", {
          method: "GET",
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        });

        const data = await response.json();

        if (response.ok) {
          const { email } = data;
          setUserEmail(email);
        } else {
          console.log("Error: " + response.status);
        }
      } catch (error) {
        console.log(error);
      }
    };

    fetchUserDetails();
  }, []);

  const handleTrainClick = async () => {
    try {
      const response = await fetch("http://192.168.72.6:5000/admin-training", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });
      const data = await response.json();

      if (response.ok) {
        // Handle success if needed
console.log(data);
alert(data.message);
      } else {
        console.log("Error: " + response.status);
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="min-h-screen bg-[#162544]">
      <Navbar user={"ADMIN"} image={profilepic} email={userEmail} />
      <div className="flex flex-col mt-[80px] font-bold">
        <Link to='/admindashboard/addteachers' className="mx-auto">
          <button className="text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg">
            ADD TEACHERS
          </button>
        </Link>
        <Link to='/admindashboard/editteachers' className="mx-auto">
          <button className="text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg">
            EDIT TEACHERS
          </button>
        </Link>
        <Link to='/admindashboard/addstudents' className="mx-auto">
          <button className="text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg">
            ADD STUDENTS
          </button>
        </Link>
        <Link className="mx-auto">
          <button className="text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg">
            EDIT STUDENTS
          </button>
        </Link>
        <Link to='/admindashboard/addadmin' className="mx-auto">
          <button className="text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg">
            ADD ADMIN
          </button>
        </Link>
        <Link className="mx-auto">
          <button className="text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg">
            EDIT ADMIN
          </button>
        </Link>
        <button
          className="fixed bottom-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-lg"
          onClick={handleTrainClick}
        >
          Train
        </button>
      </div>
    </div>
  );
};

export default AdminDashboard;
