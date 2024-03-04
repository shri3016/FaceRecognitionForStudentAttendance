import React, { useState, useEffect } from "react";
import Navbar from "../components/Navbar";
import profilepic from "../assets/images/emailavatar.png";
import Select from "react-select";

const AddStudents = () => {
  const [userEmail, setUserEmail] = useState("");
  const [studentData, setStudentData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    rollNumber: "",
    department: "",
    dateOfBirth: "",
    phoneNumber: "",
    year: "",
  });

  useEffect(() => {
    const fetchUserDetails = async () => {
      try {
        const response = await fetch(
          "http://localhost:4000/api/v1/user-details",
          {
            method: "GET",
            headers: {
              Authorization: `Bearer ${localStorage.getItem("token")}`,
            },
          }
        );

        if (response.ok) {
          const data = await response.json();
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

  const handleInputChange = (e) => {
    const { name, value } = e.target;

    if (name === "department") {
      setStudentData((prevData) => ({
        ...prevData,
        [name]: value,
      }));
    } else {
      setStudentData((prevData) => ({
        ...prevData,
        [name]: value,
        department: "", // Clear department when other fields are changed
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log(studentData);
    const formData = new FormData();
    formData.append("name", studentData.firstName);
    try {
      const response = await fetch("http://localhost:4000/api/v1/addstudent", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(studentData),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("User created successfully:", data);
        alert("User created successfully");
        setStudentData({
          firstName: "",
          lastName: "",
          email: "",
          rollNumber: "",
          department: "",
          dateOfBirth: "",
          phoneNumber: "",
          year: "",
        });
        console.log(studentData.firstName + studentData.lastName);
        // Trigger the first API to capture images
        const captureImagesResponse = await fetch(
          "http://192.168.72.6:5000/admin-train-images",
          {
            method: "POST",

            headers: {
              // Don't set Content-Type, it will be automatically set by FormData
              Authorization: `Bearer ${localStorage.getItem("token")}`,
            },
            body: formData,
          }
        );
        const captureImagesData = await captureImagesResponse.json();
        alert(captureImagesData.message);
        console.log("Capture images response:", captureImagesData.message);

        // Trigger the second API to capture test images
        const captureTestImagesResponse = await fetch(
          "http://192.168.72.6:5000/admin-test-images",
          {
            method: "POST",
            headers: {
              // Don't set Content-Type, it will be automatically set by FormData
              Authorization: `Bearer ${localStorage.getItem("token")}`,
            },
            body: formData,
          }
        );

        const captureTestImagesData = await captureTestImagesResponse.json();
        alert(captureTestImagesData.message);
        console.log(
          "Capture test images response:",
          captureTestImagesData.message
        );
      } else {
        console.error("User registration failed");
        alert("User registration failed");
      }
    } catch (error) {
      console.log(error);
      console.error("Error occurred while submitting form", error);
    }
  };

  const departmentOptions = [
    { value: "Comp", label: "Comp" },
    { value: "EComp", label: "EComp" },
    { value: "IT", label: "IT" },
    { value: "Mech", label: "Mech" },
  ];

  const yearOptions = [
    { value: "1st", label: "1st" },
    { value: "2nd", label: "2nd" },
    { value: "3rd", label: "3rd" },
    { value: "4th", label: "4th" },
  ];

  return (
    <div className="min-h-screen bg-[#162544]">
      <Navbar user={"ADMIN"} image={profilepic} email={userEmail} />
      <div className="flex flex-cols justify-center m-5 space-y-5">
        <form className="flex flex-col" onSubmit={handleSubmit}>
          <label htmlFor="firstName">First Name</label>
          <input
            type="text"
            id="firstName"
            name="firstName"
            placeholder="Enter first name"
            value={studentData.firstName}
            onChange={handleInputChange}
          />
          <label htmlFor="lastName">Last Name</label>
          <input
            type="text"
            id="lastName"
            name="lastName"
            placeholder="Enter last name"
            value={studentData.lastName}
            onChange={handleInputChange}
          />
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            name="email"
            placeholder="Enter email"
            value={studentData.email}
            onChange={handleInputChange}
          />
          <label htmlFor="rollNumber">Roll Number</label>
          <input
            type="text"
            id="rollNumber"
            name="rollNumber"
            placeholder="Enter roll number"
            value={studentData.rollNumber}
            onChange={handleInputChange}
          />
          <label htmlFor="department">Department</label>
          <div className="flex space-x-2">
            {departmentOptions.map((option) => (
              <div key={option.value}>
                <input
                  type="radio"
                  id={option.value}
                  name="department"
                  value={option.value}
                  checked={studentData.department === option.value}
                  onChange={handleInputChange}
                />
                <label htmlFor={option.value}>{option.label}</label>
              </div>
            ))}
          </div>

          <label htmlFor="year">Year</label>
          <div className="flex space-x-2">
            {yearOptions.map((option) => (
              <div key={option.value}>
                <input
                  type="radio"
                  id={option.value}
                  name="year"
                  value={option.value}
                  checked={studentData.year === option.value}
                  onChange={handleInputChange}
                />
                <label htmlFor={option.value}>{option.label}</label>
              </div>
            ))}
          </div>
          <label htmlFor="dateOfBirth">Date of Birth</label>
          <input
            type="date"
            id="dateOfBirth"
            name="dateOfBirth"
            placeholder="Enter date of birth"
            value={studentData.dateOfBirth}
            onChange={handleInputChange}
          />
          <label htmlFor="phoneNumber">Phone Number</label>
          <input
            type="text"
            id="phoneNumber"
            name="phoneNumber"
            placeholder="Enter phone number"
            value={studentData.phoneNumber}
            onChange={handleInputChange}
          />
          
          <button
            type="submit"
            className="bg-blue-500 text-white font-bold py-2 mt-2 px-4 rounded"
          >
            Add Student
          </button>
        </form>
      </div>
    </div>
  );
};

export default AddStudents;
