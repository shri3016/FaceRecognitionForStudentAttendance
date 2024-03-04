import React, { useState, useEffect } from "react";
import Navbar from "../components/Navbar";
import profilepic from "../assets/images/emailavatar.png";

const AddAdmins = () => {
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

  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    phoneNumber: "",
    gender: "",
    dateOfBirth: "",
  });

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch("http://localhost:4000/api/v1/adminsignup", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        console.log("Admin created successfully");
        alert("Admin created successfully");
        setFormData({
          firstName: "",
          lastName: "",
          email: "",
          password: "",
          phoneNumber: "",
          gender: "",
          dateOfBirth: "",
        });
      } else {
        console.error("Admin registration failed");
        alert("Admin registration failed");
      }
    } catch (error) {
      console.error("Error occurred while submitting form", error);
    }
  };

  return (
    <div className="min-h-screen bg-[#162544]">
      <Navbar user={"ADMIN"} image={profilepic} email={userEmail} />
      <div className="flex flex-cols justify-center m-5 space-y-5">
        <form className="flex flex-col" onSubmit={handleSubmit}>
          <label className="text-white" htmlFor="firstName">
            First Name
          </label>
          <input
            placeholder="Enter First Name"
            type="text"
            name="firstName"
            id="firstName"
            value={formData.firstName}
            onChange={handleInputChange}
            required
          />
          <label htmlFor="lastName">Last Name</label>
          <input
            placeholder="Enter Last Name"
            type="text"
            name="lastName"
            id="lastName"
            value={formData.lastName}
            onChange={handleInputChange}
            required
          />
          <label htmlFor="email">Email</label>
          <input
            placeholder="Enter Email"
            type="email"
            name="email"
            id="email"
            value={formData.email}
            onChange={handleInputChange}
            required
          />
          <label htmlFor="password">Password</label>
          <input
            placeholder="Enter Password"
            type="password"
            name="password"
            id="password"
            value={formData.password}
            onChange={handleInputChange}
            required
          />
          <label htmlFor="phoneNumber">Phone No</label>
          <input
            placeholder="Enter Phone"
            type="tel"
            name="phoneNumber"
            id="phoneNumber"
            value={formData.phoneNumber}
            onChange={handleInputChange}
            pattern="[0-9]{10}" //to keep phone number to 10
            required
          />
          <fieldset className="flex flex-row">
            <legend>Gender</legend>
            <div>
              <input
                type="radio"
                id="male"
                name="gender"
                value="male"
                checked={formData.gender === "male"}
                onChange={handleInputChange}
                required
              />
              <label htmlFor="male" className="m-1">
                Male
              </label>
            </div>
            <div>
              <input
                type="radio"
                id="female"
                name="gender"
                value="female"
                checked={formData.gender === "female"}
                onChange={handleInputChange}
                required
              />
              <label htmlFor="female" className="m-1">
                Female
              </label>
            </div>
          </fieldset>
          <label htmlFor="dateOfBirth">Date of Birth</label>
          <input
            type="date"
            name="dateOfBirth"
            id="dateOfBirth"
            value={formData.dateOfBirth}
            onChange={handleInputChange}
            required
          />
          <button
            type="submit"
            className="text-[#162544] m-2 bg-[#ffc42a] w-24 p-2 mx-auto rounded-lg"
          >
            Submit
          </button>
        </form>
      </div>
    </div>
  );
};

export default AddAdmins;
