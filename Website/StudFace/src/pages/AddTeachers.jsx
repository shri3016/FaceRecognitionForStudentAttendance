import React, { useState, useEffect } from "react";
import Navbar from "../components/Navbar";
import Select from "react-select";
import profilepic from "../assets/images/emailavatar.png";

const AddTeachers = () => {
  const [userEmail, setUserEmail] = useState("");

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
    department: "",
    subjects: [],
    gender: "",
    dateOfBirth: "",
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:4000/api/v1/signup", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("User created successfully:", data);
        alert("User created successfully");
        setFormData({
          firstName: "",
          lastName: "",
          email: "",
          password: "",
          phoneNumber: "",
          department: "",
          subjects: [],
          gender: "",
          dateOfBirth: "",
        });
      } else {
        console.error("User registration failed");
        alert("User registration failed");
      }
    } catch (error) {
      console.error("Error occurred while submitting form", error);
    }
  };

  const subjectsOptions = [
    { value: "Compiler Design", label: "Compiler Design" },
    { value: "Computer Networks", label: "Computer Networks" },
    // Add more subjects as needed
  ];

  const handleSubjectsChange = (selectedOptions) => {
    const selectedSubjects = selectedOptions
      ? selectedOptions.map((option) => option.value)
      : [];
    setFormData((prevState) => ({
      ...prevState,
      subjects: selectedSubjects,
    }));
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

          <label className="text-white" htmlFor="lastName">Last Name</label>
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
          <label className="text-white" htmlFor="password">Password</label>
          <input
            placeholder="Enter Password"
            type="password"
            name="password"
            id="password"
            value={formData.password}
            onChange={handleInputChange}
            required
          />
          <label className="text-white" htmlFor="phoneNumber">Phone No</label>
          <input
            placeholder="Enter Phone"
            type="tel"
            name="phoneNumber"
            id="phoneNumber"
            value={formData.phoneNumber}
            onChange={handleInputChange}
            pattern="[0-9]{10}"
            required
          />
          <label className="text-white" htmlFor="department">Department</label>
          <div>
            <input
              type="radio"
              id="comp"
              name="department"
              value="Comp"
              checked={formData.department === "Comp"}
              onChange={handleInputChange}
              required
            />
            <label  htmlFor="comp" className="m-1 text-white">
              Comp
            </label>
          </div>
          <div>
            <input
              type="radio"
              id="ecomp"
              name="department"
              value="EComp"
              checked={formData.department === "EComp"}
              onChange={handleInputChange}
              required
            />
            <label htmlFor="ecomp" className="m-1 text-white">
              EComp
            </label>
          </div>
          <div>
            <input
              type="radio"
              id="it"
              name="department"
              value="IT"
              checked={formData.department === "IT"}
              onChange={handleInputChange}
              required
            />
            <label htmlFor="it" className="m-1 text-white">
              IT
            </label>
          </div>
          <div>
            <input
              type="radio"
              id="mech"
              name="department"
              value="Mech"
              checked={formData.department === "Mech"}
              onChange={handleInputChange}
              required
            />
            <label htmlFor="mech" className="m-1 text-white">
              Mech
            </label>
          </div>
          <fieldset>
            <legend className="text-white">Select the subjects</legend>
            <Select
              isMulti
              options={subjectsOptions}
              value={subjectsOptions.filter((option) =>   //filter out selevted and store in the vqlue
                formData.subjects.includes(option.value)
              )}
              onChange={handleSubjectsChange}
              required
            />
          </fieldset>
          <fieldset className="flex flex-row">
            <legend className="text-white">Gender</legend>
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
              <label htmlFor="male" className="m-1 text-white">
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
              <label htmlFor="female" className="m-1 text-white">
                Female
              </label>
            </div>
          </fieldset>
          <label className="text-white" htmlFor="dateOfBirth">Date of Birth</label>
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

export default AddTeachers;
