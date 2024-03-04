import React, { useState } from "react";  //used formdata and setformdata
import { Link, useNavigate } from "react-router-dom";
import LoginBoy from "../assets/images/LoginPageboy.png";
import WelcomeLamp from "../assets/images/welcomelamp.png";
import welcomeLogo from "../assets/images/welcomelogo.png";

const Login = () => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch("http://localhost:4000/api/v1/adminlogin", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (response.ok) {
        const { success, token } = data;

        if (success) {
          localStorage.setItem("token", token);

          navigate("/admindashboard");
        } else {
          alert("Invalid credentials. Please try again.");
        }
      } else {
        alert("Enter valid credentials.");
      }
    } catch (error) {
      console.log(error);
      alert("Enter valid credentials.");
    }
  };

  return (
    <div className="min-h-screen bg-[#162544] flex flex-row">
      <div className="flex flex-col relative w-[50%] justify-center">
        <div className="flex justify-center">
          <img src={welcomeLogo} alt="welcome logo" className="w-[190px]" />
        </div>
        <div className="flex font-bold text-2xl justify-center py-6">
          <h2 className="text-white">Admin Login</h2>
        </div>
        <div className="flex font-bold text-2xl justify-center">
          <form className="flex flex-col" onSubmit={handleLogin}>
            <label htmlFor="adminusername" className="font-semibold text-white">
              Email
            </label>
            <input
              className="bg-transparent border-b-4 border-[#ffc42a]"
              name="adminusername"
              id="adminusername"
              type="text"
              value={formData.email}
              onChange={(e) =>
                setFormData({ ...formData, email: e.target.value })
              }
            />
            <label
              htmlFor="adminpassword"
              className="text-white font-semibold pt-4"
            >
              Password
            </label>
            <input
              className="bg-transparent border-b-4 border-[#ffc42a]"
              name="adminpassword"
              id="adminpassword"
              type="password"
              value={formData.password}
              onChange={(e) =>
                setFormData({ ...formData, password: e.target.value })
              }
            />
            <span className="text-[15px] flex justify-end text-white font-light">
              forgot password?
            </span>
            <button
              type="submit"
              className="my-4 bg-[#ffc42a] flex m-auto w-44 justify-center rounded-md p-2 text-[#00523F] font-bold"
            >
              Login
            </button>
          </form>
        </div>
      </div>
      <div className="relative flex flex-col items-center justify-center w-[50%]">
        <div>
          <img
            src={WelcomeLamp}
            alt="lamp"
            className="w-36 absolute top-0 left-[10rem]"
          />
        </div>
        <div>
          <img src={LoginBoy} alt="boy" className="w-56" />
        </div>
      </div>
    </div>
  );
};

export default Login;
