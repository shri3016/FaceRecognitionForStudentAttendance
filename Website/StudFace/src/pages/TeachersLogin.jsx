import React from "react";
import { Link } from "react-router-dom";
import LoginBoy from "../assets/images/LoginPageboy.png";
import WelcomeLamp from "../assets/images/welcomelamp.png";
import welcomeLogo from "../assets/images/welcomelogo.png";

const Login = () => {
  return (
    <div className="min-h-screen bg-[#162544] flex flex-row">
      <div className="flex flex-col  relative w-[50%] justify-center">
        <div className="flex justify-center">
          <img src={welcomeLogo} alt="welcome logo" className="w-[190px]" />
        </div>
        <div className="flex font-bold text-2xl justify-center py-6">
          <h2 className="text-white">Teacher Login</h2>
        </div>
        <div className="flex font-bold text-2xl justify-center">
          <form className="flex flex-col">
            <label htmlFor="teacherusername" className="font-semibold text-white">
              Username
            </label>
            <input
              className="bg-transparent border-b-4 border-[#ffc42a]"
              name="teacherusername"
              id="teacherusername"
              type="text"
            />
            <label
              htmlFor="teacherpassword"
              className="text-white font-semibold pt-4"
            >
              Password
            </label>
            <input
              className="bg-transparent border-b-4 border-[#ffc42a]"
              name="teacherpassword"
              id="teacherpassword"
              type="password"
            />
            <span className="text-[15px] flex justify-end text-white font-light">
              forgot password?
            </span>
          </form>
        </div>
        <div className="flex flex-col w-[100%] justify-end">
          <Link to="teacherdashboard">
            <button className="my-4 bg-[#ffc42a] flex m-auto w-44 justify-center rounded-md p-2 text-[#00523F] font-bold">
              Login
            </button>
          </Link>
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
