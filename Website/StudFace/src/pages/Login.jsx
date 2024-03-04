import React from "react";
import { Link } from "react-router-dom";
import LoginBoy from "../assets/images/LoginPageboy.png";
import WelcomeLamp from "../assets/images/welcomelamp.png";
import welcomeLogo from "../assets/images/welcomelogo.png";

const Login = () => {
  return (
    <div className="min-h-screen bg-[#162544] flex flex-row">
      <div className="flex flex-col  relative w-[50%] justify-center space-y-5">
        <div className="flex justify-center">
          <img src={welcomeLogo} alt="welcome logo" className="w-[190px]" />
        </div>
        <div className="flex font-bold text-2xl justify-center">
          <h2 className="text-white">Welcome</h2>
        </div>
        <div className="flex flex-col space-y-2 w-[100%] justify-end">
          <Link to='/adminlogin'><button className="bg-[#ffc42a] flex m-auto w-72 justify-center rounded-md p-2 text-[#00523F] font-bold">Admin</button></Link>
          {/* <Link to='teacherlogin'><button className="bg-[#ffc42a] flex m-auto w-72 justify-center rounded-md p-2 text-[#00523F] font-bold">Teacher</button></Link> */}
        </div>
      </div>
      <div className="relative flex flex-col items-center justify-center w-[50%]">
        <div>
          <img src={WelcomeLamp} alt="lamp" className="w-36 absolute top-0 left-[10rem]" />
        </div>
        <div>
          <img src={LoginBoy} alt="boy" className="w-56" />
        </div>
      </div>
    </div>
  );
};

export default Login;
