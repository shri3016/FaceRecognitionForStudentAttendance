import React from "react";

const Navbar = ({ image, email,user }) => {
  return (
    <nav className="shadow-lg grid grid-cols-3 p-4">
      <div className="col-start-2 col-end-3 flex justify-center items-center font-bold text-white">
        {user}
      </div>
      <div className="col-start-3 col-end-4 flex justify-end items-center">
        <img src={image} alt="profile" />
        <span className="pl-2 text-white">{email}</span>
      </div>
    </nav>
  );
};

export default Navbar;
