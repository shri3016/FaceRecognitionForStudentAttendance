import React from 'react';
import profilepic from '../assets/images/emailavatar.png';
import Navbar from '../components/Navbar';

const AdminDashboard = () => {
  return (
    <div className='min-h-screen bg-[rgb(22,37,68)]'>
     <Navbar user={`TEACHER`} email={`email@gmail.com`} image={profilepic}/>
      <div className='flex flex-col mt-[80px] font-bold'>
          <button className='text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg'>COMPUTER NETWORK</button>
          <button className='text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg'>DATA MINING</button>
          <button className='text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg'>CLOUD COMPUTING</button>
          <button className='text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg'>IMAGE PROCESSING</button>
          <button className='text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg'>COMPILER DESIGN</button>
          <button className='text-[#162544] m-2 bg-[#ffc42a] w-64 p-4 mx-auto rounded-lg'>MATHS</button>
      </div>
    </div>
  )
}

export default AdminDashboard;
