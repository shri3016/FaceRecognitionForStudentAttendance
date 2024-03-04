import React, { useState, useEffect } from 'react';

const EditTeachers = () => {
  const [teachers, setTeachers] = useState([]);

  useEffect(() => {
    fetchTeachers();
  }, []);

  const fetchTeachers = async () => {
    try {
      const response = await fetch('http://localhost:4000/api/v1/getallteachers');
      const data = await response.json();
      setTeachers(data.teachers); // Access the "teachers" array from the response
    } catch (error) {
      console.error(error);
    }
  };

  const handleDelete = async (id) => {
    try {
      await fetch(`/teacher/${id}`, {
        method: 'DELETE',
      });
      fetchTeachers();
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Teachers</h1>
      <table className="min-w-full border-collapse">
        <thead>
          <tr>
            <th className="py-2 px-4 border-b">First Name</th>
            <th className="py-2 px-4 border-b">Last Name</th>
            <th className="py-2 px-4 border-b">Email</th>
            <th className="py-2 px-4 border-b">Department</th>
            <th className="py-2 px-4 border-b">Subjects</th>
            <th className="py-2 px-4 border-b">Actions</th>
          </tr>
        </thead>
        <tbody>
          {teachers.map((teacher) => (
            <tr key={teacher._id}>
              <td className="py-2 px-4 border-b">{teacher.firstName}</td>
              <td className="py-2 px-4 border-b">{teacher.lastName}</td>
              <td className="py-2 px-4 border-b">{teacher.email}</td>
              <td className="py-2 px-4 border-b">{teacher.department}</td>
              <td className="py-2 px-4 border-b">{teacher.subjects.join(', ')}</td>
              <td className="py-2 px-4 border-b">
                <button className="mr-2 bg-green-500 hover:bg-green-600 text-white py-1 px-2 rounded" onClick={() => handleUpdate(teacher._id)}>
                  Update
                </button>
                <button className="mr-2 bg-red-500 hover:bg-red-600 text-white py-1 px-2 rounded" onClick={() => handleDelete(teacher._id)}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default EditTeachers;
