const Teacher=require("../models/addTeachersModel");

const addTeacher=async(req,res)=>{
    try{
        const newTeacher=new Teacher(req.body);

        await newTeacher.save();

        res.status(201).json({message:"Teacher added successfully"});
    }catch(err){
        res.status(500).json({
            message:"Error adding teacher",
            error:err
        })
    }
};

const getAllTeachers=async(req,res)=>{
    try{
        const teacher=await Teacher.find();
        res.send(students);
    }catch(error){
        res.status(500).send(error);
    }
};

const updateTeacher=async(req,res)=>{
    try{
        const {id}=req.params;
        const teacher=await Teacher.findByIdAndUpdate(id,req.body,{
            new:true,
        });
        if(!teacher){
            return res.status(404).send();
        }
        res.send(student);
    }catch(error){
        res.status(400).send(error);
    }

};

const deleteTeacher=async(req,res)=>{
    try{
        const {id}=req.params;
        const teacher=await Teacher.findByIdAndDelete(id);
        if(!teacher){
            return res.status(404).send();
        };
        res.send(teacher);
    }catch(error){
        res.status(500).send(error);
    }
}

module.exports={
    addTeacher,getAllTeachers,updateTeacher,deleteTeacher,
}