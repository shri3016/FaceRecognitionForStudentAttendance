const Dept=require("../models/deptModel");

exports.addDept=async(req,res)=>{
    try{
        const newDept=new Dept(req.body);

        await newDept.save();

        res.status(201).json({
            message:"Dept added successfully"
        });
    }catch(err){
        res.status(500).json({
            message:"Error adding dept",
            error:err
        });
    }
}