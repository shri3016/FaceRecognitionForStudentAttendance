const Subject=require("../models/subjectsModel");

exports.addSubject=async(req,res)=>{
    try{
        const newSubject=new Subject(req.body);

        await newSubject.save();

        res.status(201).json({
            message:"Added subject successfully"
        });
    }catch(err){
        res.status(500).json({
            message:'Error adding subject',
            error:err
        });
    }
}