const mongoose=require("mongoose");

const subjectSchema=new mongoose.Schema({
    name:{
        type:String,
        required:true,
    },
    department:{
        type:mongoose.Schema.Types.ObjectId,
        ref:"Dept",
        required:true,
    },
    teacher:{
        type:mongoose.Schema.Types.ObjectId,
        ref:"AddTeachers",
        required:true,
    },
});

module.exports=mongoose.model("Subject",subjectSchema);