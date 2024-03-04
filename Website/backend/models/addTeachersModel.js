// const mongoose=require("mongoose");

// const addTeachersSchema=new mongoose.Schema({
//     firstName:{
//         type:String,
//         required:true
//     },
//     lastName:{
//         type:String,
//         required:true
//     },
//     email:{
//         type:String,
//         required:true,
//         unique:true,
//         match:/^([\w-\.]+@([\w-]+\.)+[\w-]{2,4})?$/
//     },
//     password:{
//         type:String,
//         required:true,
//     },
//     phoneNumber:{
//         type:String,
//         match: /^[0-9]{10}$/
//     },
//     profilePicture:{
//         type:Buffer,
//         contentType:String
//     },
//     department:{
//         type:String,
//         required:true
//     },
// })

// module.exports=mongoose.model("AddTeachers", addTeachersSchema);