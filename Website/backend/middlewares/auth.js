const jwt=require("jsonwebtoken");
require("dotenv").config();

exports.auth=(req,res,next)=>{
    try{
        const token=req.body.token;

        if(!token){
            return res.status(401).json({
                success:false,
                message:"Token missing"
            })
        }

        try{
            const decode=jwt.verify(token,process.env.JWT_SECRET);
            console.log(decode);
            req.user=decode;
        }catch(error){
            return res.status(402).json({
                success:false,
                messsage:"token is invalid",
            })
        }
        next();
    }catch(error){
        return res.status(401).json({
            success:false,
            message:"Something went wrong, while verifying the token",
        })
    }
}

exports.isTeachers=(req,res,next)=>{
    try{
        if(req.user.role!="Teacher"){
            return res.status(401).json({
                success:false,
                message:"This is protected route for Teacher",
            })
        }
        next();
    }catch(error){
        return res.status(501).json({
            success:false,
            message:"Userrole is not matching",
        })

    }
}

exports.isAdmin=(req,res,next)=>{
    try{
        if((req.user.role!=="Admin")){
            return res.status(401).json({
                success:false,
                message:"This is admin route",
            });
        }
        next();
    }catch(error){
        return res.status(500).json({
            success:false,
            message:"user role is not matching",
        })
    }
}