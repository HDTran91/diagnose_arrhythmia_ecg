import express from "express";
import {home} from "../controller/index";
import multer from 'multer'

let router = express.Router();
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });


/**
 * Init all routes
 */

let InitRoutes =(app) => {
    router.get("/", home.mainPage);
    router.post("/upload", router.post("/upload", upload.array('images', 12), home.uploadImages), home.uploadImages)

    return app.use("/",router)
}
module.exports = InitRoutes;