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
    router.post("/upload", upload.array('files', 12), home.uploadFiles)

    return app.use("/",router)
}
module.exports = InitRoutes;