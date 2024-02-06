import express from "express";
import configViewEngine from "./config/viewEngine"
import initRoutes from "./routes/web"

let app = express();
let hostname = "localhost";
let port = 8017;

//config view engine
configViewEngine(app);

app.use(express.static("public"));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

//Init routes
initRoutes(app);

app.listen(port, hostname, ()=> {
  console.log(`Hello Hoang Tran,I am running at ${hostname}:${port}/` )
});
