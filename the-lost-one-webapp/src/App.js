import "./App.css";
import SignUp from "./components/pages/SignUp"; // update this path to point to your SignUp.js file
import RegisterSuccess from "./components/pages/RegisterSuccess";
import Home from "./components/pages/home/Home";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import AboutUs from "./components/pages/AboutUs";
import ContactUs from "./components/pages/ContactUs";
import { AuthProvider } from "./context/AuthContext";
import Login from "./components/pages/Login";
import FindChild from "./components/pages/FindChild";
import UserPage from "./components/pages/UserPage";

function App() {
  return (
    <AuthProvider>
      <div className="App">
        <Router>
          <Navbar />
          <Routes>
            <Route path="/" exact element={<Home />} />
            <Route path="/about" exact element={<AboutUs />} />
            <Route path="/contactus" element={<ContactUs />} />
            <Route path="/register" element={<SignUp />} />
            <Route path="/register-success" element={<RegisterSuccess />} />
            <Route path="/Login" element={<Login />} />
            <Route path="/findchild" element={<FindChild />} />
            <Route path="/userpage" element={<UserPage />} />
          </Routes>
          <Footer />
        </Router>
      </div>
    </AuthProvider>
  );
}

export default App; //ok
