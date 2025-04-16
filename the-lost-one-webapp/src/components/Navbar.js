import React, { useState, useContext } from "react";
import Logo from "../assets/logoPro.png";
import { Link, useNavigate } from "react-router-dom";
import { MdDensityMedium } from "react-icons/md";
import { AuthContext } from "../context/AuthContext"; // Path may vary
import "../components/Styles/Navbar.css";
import { BsFillPersonCheckFill } from "react-icons/bs";

function Navbar() {
  const [openLinks, setOpenLinks] = useState(false);
  const { currentUser, setCurrentUser } = useContext(AuthContext);
  const navigate = useNavigate();

  const toggleNavbar = () => {
    setOpenLinks(!openLinks);
  };

  const handleLogout = () => {
    // Call your firebase logout function here and after that:
    setCurrentUser(null);
    navigate("/login");
  };

  return (
    <div className="navbar">
      <div className="leftSide" id={openLinks ? "open" : "close"}>
        <img src={Logo} alt="logo" />
        <div className="hiddenLinks">
          <Link to="/"> Home </Link>
          {currentUser ? (
            <button onClick={handleLogout}>Logout</button>
          ) : (
            <Link to="/login"> Login/Sign Up </Link>
          )}
          <Link to="/about"> About Us </Link>
          <Link to="/contectus"> Contact Us </Link>
        </div>
      </div>

      <div className="rightSide">
        <Link to="/"> Home </Link>
        {currentUser ? (
          <>
            <Link to="/findchild">
              <BsFillPersonCheckFill />
            </Link>
            <span>Welcome, {currentUser.name}</span>
            <button onClick={handleLogout}>Logout</button>
          </>
        ) : (
          <Link to="/login"> Login/Sign Up </Link>
        )}
        <Link to="/about"> About Us </Link>
        <Link to="/contactus"> Contact Us </Link>
      </div>
    </div>
  );
}

export default Navbar;
