import React from "react";
import { BsFacebook } from "react-icons/bs";
import { AiFillInstagram } from "react-icons/ai";
import { AiFillTwitterCircle } from "react-icons/ai";
import { AiFillPhone } from "react-icons/ai";
import "../components/Styles/Footer.css";
function Footer() {
  return (
    <body className="footer">
      <div className="footer">
        <div className="socialMedia">
          <BsFacebook /> <AiFillInstagram /> <AiFillTwitterCircle />{" "}
          <AiFillPhone />
        </div>
        <p> &copy; 2023 TheLostOne.com</p>
      </div>
    </body>
  );
}

export default Footer;
