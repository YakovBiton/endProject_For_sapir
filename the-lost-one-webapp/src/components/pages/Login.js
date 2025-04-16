// Login.js
import React, { useState, useContext } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "../../firebase"; // Adjusted this line
import { AuthContext } from "../../context/AuthContext"; // Path may vary
import { useNavigate } from "react-router-dom";
import "../Styles/Login.css";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const { setCurrentUser } = useContext(AuthContext);
  const navigate = useNavigate(); // Hooks must be inside component

  const login = async () => {
    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        email,
        password
      );
      const user = userCredential.user;
      console.log("User logged in successfully", user);
      // Set the user in AuthContext

      setCurrentUser({
        email: user.email,
        name: user.displayName,
        profileImage: user.photoURL || "url-to-image",
      }); // Temporarily using hardcoded name and image URL
      navigate("/FindChild"); // Redirect user to the home page
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <body className="login">
      <div className="boxStyle">
        <h1> Sign In</h1>
        <input
          type="email"
          s
          placeholder="Email"
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          onChange={(e) => setPassword(e.target.value)}
        />
        <button className="btn" onClick={login}>
          Log In
        </button>
      </div>
    </body>
  );
}

export default Login;
